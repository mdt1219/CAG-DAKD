from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import logging
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dst

from utils import AverageMeter, accuracy
from utils import save_checkpoint
from utils import create_exp_dir, count_parameters_in_MB
from network import define_tsnet

parser = argparse.ArgumentParser(description='train teacher and student network')

parser.add_argument('--save_root', type=str, default='./results/base/', help='result')
parser.add_argument('--img_root', type=str, default='./datasets', help='dataset path')

parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
parser.add_argument('--epochs', type=int, default=200, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--num_class', type=int, default=100, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--note', type=str, default='new-base-c100-resnet20-gs', help='file name')

parser.add_argument('--data_name', type=str, default='cifar100', help='dataset')
parser.add_argument('--net_name', type=str, default='resnet20', help='net_name')

args, unparsed = parser.parse_known_args()

args.save_root = os.path.join(args.save_root, args.note)
create_exp_dir(args.save_root)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

device = torch.device("cuda:0")

def main():
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)
		cudnn.enabled = True
		cudnn.benchmark = True
	logging.info("args = %s", args)
	logging.info("unparsed_args = %s", unparsed)
	logging.info('----------- Network Initialization --------------')
	net = define_tsnet(name=args.net_name, num_class=args.num_class, cuda=args.cuda)
	logging.info('%s', net)
	logging.info("param size = %fMB", count_parameters_in_MB(net))
	logging.info('-----------------------------------------------')
	logging.info('Saving initial parameters......') 
	save_path = os.path.join(args.save_root, 'initial_r{}.pth.tar'.format(args.net_name[6:]))
	torch.save({
		'epoch': 0,
		'net': net.state_dict(),
		'prec@1': 0.0,
		'prec@5': 0.0,
	}, save_path)

	optimizer = torch.optim.SGD(net.parameters(),
								lr = args.lr, 
								momentum = args.momentum, 
								weight_decay = args.weight_decay,
								nesterov = True)
	if args.cuda:
		criterion = torch.nn.CrossEntropyLoss().cuda()
	else:
		criterion = torch.nn.CrossEntropyLoss()

	if args.data_name == 'cifar10':
		dataset = dst.CIFAR10
		mean = (0.4914, 0.4822, 0.4465)
		std  = (0.2470, 0.2435, 0.2616)
	elif args.data_name == 'cifar100':
		dataset = dst.CIFAR100
		mean = (0.5071, 0.4865, 0.4409)
		std  = (0.2673, 0.2564, 0.2762)
	else:
		raise Exception('Invalid dataset name...')

	train_transform = transforms.Compose([
			transforms.Pad(4, padding_mode='reflect'),
			transforms.RandomCrop(32),
			transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
			transforms.RandomHorizontalFlip(),	
			transforms.ToTensor(),
			transforms.Normalize(mean=mean,std=std)
		])
	test_transform = transforms.Compose([
			transforms.CenterCrop(32),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean,std=std)
		])

	train_loader = torch.utils.data.DataLoader(
			dataset(root      = args.img_root,
					transform = train_transform,
					train     = True,
					download  = False),
			batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(
			dataset(root      = args.img_root,
					transform = test_transform,
					train     = False,
					download  = False),
			batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

	best_top1 = 0
	best_top5 = 0
	for epoch in range(1, args.epochs+1):
		adjust_lr(optimizer, epoch)

		epoch_start_time = time.time()
		train(train_loader, net, optimizer, criterion, epoch)

		logging.info('Testing the models......')
		test_top1, test_top5 = test(test_loader, net, criterion)

		epoch_duration = time.time() - epoch_start_time
		logging.info('Epoch time: {}s'.format(int(epoch_duration)))

		# save model
		is_best = False
		if test_top1 > best_top1:
			best_top1 = test_top1
			best_top5 = test_top5
			is_best = True
		logging.info('Saving models......')
		save_checkpoint({
			'epoch': epoch,
			'net': net.state_dict(),
			'prec@1': test_top1,
			'prec@5': test_top5,
			'best@1': best_top1,
			'best@5': best_top5,
		}, is_best, args.save_root)


def train(train_loader, net, optimizer, criterion, epoch):
	batch_time = AverageMeter()
	data_time  = AverageMeter()
	losses     = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	net.train()

	end = time.time()
	for i, (img, target) in enumerate(train_loader, start=1):
		data_time.update(time.time() - end)

		if args.cuda:
			img = img.cuda(non_blocking=False)
			target = target.cuda(non_blocking=False)

		_, _, _, out = net(img)
		loss = criterion(out, target)

		prec1, prec5 = accuracy(out, target, topk=(1,5))
		losses.update(loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
					   'Time:{batch_time.val:.4f} '
					   'Data:{data_time.val:.4f}  '
					   'loss:{losses.val:.4f}({losses.avg:.4f})  '
					   'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
					   'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
					   epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
					   losses=losses, top1=top1, top5=top5))
			logging.info(log_str)


def test(test_loader, net, criterion):
	losses = AverageMeter()
	top1   = AverageMeter()
	top5   = AverageMeter()

	net.eval()

	end = time.time()
	for i, (img, target) in enumerate(test_loader, start=1):
		if args.cuda:
			img = img.cuda(non_blocking=False)
			target = target.cuda(non_blocking=False)

		with torch.no_grad():
			_, _, _, out = net(img)
			loss = criterion(out, target)

		prec1, prec5 = accuracy(out, target, topk=(1,5))
		losses.update(loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

	f_l = [losses.avg, top1.avg, top5.avg]
	logging.info('Loss: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))

	return top1.avg, top5.avg


def adjust_lr(optimizer, epoch):
	scale   = 0.1
	lr_list =  [args.lr] * 100
	lr_list += [args.lr*scale] * 50
	lr_list += [args.lr*scale*scale] * 50

	lr = lr_list[epoch-1]
	logging.info('Epoch: {}  lr: {:.3f}'.format(epoch, lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


if __name__ == '__main__':
	main()