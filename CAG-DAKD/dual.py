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
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dst

from utils import AverageMeter, accuracy, attention_map
from utils import load_pretrained_model, save_checkpoint
from utils import create_exp_dir, count_parameters_in_MB
from network import define_tsnet
from SoftTarget import SoftTarget, SoftTarget_none

parser = argparse.ArgumentParser(description='train kd')

parser.add_argument('--save_root', type=str, default='./results/ours/', help='results path')
parser.add_argument('--img_root', type=str, default='./datasets', help='dataset path')
parser.add_argument('--s_init', type=str, default='')
parser.add_argument('--t_model1', type=str, default='')
parser.add_argument('--t_model2', type=str, default='')
parser.add_argument('--print_freq', type=int, default=1, help='print frequency')
parser.add_argument('--epochs', type=int, default=200, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--num_class', type=int, default=100, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--note', type=str, default='', help='file name')
parser.add_argument('--data_name', type=str, default='cifar100', help='name of dataset')
parser.add_argument('--t1_name', type=str, default='', help='name of teacher network')
parser.add_argument('--t2_name', type=str, default='', help='name of teacher network')
parser.add_argument('--s_name', type=str, default='',help='name of student network')
parser.add_argument('--lambda_kd', type=float, default=0.1, help='trade-off parameter for kd loss')
parser.add_argument('--beta', type=float, default=10, help='trade-off parameter for attention distillation loss')
parser.add_argument('--T', type=float, default=2.0, help='temperature')
parser.add_argument("--weight_style", default="ADAPTIVE", type=str, choices=['AVERAGE', 'ADAPTIVE'])

args, unparsed = parser.parse_known_args()

args.save_root = os.path.join(args.save_root, args.note)
create_exp_dir(args.save_root)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

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
	snet = define_tsnet(name=args.s_name, num_class=args.num_class, cuda=args.cuda)
	checkpoint = torch.load(args.s_init)
	load_pretrained_model(snet, checkpoint['net'])
	logging.info('Student: %s', snet)
	logging.info('Student param size = %fMB', count_parameters_in_MB(snet))

	tnet1 = define_tsnet(name=args.t1_name, num_class=args.num_class, cuda=args.cuda)
	checkpoint = torch.load(args.t_model1)
	load_pretrained_model(tnet1, checkpoint['net'])
	tnet1.eval()
	for param in tnet1.parameters():
		param.requires_grad = False
	logging.info('Teacher: %s', tnet1)
	logging.info('Teacher param size = %fMB', count_parameters_in_MB(tnet1))
	logging.info('-----------------------------------------------')

	tnet2 = define_tsnet(name=args.t2_name, num_class=args.num_class, cuda=args.cuda)
	checkpoint = torch.load(args.t_model2)
	load_pretrained_model(tnet2, checkpoint['net'])
	tnet2.eval()
	for param in tnet2.parameters():
		param.requires_grad = False
	logging.info('Teacher: %s', tnet2)
	logging.info('Teacher param size = %fMB', count_parameters_in_MB(tnet2))
	logging.info('-----------------------------------------------')

	if args.cuda:
		criterionCls = torch.nn.CrossEntropyLoss().cuda()
	else:
		criterionCls = torch.nn.CrossEntropyLoss()

	criterionKD = SoftTarget(args.T)
	criterionKD_none = SoftTarget_none(args.T)

	optimizer = torch.optim.SGD(snet.parameters(),
								lr = args.lr,
								momentum = args.momentum,
								weight_decay = args.weight_decay,
								nesterov = True)

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
					download  = True),
			batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(
			dataset(root      = args.img_root,
					transform = test_transform,
					train     = False,
					download  = True),
			batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

	nets = {'snet':snet, 'tnet1':tnet1, 'tnet2':tnet2}
	criterions = {'criterionCls':criterionCls, 'criterionKD':criterionKD, 'criterionKD_none':criterionKD_none}

	best_top1 = 0
	best_top5 = 0
	for epoch in range(1, args.epochs+1):
		adjust_lr(optimizer, epoch)

		epoch_start_time = time.time()
		train(train_loader, nets, optimizer, criterions, epoch)

		logging.info('Testing the models......')
		test_top1, test_top5 = test(test_loader, nets, criterions, epoch)

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
			'snet': snet.state_dict(),
			'tnet1': tnet1.state_dict(),
			'tnet2': tnet2.state_dict(),
			'prec@1': test_top1,
			'prec@5': test_top5,
			'best@1': best_top1,
			'best@5': best_top5,
		}, is_best, args.save_root)

def train(train_loader, nets, optimizer, criterions, epoch):
	batch_time = AverageMeter()
	data_time  = AverageMeter()
	cls_losses = AverageMeter()
	kd_losses  = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	snet = nets['snet']
	tnet1 = nets['tnet1']
	tnet2 = nets['tnet2']

	criterionCls = criterions['criterionCls']
	criterionKD  = criterions['criterionKD']
	criterionKD_none = criterions['criterionKD_none']

	snet.train()
	end = time.time()
	for i, (img, target) in enumerate(train_loader, start=1):
		data_time.update(time.time() - end)

		if args.cuda:
			img = img.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)

		out_t_list = []
		attention1_s, attention2_s, attention3_s, out_s = snet(img)
		attention1_t1, attention2_t1, attention3_t1, out_t1 = tnet1(img)
		out_t2 = tnet2(img)
		out_t_list.append(out_t1)
		out_t_list.append(out_t2)
		cls_loss = criterionCls(out_s, target)
		if args.weight_style == "AVERAGE":
			kd_loss1 = criterionKD(out_s, out_t1.detach())
			kd_loss2 = criterionKD(out_s, out_t2.detach())
			kd_loss_list = []
			kd_loss_list.append(kd_loss1)
			kd_loss_list.append(kd_loss2)
			kd_loss = torch.stack(kd_loss_list).mean(0)
		if args.weight_style == "ADAPTIVE":
			device = torch.device('cuda:0')
			criterion_cls_lc = nn.CrossEntropyLoss(reduction='none')
			criterion_cls_lc1 = criterion_cls_lc(out_t1, target)
			criterion_cls_lc2 = criterion_cls_lc(out_t2, target)

			topk = (1,)
			maxk = max(topk)
			batch_size = target.size(0)
			softmax_out_t1 = F.softmax(out_t1, dim=1)
			softmax_out_t2 = F.softmax(out_t2, dim=1)
			_1, pred1 = softmax_out_t1.topk(maxk, 1, True, True)
			pred1 = pred1.t()
			correct1 = pred1.eq(target.view(1, -1).expand_as(pred1))
			_2, pred2 = softmax_out_t2.topk(maxk, 1, True, True)
			pred2 = pred2.t()
			correct2 = pred2.eq(target.view(1, -1).expand_as(pred2))
			weight = torch.zeros(2, batch_size).to(device)
			for mm in range(batch_size):
				if correct1[0, mm] == True and correct2[0, mm] == True:

					weight[0, mm] = 1 - float(criterion_cls_lc1[mm] / (criterion_cls_lc1[mm] + criterion_cls_lc2[mm]))
					weight[1, mm] = 1 - float(criterion_cls_lc2[mm] / (criterion_cls_lc1[mm] + criterion_cls_lc2[mm]))

				elif correct1[0, mm] or correct2[0, mm]:
					if correct1[0, mm]:
						weight[0, mm] = 0.0
						weight[1, mm] = 1.0
					else:
						weight[0, mm] = 0.0
						weight[1, mm] = 1.0

				else:
					weight[0, mm] = 0.0
					weight[1, mm] = 0.0
			weight = torch.where(torch.isnan(weight), torch.full_like(weight, 0), weight)
			loss_div_list = []
			soft_1 = criterionKD_none(out_s, out_t1)
			soft_2 = criterionKD_none(out_s, out_t2)
			loss_div_list.append(soft_1)
			loss_div_list.append(soft_2)
			loss_div = torch.stack(loss_div_list, dim=0)
			kd_loss = (torch.mul(weight, loss_div).sum()) / batch_size

		# 3.注意力蒸馏损失
		# ________________________modify-1.0 start________________________
		# s_attention1 = attention_map(out_s[0])
		# s_attention2 = attention_map(out_s[1])
		# s_attention3 = attention_map(out_s[2])
		# t_attention1 = attention_map(out_t1[0])
		# t_attention2 = attention_map(out_t1[1])
		# t_attention3 = attention_map(out_t1[2])
		# attention_distillation_1 = F.l1_loss(s_attention1, t_attention1)
		# attention_loss = attention_distillation_1
		# attention_distillation_2 = F.l1_loss(s_attention2, t_attention2)
		# attention_loss = attention_loss + attention_distillation_2
		# attention_distillation_3 = F.l1_loss(s_attention3, t_attention3)
		# attention_loss = attention_loss + attention_distillation_3
		# ________________________modify-1.0 ended________________________
		# ________________________modify-2.0 before________________________
		attention_loss1 = F.mse_loss(attention1_s, attention1_t1.detach())
		attention_loss2 = F.mse_loss(attention2_s, attention2_t1.detach())
		attention_loss3 = F.mse_loss(attention3_s, attention3_t1.detach())
		attention_loss = (attention_loss1 + attention_loss2 + attention_loss3) / 3.0
		# ________________________modify-2.0 ended________________________

		loss = (1 - args.lambda_kd) * cls_loss + kd_loss * args.lambda_kd + attention_loss * args.beta


		prec1, prec5 = accuracy(out_s, target, topk=(1,5))
		cls_losses.update(cls_loss.item(), img.size(0))
		kd_losses.update(kd_loss.item(), img.size(0))
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
					   'Cls:{cls_losses.val:.4f}({cls_losses.avg:.4f})  '
					   'KD:{kd_losses.val:.4f}({kd_losses.avg:.4f})  '
					   'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
					   'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
					   epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
					   cls_losses=cls_losses, kd_losses=kd_losses, top1=top1, top5=top5))
			logging.info(log_str)

def test(test_loader, nets, criterions, epoch):
	cls_losses = AverageMeter()
	kd_losses  = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	snet = nets['snet']
	tnet1 = nets['tnet1']
	tnet2 = nets['tnet2']

	criterionCls = criterions['criterionCls']
	criterionKD  = criterions['criterionKD']
	criterionKD_none = criterions['criterionKD_none']

	snet.eval()

	for i, (img, target) in enumerate(test_loader, start=1):
		if args.cuda:
			img = img.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)

		with torch.no_grad():
			out_s = snet(img)
			out_t1 = tnet1(img)
			out_t2 = tnet2(img)

		cls_loss = criterionCls(out_s, target)

		device = torch.device('cuda:0')
		criterion_cls_lc = nn.CrossEntropyLoss(reduction='none')
		criterion_cls_lc1 = criterion_cls_lc(out_t1, target)
		criterion_cls_lc2 = criterion_cls_lc(out_t2, target)

		topk = (1,)
		maxk = max(topk)
		batch_size = target.size(0)
		softmax_out_t1 = F.softmax(out_t1, dim=1)
		softmax_out_t2 = F.softmax(out_t2, dim=1)
		_1, pred1 = softmax_out_t1.topk(maxk, 1, True, True)
		pred1 = pred1.t()
		correct1 = pred1.eq(target.view(1, -1).expand_as(pred1))
		_2, pred2 = softmax_out_t2.topk(maxk, 1, True, True)
		pred2 = pred2.t()
		correct2 = pred2.eq(target.view(1, -1).expand_as(pred2))
		weight = torch.zeros(2, batch_size).to(device)
		for mm1 in range(batch_size):
			if correct1[0, mm1] == True and correct2[0, mm1] == True:

				weight[0, mm1] = 1 - float(criterion_cls_lc1[mm1] / (criterion_cls_lc1[mm1] + criterion_cls_lc2[mm1]))
				weight[1, mm1] = 1 - float(criterion_cls_lc2[mm1] / (criterion_cls_lc1[mm1] + criterion_cls_lc2[mm1]))

			elif correct1[0, mm1] or correct2[0, mm1]:
				if correct1[0, mm1]:
					weight[0, mm1] = 0.0
					weight[1, mm1] = 1.0
				else:
					weight[0, mm1] = 0.0
					weight[1, mm1] = 1.0

			else:
				weight[0, mm1] = 0.0
				weight[1, mm1] = 0.0

		loss_div_list = []
		loss_div_list.append(criterionKD_none(out_s, out_t1))
		loss_div_list.append(criterionKD_none(out_s, out_t2))
		loss_div = torch.stack(loss_div_list, dim=0)
		kd_loss = (torch.mul(weight, loss_div).sum())
		kd_loss = kd_loss * args.lambda_kd

		prec1, prec5 = accuracy(out_s, target, topk=(1,5))
		cls_losses.update(cls_loss.item(), img.size(0))
		kd_losses.update(kd_loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

	f_l = [cls_losses.avg, kd_losses.avg, top1.avg, top5.avg]
	logging.info('Cls: {:.4f}, KD: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))
	best_Acc = 0.0
	if top1.avg >= best_Acc:
		best_Acc = top1.avg
		print('Current best accuracy (top-1): %.2f%%' % (best_Acc))
	else:
		best_Acc = best_Acc
		print('Current best accuracy (top-1): %.2f%%' % (best_Acc))
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
