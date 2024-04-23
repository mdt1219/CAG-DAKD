from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SoftTarget(nn.Module):
	'''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''
	def __init__(self, T):
		super(SoftTarget, self).__init__()
		self.T = T

	def forward(self, out_s, out_t):
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
						F.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T

		return loss


class SoftTarget_Ours(nn.Module):

	def __init__(self, T1, T2):
		super().__init__()
		self.T1 = T1
		self.T2 = T2

	def forward(self, out_s, out_t, target):
		topk = (1,)
		maxk = max(topk)
		softmax_out_t = F.softmax(out_t, dim=1)
		max1, pred1 = softmax_out_t.topk(maxk, 1, True, True)
		out_t_argmax = torch.argmax(softmax_out_t, dim=1)
		mask = torch.eq(target, out_t_argmax).float()
		count = (mask[mask == 1]).size(0)
		mask = mask.unsqueeze(-1)
		batchsize = out_t.size(0)
		temperature = np.zeros_like(max1)
		for i in range(batchsize):
			if max1[i][0] > 0.4:
				temperature[i][0] = self.T1
			else:
				temperature[i][0] = self.T2

		soft_out_s = out_s.div(temperature)
		soft_out_t = out_t.div(temperature)
		out_s_softmax = F.log_softmax(soft_out_s, dim=1)
		out_t_softmax = F.softmax(soft_out_t, dim=1)
		correct_s = out_s_softmax.mul(mask)
		correct_t = out_t_softmax.mul(mask)
		correct_t[correct_t == 0.0] = 1.0

		loss = F.kl_div(correct_s, correct_t, reduction='sum') * (self.T1 ** 2) / count


		return loss

class SoftTarget_Ours1(nn.Module):

	def __init__(self, T_min, T_max):
		super().__init__()
		self.T_min = T_min
		self.T_max = T_max

	def forward(self, out_s, out_t, target):

		topk = (1,)
		maxk = max(topk)
		softmax_out_t = F.softmax(out_t, dim=1)
		max1, pred1 = softmax_out_t.topk(maxk, 1, True, True)
		out_t_argmax = torch.argmax(softmax_out_t, dim=1)
		mask = torch.eq(target, out_t_argmax).float()
		count = (mask[mask == 1]).size(0)
		mask = mask.unsqueeze(-1)
		batchsize = out_t.size(0)
		temperature = np.zeros_like(max1)
		for i in range(batchsize):
			if max1[i][0] > 0.4:
				temperature[i][0] = self.T_min
			else:
				temperature[i][0] = self.T_max

		soft_out_s = out_s.div(temperature)
		soft_out_t = out_t.div(temperature)
		out_s_softmax = F.log_softmax(soft_out_s, dim=1)
		out_t_softmax = F.softmax(soft_out_t, dim=1)
		correct_s = out_s_softmax.mul(mask)
		correct_t = out_t_softmax.mul(mask)
		correct_t[correct_t == 0.0] = 1.0

		loss = F.kl_div(correct_s, correct_t, reduction='sum') * (self.T1 ** 2) / count


		return loss