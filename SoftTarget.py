from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTarget(nn.Module):

    def __init__(self, T):
        super(SoftTarget, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='batchmean') * self.T * self.T

        return loss

class SoftTarget_none(nn.Module):

    def __init__(self, T):
        super(SoftTarget_none, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = (F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='none') * self.T * self.T).sum(-1)

        return loss
