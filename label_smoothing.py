import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from torch import nn
import numpy as np
import scipy
import os
import torch.nn.functional as F

class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.1,
                 ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        n_valid = label.size(0)
        lb_one_hot = label.clone()
        smooth_label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)
        smooth_label=smooth_label.unsqueeze(1)
        smooth_label=torch.cat((1-smooth_label,smooth_label),1)

        loss = -torch.sum(torch.sum(logs*smooth_label, dim=1)) / n_valid
        return loss