import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from torch import nn
import numpy as np

def resnet18_bcd():
    model = models.resnet18(pretrained=False)
    model.fc=nn.Linear(512, 2)
    model.conv1=nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    )

    model.layer2[0].downsample=nn.Sequential(
        nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
        nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    model.layer2[0].conv1=nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.layer2[0].conv2=nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    model.layer3[0].downsample=nn.Sequential(
        nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
        nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    model.layer3[0].conv1=nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.layer3[0].conv2=nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    model.layer4[0].downsample=nn.Sequential(
        nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
        nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    model.layer4[0].conv1=nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.layer4[0].conv2=nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    
    return model


def resnet50_bcd():
    model = models.resnet50(pretrained=False)
    model.fc=nn.Sequential(
        nn.Linear(2048, 2)
    )

    model.conv1=nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    )

    model.layer2[0].downsample=nn.Sequential(
        nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
        nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    model.layer3[0].downsample=nn.Sequential(
        nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
        nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    model.layer4[0].downsample=nn.Sequential(
        nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
        nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )

    return model