from os.path import splitext
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
import os
import random
from PIL import Image
from skimage import io
from cv2 import cv2 as cv
class teacher_dataset(Dataset):
    def __init__(self, src_root, T, transform=None):
        super(teacher_dataset, self).__init__()
        self.src_root = src_root
        self.transform = transform
        self.T=T
        self.src_path, self.label ,self.label_onehot= self.get_path(src_root)
        

    def get_path(self, dir_src):
        src_path=[]
        label=[]
        label_onehot=[]
        txt=open('preds.txt')
        TXT=txt.readlines()
        cat_root=dir_src+'cat/'
        dog_root=dir_src+'dog/'
        for name in os.listdir(cat_root):
            src_path.append(cat_root+str(name))
            label_one=[float(TXT[int(str(name).split('.')[1])].split(',')[0])/self.T,float(TXT[int(str(name).split('.')[1])].split(',')[1])/self.T]
            label.append(label_one)
            label_onehot.append(0)
        for name in os.listdir(dog_root):
            src_path.append(dog_root+str(name))
            label_one=[float(TXT[int(str(name).split('.')[1])+len(os.listdir(cat_root))].split(',')[0])/self.T,float(TXT[int(str(name).split('.')[1])+len(os.listdir(cat_root))].split(',')[1])/self.T]
            label.append(label_one)
            label_onehot.append(1)
        return src_path, label,label_onehot

    def __len__(self):
        return len(self.src_path)

    def __getitem__(self, idx):
        src_path = self.src_path[idx]
        src_img=io.imread(src_path)
        src_img = self.transform(src_img)

        label=self.label[idx]
        label_onehot=self.label_onehot[idx]
        return src_img, label,label_onehot