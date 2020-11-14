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
class myDataset(Dataset):
    def __init__(self, src_root, transform=None):
        super(myDataset, self).__init__()
        self.src_root = src_root
        self.transform = transform
        self.src_path, self.label = self.get_path(src_root)

    def get_path(self, dir_src):
        src_path=[]
        label=[]
        cat_root=dir_src+'cat/'
        dog_root=dir_src+'dog/'
        for name in os.listdir(cat_root):
            src_path.append(cat_root+str(name))
            label.append(0)
        for name in os.listdir(dog_root):
            src_path.append(dog_root+str(name))
            label.append(1)
        return src_path, label

    def __len__(self):
        return len(self.src_path)

    def __getitem__(self, idx):
        src_path = self.src_path[idx]
        src_img=io.imread(src_path)
        src_img = self.transform(src_img)

        label=self.label[idx]
        return src_img, label