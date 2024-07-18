import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

dataset_paths = {
    "DIV2K_train": '/mnt/data/datasets/DIV2K_train',
    "DF2K": '/mnt/data/datasets/DF2K',
    "Set5": '/mnt/data/datasets/Set5',
    "Set14": '/mnt/data/datasets/Set14',
    "B100": '/mnt/data/datasets/BSD100',
    "U100": '/mnt/data/datasets/Urban100',
    "G100": '/mnt/data/datasets/General100',
    "M109": '/mnt/data/datasets/manga109',
    "DIV2K_valid": '/mnt/data/datasets/DIV2K_valid',
}


class TrainDataset(Dataset):
    def __init__(self, dataset_name, scale, patch_size, repeat):
        super(TrainDataset, self).__init__()
        self.hrs = []
        self.lrs = []

        path = dataset_paths[dataset_name]
        hr_path = os.path.join(path, 'HR')
        lr_path = os.path.join(path, 'LR_bicubic', 'X' + str(scale))
        for fn in os.listdir(hr_path):
            hr = cv2.imread(os.path.join(hr_path, fn))
            hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
            lr = cv2.imread(os.path.join(lr_path, fn.replace(".png", "x" + str(scale) + ".png")))
            lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
            self.hrs.append(hr)
            self.lrs.append(lr)
        self.patch_size = patch_size
        self.repeat = repeat
        self.scale = self.hrs[0].shape[0] // self.lrs[0].shape[0]

    def __len__(self):
        return len(self.hrs) * self.repeat

    def __getitem__(self, idx):
        idx %= len(self.lrs)
        lr = self.lrs[idx]
        hr = self.hrs[idx]

        lr_h, lr_w, _ = lr.shape
        lx, ly = random.randrange(0, lr_w - self.patch_size + 1), random.randrange(0, lr_h - self.patch_size + 1)
        hx, hy = lx * self.scale, ly * self.scale
        lr_patch = lr[ly:ly + self.patch_size, lx:lx + self.patch_size]
        hr_patch = hr[hy:hy + self.patch_size * self.scale, hx:hx + self.patch_size * self.scale]
        hflip = random.random() > 0.5
        vflip = random.random() > 0.5
        rot90 = random.random() > 0.5
        if hflip:
            lr_patch, hr_patch = lr_patch[:, ::-1, :], hr_patch[:, ::-1, :]
        if vflip:
            lr_patch, hr_patch = lr_patch[::-1, :, :], hr_patch[::-1, :, :]
        if rot90:
            lr_patch, hr_patch = lr_patch.transpose((1, 0, 2)), hr_patch.transpose((1, 0, 2))
        return (torch.from_numpy(np.ascontiguousarray(lr_patch)).float().permute((2, 0, 1)) / 255.,
                torch.from_numpy(np.ascontiguousarray(hr_patch)).float().permute((2, 0, 1)) / 255.)


class EvalDataset(Dataset):
    def __init__(self, dataset_name, scale):
        super(EvalDataset, self).__init__()
        self.hrs = []
        self.lrs = []

        path = dataset_paths[dataset_name]
        hr_path = os.path.join(path, 'HR')
        lr_path = os.path.join(path, 'LR_bicubic', 'X' + str(scale))
        for fn in os.listdir(hr_path):
            hr = cv2.imread(os.path.join(hr_path, fn))
            hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
            lr = cv2.imread(os.path.join(lr_path, fn.replace(".png", "x" + str(scale) + ".png")))
            lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
            self.hrs.append(hr)
            self.lrs.append(lr)
        self.scale = int(round(self.hrs[0].shape[0] / self.lrs[0].shape[0], 0))

    def __len__(self):
        return len(self.hrs)

    def __getitem__(self, idx):
        lr = self.lrs[idx]
        hr = self.hrs[idx]
        hr = hr[:lr.shape[0] * self.scale, :lr.shape[1] * self.scale]
        return (torch.from_numpy(np.ascontiguousarray(lr)).float().permute((2, 0, 1)) / 255.,
                torch.from_numpy(np.ascontiguousarray(hr)).float().permute((2, 0, 1)) / 255.)
