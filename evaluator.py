import time

import lpips
import torch
from basicsr import rgb2ycbcr_pt
from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt
from torch.utils.data import DataLoader

from dataset import EvalDataset
from utils import imresize


class Evaluator:
    def __init__(self, scale=4, device=torch.device("cuda")):
        self.lpips_model = lpips.LPIPS().to(device)
        self.scale = scale
        self.device = device
        self.test_dataloaders = {
            "Set5": None,
            "Set14": None,
            "B100": None,
            "U100": None,
            "G100": None,
            "M109": None,
            "DIV2K_valid": None,
        }

    def eval(self, prefix, dataset_name, model):
        torch.set_grad_enabled(False)
        avg_psnr = 0
        avg_ssim = 0
        avg_lpips = 0
        avg_lr_psnr = 0
        test_loader = self.test_dataloaders[dataset_name]
        if test_loader is None:
            test_dataset = EvalDataset(dataset_name, self.scale)
            test_loader = DataLoader(dataset=test_dataset, pin_memory=True)
            self.test_dataloaders[dataset_name] = test_loader
        for lr, hr, _ in test_loader:
            lr = lr.to(self.device)
            hr = hr.to(self.device)
            sr = model(lr).clamp(0, 1)

            avg_lpips += self.lpips_model(sr[0], hr[0], normalize=True)

            lr = lr[:, :, 1:-1, 1:-1]
            sr = sr[:, :, self.scale:-self.scale, self.scale:-self.scale]
            hr = hr[:, :, self.scale:-self.scale, self.scale:-self.scale]
            sr_y = rgb2ycbcr_pt(sr, y_only=True)
            hr_y = rgb2ycbcr_pt(hr, y_only=True)
            avg_psnr += calculate_psnr_pt(sr_y, hr_y, 0)
            avg_lr_psnr += calculate_psnr_pt(lr,
                                             imresize(sr[0].cpu(), 1.0 / self.scale).unsqueeze(0).clamp(0,
                                                                                                        1).cuda(),
                                             0,
                                             test_y_channel=True)
            avg_ssim += calculate_ssim_pt(sr_y, hr_y, 0)

        avg_psnr = float(avg_psnr) / len(test_loader)
        avg_ssim = float(avg_ssim) / len(test_loader)
        avg_lpips = float(avg_lpips) / len(test_loader)
        avg_lr_psnr = float(avg_lr_psnr) / len(test_loader)
        print(time.asctime(),
              "{}, psnr:{:.4f}, ssim:{:.4f}, lpips:{:.4f}, lr_psnr:{:.4f}".format(
                  prefix, avg_psnr, avg_ssim, avg_lpips, avg_lr_psnr),
              flush=True)
        return avg_psnr, avg_ssim, avg_lpips, avg_lr_psnr
