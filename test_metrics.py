import os

import cv2
import lpips
import torch
from basicsr import rgb2ycbcr_pt
from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt

import utils

torch.set_grad_enabled(False)
lpips_model = lpips.LPIPS().cuda()


def eval(hr_path, lr_path, sr_path):
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    avg_lr_psnr = 0
    n = 0
    for fn in os.listdir(hr_path):
        lr_bgr = cv2.imread(os.path.join(lr_path, fn.replace(".png", "x4.png")))[1:-1, 1:-1]
        lr_rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
        lr = torch.from_numpy(lr_rgb).cuda().permute((2, 0, 1)) / 255.

        sr_bgr = cv2.imread(os.path.join(sr_path, fn))[4:-4, 4:-4]
        sr_rgb = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB)
        sr = torch.from_numpy(sr_rgb).cuda().permute((2, 0, 1)) / 255.

        hr_bgr = cv2.imread(os.path.join(hr_path, fn))[4:sr_bgr.shape[0] + 4, 4:sr_bgr.shape[1] + 4]
        hr_rgb = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB)
        hr = torch.from_numpy(hr_rgb).cuda().permute((2, 0, 1)) / 255.

        avg_lpips += lpips_model(sr, hr, normalize=True)

        lr = lr.unsqueeze(0)
        sr = sr.unsqueeze(0)
        hr = hr.unsqueeze(0)

        sr_y = rgb2ycbcr_pt(sr, y_only=True)
        hr_y = rgb2ycbcr_pt(hr, y_only=True)
        avg_psnr += calculate_psnr_pt(sr_y, hr_y, 0)
        avg_lr_psnr += calculate_psnr_pt(lr,
                                         utils.imresize(sr[0].cpu(), 1.0 / 4).unsqueeze(0).clamp(0, 1).cuda(),
                                         0,
                                         test_y_channel=True)
        avg_ssim += calculate_ssim_pt(sr_y, hr_y, 0)
        n += 1

    avg_psnr = float(avg_psnr) / n
    avg_ssim = float(avg_ssim) / n
    avg_lpips = float(avg_lpips) / n
    avg_lr_psnr = float(avg_lr_psnr) / n
    print("psnr:{:.4f}, ssim:{:.4f}, lpips:{:.4f}, lr_psnr:{:.4f}".format(
        avg_psnr, avg_ssim, avg_lpips, avg_lr_psnr))


eval("/mnt/data/datasets/Set5/HR", "/mnt/data/datasets/Set5/LR_bicubic/X4", "/mnt/data/sr_results/MOBOSR-c/Set5")
