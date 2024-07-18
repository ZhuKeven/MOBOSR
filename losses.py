import torch
import torch.nn.functional as F
from torch import nn


class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, sr, hr):
        sr_fft = torch.fft.rfft2(sr)
        hr_fft = torch.fft.rfft2(hr)

        sr_fft = torch.stack([sr_fft.real, sr_fft.imag], dim=-1)
        hr_fft = torch.stack([hr_fft.real, hr_fft.imag], dim=-1)

        return self.criterion(sr_fft, hr_fft)


class GradientLoss(nn.Module):
    def __init__(self, device):
        super(GradientLoss, self).__init__()
        self.sobel_filter_X = torch.tensor([[-1, -2, -1],
                                            [0, 0, 0],
                                            [1, 2, 1]],
                                           dtype=torch.float,
                                           requires_grad=False,
                                           device=device).view(1, 1, 3, 3)
        self.sobel_filter_Y = torch.tensor([[-1, 0, 1],
                                            [-2, 0, 2],
                                            [-1, 0, 1]],
                                           dtype=torch.float,
                                           requires_grad=False,
                                           device=device).view(1, 1, 3, 3)

    def forward(self, sr, hr):
        b, c, h, w = sr.size()

        output_X_c, output_Y_c = [], []
        gt_X_c, gt_Y_c = [], []
        for i in range(c):
            output_grad_X = F.conv2d(sr[:, i:i + 1, :, :], self.sobel_filter_X, bias=None, stride=1, padding=1)
            output_grad_Y = F.conv2d(sr[:, i:i + 1, :, :], self.sobel_filter_Y, bias=None, stride=1, padding=1)
            gt_grad_X = F.conv2d(hr[:, i:i + 1, :, :], self.sobel_filter_X, bias=None, stride=1, padding=1)
            gt_grad_Y = F.conv2d(hr[:, i:i + 1, :, :], self.sobel_filter_Y, bias=None, stride=1, padding=1)

            output_X_c.append(output_grad_X)
            output_Y_c.append(output_grad_Y)
            gt_X_c.append(gt_grad_X)
            gt_Y_c.append(gt_grad_Y)

        output_X = torch.cat(output_X_c, dim=1)
        output_Y = torch.cat(output_Y_c, dim=1)
        gt_X = torch.cat(gt_X_c, dim=1)
        gt_Y = torch.cat(gt_Y_c, dim=1)

        grad_loss = torch.mean(torch.abs(output_X - gt_X)) + torch.mean(torch.abs(output_Y - gt_Y))

        return grad_loss


class ConsistencyLoss(nn.Module):
    def __init__(self, scale, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale
        self.criterion = nn.L1Loss()

    def forward(self, sr, lr):
        return self.criterion(torch.nn.functional.interpolate(sr, scale_factor=1.0 / self.scale, mode='bicubic'), lr)
