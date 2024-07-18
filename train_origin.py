import os
import sys
import time

import torch
from basicsr.archs.discriminator_arch import VGGStyleDiscriminator
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.losses.basic_loss import PerceptualLoss
from basicsr.losses.gan_loss import GANLoss
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import EvalDataset, TrainDataset
from evaluator import Evaluator
from utils import setup_seed, ModelEMA

setup_seed(1234)

batch_size = 16
scale = 4
epochs = 1000
exp_name = 'origin'
ckpt_path = os.path.join("ckpts", exp_name)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    device = torch.device('cuda')

    div2k_train = TrainDataset('DIV2K_train', scale, 32, repeat=batch_size)
    train_dataloader = DataLoader(dataset=div2k_train, num_workers=4, batch_size=batch_size, shuffle=True,
                                  pin_memory=True)

    div2k_valid = EvalDataset('DIV2K_valid', scale)
    div2k_valid_dataloader = DataLoader(dataset=div2k_valid, pin_memory=True)

    network_g = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23).to(device)
    network_g_ema = ModelEMA(network_g)
    network_d = VGGStyleDiscriminator(num_in_ch=3, num_feat=64).to(device)

    opt_g = optim.Adam(network_g.parameters(), 1e-4)
    opt_d = optim.Adam(network_d.parameters(), 1e-4)

    sch_g = torch.optim.lr_scheduler.StepLR(opt_g, 250, 0.5)
    sch_d = torch.optim.lr_scheduler.StepLR(opt_d, 250, 0.5)

    l1_loss_fn = nn.L1Loss()
    vgg_5_loss_fn = PerceptualLoss(layer_weights={'conv5_4': 1}, vgg_type='vgg19', use_input_norm=True,
                                   range_norm=False, perceptual_weight=1.0, style_weight=0, criterion='l1').to(device)
    gan_loss_fn = GANLoss(gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0, loss_weight=1).to(device)

    evaluator = Evaluator(device=device)

    for epoch in range(epochs):
        torch.set_grad_enabled(True)
        network_g.train()

        total_losses = 0
        l1_losses = 0
        vgg_5_losses = 0
        gan_losses = 0
        for lr, hr in train_dataloader:
            lr = lr.to(device)
            hr = hr.to(device)

            for p in network_d.parameters():
                p.requires_grad = False

            opt_g.zero_grad()
            sr = network_g(lr)
            l1_loss = l1_loss_fn(sr, hr)
            l1_losses += l1_loss.item()
            vgg_5_loss, _ = vgg_5_loss_fn(sr, hr)
            vgg_5_losses += vgg_5_loss.item()
            real_d_pred = network_d(hr).detach()
            fake_d_pred = network_d(sr)
            loss_g_real = gan_loss_fn(real_d_pred - torch.mean(fake_d_pred), False, is_disc=False)
            loss_g_fake = gan_loss_fn(fake_d_pred - torch.mean(real_d_pred), True, is_disc=False)
            gan_loss = (loss_g_real + loss_g_fake) / 2
            gan_losses += gan_loss.item()
            total_loss = l1_loss * 1e-2 + vgg_5_loss + gan_loss * 5e-3
            total_losses += total_loss.item()
            total_loss.backward()
            opt_g.step()
            network_g_ema.model_ema()

            for p in network_d.parameters():
                p.requires_grad = True

            opt_d.zero_grad()
            fake_d_pred = network_d(sr).detach()
            real_d_pred = network_d(hr)
            loss_d_real = gan_loss_fn(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
            loss_d_real.backward()
            fake_d_pred = network_d(sr.detach())
            loss_d_fake = gan_loss_fn(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
            loss_d_fake.backward()
            opt_d.step()
        sch_g.step()
        sch_d.step()
        total_losses /= len(train_dataloader)
        l1_losses /= len(train_dataloader)
        vgg_5_losses /= len(train_dataloader)
        gan_losses /= len(train_dataloader)
        print(time.asctime(),
              "epoch:{:3d}, losses: total:{:.4f}, l1:{:.4f}, vgg_5:{:.4f}, gan:{:.4f}".format(
                  epoch, total_losses, l1_losses, vgg_5_losses, gan_losses),
              flush=True)

        psnr, ssim, lpips, lr_psnr = evaluator.eval("epoch:{:3d}".format(epoch), "DIV2K_valid",
                                                    network_g)

        torch.save(network_g.state_dict(), os.path.join(ckpt_path, "g_{}.pth".format(epoch)))
        torch.save(network_d.state_dict(), os.path.join(ckpt_path, "d_{}.pth".format(epoch)))
