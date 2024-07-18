import os
import sys
import time

import lpips
import pytorch_msssim
import torch
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from basicsr.archs.discriminator_arch import VGGStyleDiscriminator
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.losses.basic_loss import PerceptualLoss
from basicsr.losses.gan_loss import GANLoss
from torch import nn, optim
from torch.utils.data import DataLoader

import losses
from dataset import TrainDataset
from evaluator import Evaluator
from utils import setup_seed

setup_seed(1234)

batch_size = 16
scale = 4
epochs = 1000
exp_name = 'all_loss'
ckpt_path = os.path.join("ckpts", exp_name)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    device = torch.device('cuda')

    parameters = [
        {
            'name': 'w_l1',
            'type': 'range',
            'bounds': [0, 1],
            'value_type': 'float'
        },
        {
            'name': 'w_l2',
            'type': 'range',
            'bounds': [0, 1],
            'value_type': 'float'
        },
        {
            'name': 'w_vgg_2',
            'type': 'range',
            'bounds': [0, 1],
            'value_type': 'float'
        },
        {
            'name': 'w_vgg_3',
            'type': 'range',
            'bounds': [0, 1],
            'value_type': 'float'
        },
        {
            'name': 'w_vgg_4',
            'type': 'range',
            'bounds': [0, 1],
            'value_type': 'float'
        },
        {
            'name': 'w_vgg_5',
            'type': 'range',
            'bounds': [0, 1],
            'value_type': 'float'
        },
        {
            'name': 'w_gan',
            'type': 'range',
            'bounds': [0, 1],
            'value_type': 'float'
        },
        {
            'name': 'w_ssim',
            'type': 'range',
            'bounds': [0, 1],
            'value_type': 'float'
        },
        {
            'name': 'w_gradient',
            'type': 'range',
            'bounds': [0, 1],
            'value_type': 'float'
        },
        {
            'name': 'w_fft',
            'type': 'range',
            'bounds': [0, 1],
            'value_type': 'float'
        },
        {
            'name': 'w_lpips_vgg',
            'type': 'range',
            'bounds': [0, 1],
            'value_type': 'float'
        },
        {
            'name': 'w_consistency',
            'type': 'range',
            'bounds': [0, 1],
            'value_type': 'float'
        },
    ]

    objectives = {
        "psnr": ObjectiveProperties(minimize=False),
        "lpips": ObjectiveProperties(minimize=True)
    }

    ax_client = AxClient()
    ax_client.create_experiment(name="sr", parameters=parameters, objectives=objectives)

    div2k_train = TrainDataset('DIV2K_train', scale, 32, repeat=batch_size)
    train_dataloader = DataLoader(dataset=div2k_train, num_workers=4, batch_size=batch_size, shuffle=True,
                                  pin_memory=True)

    network_g = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23).to(device)
    network_g.load_state_dict(torch.load("ckpts/origin/g_250.pth"))
    network_d = VGGStyleDiscriminator(num_in_ch=3, num_feat=64).to(device)
    network_d.load_state_dict(torch.load("ckpts/origin/d_250.pth"))

    opt_g = optim.Adam(network_g.parameters(), 5e-5)
    opt_d = optim.Adam(network_d.parameters(), 5e-5)

    sch_g = torch.optim.lr_scheduler.StepLR(opt_g, 250, 0.5)
    sch_d = torch.optim.lr_scheduler.StepLR(opt_d, 250, 0.5)

    l1_loss_fn = nn.L1Loss()
    l2_loss_fn = nn.MSELoss()
    vgg_2_loss_fn = PerceptualLoss(layer_weights={'conv2_2': 1}, vgg_type='vgg19', use_input_norm=True,
                                   range_norm=False, perceptual_weight=1.0, style_weight=0, criterion='l1').to(device)
    vgg_3_loss_fn = PerceptualLoss(layer_weights={'conv3_4': 1}, vgg_type='vgg19', use_input_norm=True,
                                   range_norm=False, perceptual_weight=1.0, style_weight=0, criterion='l1').to(device)
    vgg_4_loss_fn = PerceptualLoss(layer_weights={'conv4_4': 1}, vgg_type='vgg19', use_input_norm=True,
                                   range_norm=False, perceptual_weight=1.0, style_weight=0, criterion='l1').to(device)
    vgg_5_loss_fn = PerceptualLoss(layer_weights={'conv5_4': 1}, vgg_type='vgg19', use_input_norm=True,
                                   range_norm=False, perceptual_weight=1.0, style_weight=0, criterion='l1').to(device)
    gan_loss_fn = GANLoss(gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0).to(device)
    ssim_loss_fn = pytorch_msssim.SSIM().to(device)
    fft_loss_fn = losses.FFTLoss().to(device)
    gradient_loss_fn = losses.GradientLoss(device).to(device)
    consistency_loss_fn = losses.ConsistencyLoss(scale=scale).to(device)

    lpips_model = lpips.LPIPS().to(device)

    evaluator = Evaluator(device=device)

    for epoch in range(epochs):
        torch.set_grad_enabled(True)
        network_g.train()

        parameters, trial_index = ax_client.get_next_trial()
        (w_l1, w_l2, w_vgg_2, w_vgg_3, w_vgg_4, w_vgg_5, w_gan, w_ssim, w_gradient, w_fft, w_lpips_vgg,
         w_consistency) = (parameters['w_l1'], parameters['w_l2'], parameters['w_vgg_2'], parameters['w_vgg_3'],
                           parameters['w_vgg_4'], parameters['w_vgg_5'], parameters['w_gan'], parameters['w_ssim'],
                           parameters['w_gradient'], parameters['w_fft'], parameters['w_lpips_vgg'],
                           parameters['w_consistency'])
        print(time.asctime(),
              "epoch:{:3d}, weights: l1:{:.4f}, l2:{:.4f}, vgg_2:{:.4f}, vgg_3:{:.4f}, vgg_4:{:.4f}, vgg_5:{:.4f}, \
gan:{:.4f}, ssim:{:.4f}, gradient:{:.4f}, fft:{:.4f}, lpips_vgg:{:.4f}, consistency:{:.4f}".format(
                  epoch, w_l1, w_l2, w_vgg_2, w_vgg_3, w_vgg_4, w_vgg_5, w_gan, w_ssim, w_gradient, w_fft,
                  w_lpips_vgg, w_consistency),
              flush=True)

        total_losses = 0
        l1_losses = 0
        l2_losses = 0
        vgg_2_losses = 0
        vgg_3_losses = 0
        vgg_4_losses = 0
        vgg_5_losses = 0
        gan_losses = 0
        ssim_losses = 0
        gradient_losses = 0
        fft_losses = 0
        lpips_vgg_losses = 0
        consistency_losses = 0

        for lr, hr in train_dataloader:
            lr = lr.to(device)
            hr = hr.to(device)

            for p in network_d.parameters():
                p.requires_grad = False

            opt_g.zero_grad()
            sr = network_g(lr)

            l1_loss = l1_loss_fn(sr, hr)
            l1_losses += l1_loss.item()

            l2_loss = l2_loss_fn(sr, hr)
            l2_losses += l2_loss.item()

            vgg_2_loss, _ = vgg_2_loss_fn(sr, hr)
            vgg_2_losses += vgg_2_loss.item()

            vgg_3_loss, _ = vgg_3_loss_fn(sr, hr)
            vgg_3_losses += vgg_3_loss.item()

            vgg_4_loss, _ = vgg_4_loss_fn(sr, hr)
            vgg_4_losses += vgg_4_loss.item()

            vgg_5_loss, _ = vgg_5_loss_fn(sr, hr)
            vgg_5_losses += vgg_5_loss.item()

            real_d_pred = network_d(hr).detach()
            fake_d_pred = network_d(sr)
            loss_g_real = gan_loss_fn(real_d_pred - torch.mean(fake_d_pred), False, is_disc=False)
            loss_g_fake = gan_loss_fn(fake_d_pred - torch.mean(real_d_pred), True, is_disc=False)
            gan_loss = (loss_g_real + loss_g_fake) / 2
            gan_losses += gan_loss.item()

            ssim_loss = -ssim_loss_fn(sr, hr)
            ssim_losses += ssim_loss.item()

            fft_loss = fft_loss_fn(sr, hr)
            fft_losses += fft_loss.item()

            gradient_loss = gradient_loss_fn(sr, hr)
            gradient_losses += gradient_loss.item()

            lpips_vgg_loss = torch.mean(lpips_model(sr, hr, normalize=True))
            lpips_vgg_losses += lpips_vgg_loss.item()

            consistency_loss = consistency_loss_fn(sr, lr)
            consistency_losses += consistency_loss.item()

            total_loss = l1_loss * w_l1 + l2_loss * w_l2 + vgg_2_loss * w_vgg_2 + vgg_3_loss * w_vgg_3 + \
                         vgg_4_loss * w_vgg_4 + vgg_5_loss * w_vgg_5 + gan_loss * w_gan + ssim_loss * w_ssim + \
                         fft_loss * w_fft + gradient_loss * w_gradient + lpips_vgg_loss * w_lpips_vgg + \
                         consistency_loss * w_consistency
            total_losses += total_loss.item()
            total_loss.backward()
            opt_g.step()

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
        l2_losses /= len(train_dataloader)
        vgg_2_losses /= len(train_dataloader)
        vgg_3_losses /= len(train_dataloader)
        vgg_4_losses /= len(train_dataloader)
        vgg_5_losses /= len(train_dataloader)
        gan_losses /= len(train_dataloader)
        ssim_losses /= len(train_dataloader)
        gradient_losses /= len(train_dataloader)
        fft_losses /= len(train_dataloader)
        lpips_vgg_losses /= len(train_dataloader)
        consistency_losses /= len(train_dataloader)
        print(time.asctime(),
              "epoch:{:3d}, losses: total:{:.4f}, l1:{:.4f}, l2:{:.4f}, vgg_2:{:.4f}, vgg_3:{:.4f}, vgg_4:{:.4f}, \
vgg_5:{:.4f}, gan:{:.4f}, ssim:{:.4f}, gradient:{:.4f}, fft:{:.4f}, lpips_vgg:{:.4f}, consistency:{:.4f}".format(
                  epoch, total_losses, l1_losses, l2_losses, vgg_2_losses, vgg_3_losses, vgg_4_losses, vgg_5_losses,
                  gan_losses, ssim_losses, gradient_losses, fft_losses, lpips_vgg_losses, consistency_losses),
              flush=True)

        psnr, ssim, lpips, lr_psnr = evaluator.eval("epoch:{:3d}".format(epoch), "DIV2K_valid",
                                                    network_g)

        torch.save(network_g.state_dict(), os.path.join(ckpt_path, "g_{}.pth".format(epoch)))
        torch.save(network_d.state_dict(), os.path.join(ckpt_path, "d_{}.pth".format(epoch)))

        ax_client.complete_trial(trial_index=trial_index, raw_data={
            "psnr": psnr,
            "lpips": lpips
        })
        ax_client.save_to_json_file(os.path.join(ckpt_path, "o_{}.pth".format(epoch)))
