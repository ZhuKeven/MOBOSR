import os

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from torch.utils.data import DataLoader

from dataset import EvalDataset


class Inference:
    def __init__(self, scale=4, device=torch.device("cuda")):
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

    def inf(self, dir, dataset_name, model):
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.set_grad_enabled(False)
        test_loader = self.test_dataloaders[dataset_name]
        if test_loader is None:
            test_dataset = EvalDataset(dataset_name, self.scale)
            test_loader = DataLoader(dataset=test_dataset, pin_memory=True)
            self.test_dataloaders[dataset_name] = test_loader
        i = 0
        for lr, _, fn in test_loader:
            lr = lr.to(self.device)
            sr = model(lr).clamp(0, 1)

            sr_rgb = (sr[0] * 255.).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            cv2.imwrite(os.path.join(dir, fn[0]), cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR))
            i += 1


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
scale = 4
device = torch.device('cuda')
torch.set_grad_enabled(False)

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23)
model.eval()
model = model.to(device)
test_datasets = ["Set5", "Set14", "DIV2K_valid", "B100", "U100", "G100", "M109"]
inf = Inference(scale, device)
model_path = 'ckpts/all_loss/g_476.pth'
model.load_state_dict(torch.load(model_path), strict=True)
for name in test_datasets:
    inf.inf(os.path.join("/mnt/data/sr_results/MOBOSR-c", name), name, model)
