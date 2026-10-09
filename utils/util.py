import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch

from ldm.util import instantiate_from_config

import torchvision.transforms.functional as F

import numpy as np

import torchvision
from skimage.metrics import structural_similarity as ssim


# -------------------------
# 0) 小工具
# -------------------------
def to_minus1_1(x01):
    # x01: [0,1] -> [-1,1]
    return x01 * 2.0 - 1.0

def to_0_1(xm11):
    # [-1,1] -> [0,1]
    return (xm11 + 1.0) / 2.0

def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd#["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    model.cuda()
    model.train()
    return model

def save_image_grid(lq01, recon01, gt01, out_path, max_items=1):
    # lq/recon/gt: [B,3,H,W] in [0,1]
    b = min(lq01.shape[0], max_items)
    lq01, recon01, gt01 = lq01[:b], recon01[:b], gt01[:b]
    grid = torch.cat([lq01, recon01, gt01], dim=0)  # [3B,3,H,W]
    torchvision.utils.save_image(grid, out_path, nrow=b, padding=10)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

@torch.no_grad()
def calc_psnr(pred01, gt01, eps=1e-12):
    # pred01/gt01: [B,3,H,W] in [0,1]
    mse = torch.mean((pred01 - gt01) ** 2, dim=[1,2,3]).clamp_min(eps)
    return (10.0 * torch.log10(1.0 / mse)).mean().item()

@torch.no_grad()
def calc_ssim_batch(pred01, gt01):
    # skimage 走 numpy；逐张算
    # pred01/gt01: [B,3,H,W] in [0,1]
    pred = pred01.detach().cpu().permute(0,2,3,1).numpy()
    gt   = gt01.detach().cpu().permute(0,2,3,1).numpy()
    vals = []
    for i in range(pred.shape[0]):
        vals.append(ssim(gt[i], pred[i], channel_axis=2, data_range=1.0, win_size=7))
    return float(np.mean(vals))
