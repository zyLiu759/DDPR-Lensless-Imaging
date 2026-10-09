import os
from glob import glob
from skimage.metrics import peak_signal_noise_ratio,mean_squared_error
from natsort import natsorted
from PIL import Image
import numpy as np
import torch
import lpips
from pytorch_msssim import ssim 
LPIPS = lpips.LPIPS(net="alex")

tar_dir = r"gt"
inp_dir = r"recon"


inp_files = natsorted(glob(os.path.join(inp_dir, '*png')))
tar_files = natsorted(glob(os.path.join(tar_dir, '*png')))


lpips_sum = 0
psnr_sum = 0
ssim_sum = 0
mae_sum = 0.0
i = 0
for inp_file, tar_file in zip(inp_files, tar_files):
    # print('-------------------------------------'+'第{}张图片'.format(i+1)+'----------------------------------')
    tar_img = Image.open(tar_file)
    inp_img = Image.open(inp_file)
    inp_img = inp_img.resize((192, 192), Image.Resampling.LANCZOS)

    inp_img = np.array(inp_img).transpose(2, 0, 1)
    tar_img = np.array(tar_img).transpose(2, 0, 1)
    
    inp_tensor_01  = torch.from_numpy(inp_img.copy()).type(torch.FloatTensor).unsqueeze(0) / 255.0
    tar_tensor_01  = torch.from_numpy(tar_img.copy()).type(torch.FloatTensor).unsqueeze(0) / 255.0
    inp_tensor_lpips = inp_tensor_01 * 2.0 - 1.0
    tar_tensor_lpips = tar_tensor_01 * 2.0 - 1.0
    lpips = LPIPS(inp_tensor_lpips, tar_tensor_lpips).item()

    psnr = peak_signal_noise_ratio(tar_img, inp_img)

    ssims = ssim(inp_tensor_01, tar_tensor_01, data_range=1.0, size_average=True)

    mae = torch.mean(torch.abs(inp_tensor_01 - tar_tensor_01)).item()

    i = i + 1
    lpips_sum = lpips_sum + lpips
    psnr_sum = psnr_sum + psnr
    ssim_sum = ssim_sum + ssims
    mae_sum = mae_sum + mae
print('----------------------------------Average--------------------------------------')


mae_ave = mae_sum/i
lpips_ave = lpips_sum/i
psnr_ave = psnr_sum/i
ssim_ave = ssim_sum/i
print('MAE is :{:.4f}, LPIPS is :{:.4f}, PSNR is :{:.4f}, SSIM is :{:.4f}'.format(mae_ave, lpips_ave, psnr_ave, ssim_ave))