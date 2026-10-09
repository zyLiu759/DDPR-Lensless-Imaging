import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import random
import glob
import natsort




class PairResizePad:
    """
    Pair transform: keep-aspect resize (fit-in) + symmetric pad to fixed (H,W),
    applied identically to (img_a, img_b).

    - Works with PIL.Image input (recommended, since you use PairToTensor later).
    - Padding supports: "reflect", "edge" (replicate), "constant".
    - Returns PIL.Image for both.
    """
    def __init__(
        self,
        out_hw=(256, 384),          # (H, W)
        interp=Image.BICUBIC,
        pad_mode="reflect",         # "reflect" | "edge" | "constant"
        pad_value=0,                # used if pad_mode == "constant"
    ):
        self.out_h, self.out_w = int(out_hw[0]), int(out_hw[1])
        self.interp = interp
        self.pad_mode = pad_mode
        self.pad_value = pad_value

    def __call__(self, img_a, img_b):
        img_a = self._to_pil(img_a)
        img_b = self._to_pil(img_b)

        # must be spatially aligned before transform
        if img_a.size != img_b.size:  # PIL size is (W,H)
            raise ValueError(f"PairResizePad expects aligned pair, got {img_a.size} vs {img_b.size}")

        img_a = self._resize_keep_ar(img_a)
        img_b = self._resize_keep_ar(img_b)

        img_a = self._pad_to_target(img_a)
        img_b = self._pad_to_target(img_b)
        return img_a, img_b

    @staticmethod
    def _to_pil(x):
        if isinstance(x, Image.Image):
            return x
        if isinstance(x, np.ndarray):
            return Image.fromarray(x)
        raise TypeError(f"Unsupported type: {type(x)}. Please pass PIL.Image or np.ndarray.")

    def _resize_keep_ar(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = min(self.out_h / h, self.out_w / w)  # fit inside target
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        if (new_w, new_h) == (w, h):
            return img
        return img.resize((new_w, new_h), resample=self.interp)

    def _pad_to_target(self, img: Image.Image) -> Image.Image:
        # Convert to numpy for reflect/edge padding
        arr = np.array(img)
        if arr.ndim == 2:
            arr = arr[:, :, None]  # H,W,1

        h, w, c = arr.shape
        pad_h = max(0, self.out_h - h)
        pad_w = max(0, self.out_w - w)

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        if pad_h == 0 and pad_w == 0:
            out = arr
        else:
            if self.pad_mode == "reflect":
                mode = "reflect"
                out = np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode=mode)
            elif self.pad_mode == "edge":
                mode = "edge"
                out = np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode=mode)
            elif self.pad_mode == "constant":
                out = np.pad(
                    arr,
                    ((top, bottom), (left, right), (0, 0)),
                    mode="constant",
                    constant_values=self.pad_value,
                )
            else:
                raise ValueError(f"Unknown pad_mode: {self.pad_mode}")

        # Safety: ensure exact size
        out = out[: self.out_h, : self.out_w, :]

        if out.shape[2] == 1:
            out = out[:, :, 0]
        return Image.fromarray(out)
    
class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label
    
class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label)
    
class PairCompose(transforms.Compose):
    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label
    
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dir, filelist=None, train=True):
        super().__init__()

        self.input_names = natsort.natsorted(glob.glob(os.path.join(dir, filelist,'low', '*.png')))
        self.groundtruth_names = natsort.natsorted(glob.glob(os.path.join(dir, filelist, 'high', '*.png')))

        if train:
            self.transforms = PairCompose([
                PairResizePad(out_hw=(384,384), pad_mode="reflect"),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ])
            self.input_names = self.input_names
            self.groundtruth_names = self.groundtruth_names
        else:
            self.transforms = PairCompose([
                PairResizePad(out_hw=(384,384), pad_mode="reflect"),
                # PairRandomCrop(self.patch_size),
                PairToTensor()
            ])
            self.input_names = self.input_names[:100]
            self.groundtruth_names = self.groundtruth_names[:100]

    def __getitem__(self, index):
        low_img_name, high_img_name = self.input_names[index], self.groundtruth_names[index]
        
        img_id = low_img_name.split('\\')[-1]
        low_img, high_img = Image.open(low_img_name), Image.open(high_img_name)

        low_img, high_img = self.transforms(low_img, high_img)

        return {"lq": low_img, "hq": high_img, "img_id": img_id}

    def __len__(self):
        return len(self.input_names)


class MyDataset1(torch.utils.data.Dataset):
    def __init__(self, dir, filelist=None, train=True):
        super().__init__()

        self.input_names = natsort.natsorted(glob.glob(os.path.join(dir, filelist,'inputs', '*.png')))
        self.groundtruth_names = natsort.natsorted(glob.glob(os.path.join(dir, filelist, 'gts', '*.png')))

        if train:
            self.transforms = PairCompose([
                PairResizePad(out_hw=(384,384), pad_mode="reflect"),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ])
            self.input_names = self.input_names
            self.groundtruth_names = self.groundtruth_names
        else:
            self.transforms = PairCompose([
                PairResizePad(out_hw=(384,384), pad_mode="reflect"),
                # PairRandomCrop(self.patch_size),
                PairToTensor()
            ])
            self.input_names = self.input_names[:100]
            self.groundtruth_names = self.groundtruth_names[:100]

    def __getitem__(self, index):
        low_img_name, high_img_name = self.input_names[index], self.groundtruth_names[index]
        
        img_id = low_img_name.split('\\')[-1]
        low_img, high_img = Image.open(low_img_name), Image.open(high_img_name)

        low_img, high_img = self.transforms(low_img, high_img)

        return {"lq": low_img, "hq": high_img, "img_id": img_id}

    def __len__(self):
        return len(self.input_names)

class MyDataset2(torch.utils.data.Dataset):
    def __init__(self, train=True):
        super().__init__()

        

        if train:
            self.input_names = natsort.natsorted(glob.glob(os.path.join('train/low', '*.png')))[:10000]
            self.groundtruth_names = natsort.natsorted(glob.glob(os.path.join('train/high', '*.png')))
            
            self.transforms = PairCompose([
                PairResizePad(out_hw=(384,384), pad_mode="reflect"),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ])
            self.input_names = self.input_names
            self.groundtruth_names = self.groundtruth_names
        else:
            self.input_names = natsort.natsorted(glob.glob(os.path.join('val/low2', '*.png')))[:1000]
            self.groundtruth_names = natsort.natsorted(glob.glob(os.path.join('val/high', '*.png')))
            
            self.transforms = PairCompose([
                PairResizePad(out_hw=(384,384), pad_mode="reflect"),
                # PairRandomCrop(self.patch_size),
                PairToTensor()
            ])
            self.input_names = self.input_names[:100]
            self.groundtruth_names = self.groundtruth_names[:100]

    def __getitem__(self, index):
        low_img_name, high_img_name = self.input_names[index], self.groundtruth_names[index]
        
        img_id = low_img_name.split('\\')[-1]
        low_img, high_img = Image.open(low_img_name), Image.open(high_img_name)

        low_img, high_img = self.transforms(low_img, high_img)

        return {"lq": low_img, "hq": high_img, "img_id": img_id}

    def __len__(self):
        return len(self.input_names)

class MyDataset3(torch.utils.data.Dataset):
    def __init__(self, train=True):
        super().__init__()

        if train:
            self.input_names = natsort.natsorted(glob.glob(os.path.join('train', '*.png')))[:11000]
            self.groundtruth_names = natsort.natsorted(glob.glob(os.path.join('GT', '*.jpg')))[1500:]
            
            self.transforms = PairCompose([
                # PairResizePad(out_hw=(384,384), pad_mode="reflect"),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ])
            self.input_names = self.input_names
            self.groundtruth_names = self.groundtruth_names
        else:
            self.input_names = natsort.natsorted(glob.glob(os.path.join('test', '*.png')))[:1500]
            self.groundtruth_names = natsort.natsorted(glob.glob(os.path.join('GT', '*.jpg')))[:1500]
            self.transforms = PairCompose([
                # PairResizePad(out_hw=(384,384), pad_mode="reflect"),
                # PairRandomCrop(self.patch_size),
                PairToTensor()
            ])
            self.input_names = self.input_names[:100]
            self.groundtruth_names = self.groundtruth_names[:100]

    def __getitem__(self, index):
        low_img_name, high_img_name = self.input_names[index], self.groundtruth_names[index]
        
        img_id = low_img_name.split('\\')[-1]
        low_img, high_img = Image.open(low_img_name), Image.open(high_img_name)

        low_img, high_img = self.transforms(low_img, high_img)

        return {"lq": low_img, "hq": high_img, "img_id": img_id}

    def __len__(self):
        return len(self.input_names)
    
