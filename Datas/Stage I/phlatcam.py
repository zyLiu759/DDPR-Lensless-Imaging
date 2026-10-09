"""PhlatCam dataset reader, preserving the original crop and downsampling settings."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


SIZE_ORIGINAL = (960, 1280)  # (height, width) before downsampling
DOWNSAMPLE = 2
SIZE_DOWN = (SIZE_ORIGINAL[0] // DOWNSAMPLE, SIZE_ORIGINAL[1] // DOWNSAMPLE)
SIZE = (192, 192)  # (height, width) of the ground truth

psf_crop_top = 808 - SIZE_ORIGINAL[0] // 2
psf_crop_bottom = 808 + SIZE_ORIGINAL[0] // 2
psf_crop_left = 965 - SIZE_ORIGINAL[1] // 2
psf_crop_right = 965 + SIZE_ORIGINAL[1] // 2

DEFAULT_ROOT = Path('dataset/phlatcam/Display-Captures')


def region_of_interest(x):
    h, w = SIZE_DOWN
    crop_h, crop_w = SIZE
    top = h // 2 - crop_h // 2
    left = w // 2 - crop_w // 2
    return x[..., top:top + crop_h, left:left + crop_w]


def load_psf(psf_path=None, root=DEFAULT_ROOT):
    """Load a calibrated PhlatCam PSF using the original crop and 0.1 scaling."""
    psf_path = Path(psf_path) if psf_path is not None else Path(root) / 'phase_psf' / 'psf.npy'
    psf = np.load(psf_path)
    cropped = psf[psf_crop_top:psf_crop_bottom, psf_crop_left:psf_crop_right]
    if cropped.shape[:2] != SIZE_ORIGINAL:
        raise ValueError(f'PSF crop shape is {cropped.shape[:2]}, expected {SIZE_ORIGINAL}; check PSF coordinates.')
    cropped = cv2.resize(cropped, (SIZE_DOWN[1], SIZE_DOWN[0]), interpolation=cv2.INTER_AREA)
    psf_tensor = to_tensor(cropped)
    return psf_tensor * 0.1


def get_img_from_raw(raw):
    """Convert an RGGB Bayer image to RGB, then crop once at the calibrated location."""
    if raw.ndim != 2:
        raise ValueError(f'Expected 2-D raw Bayer image, got shape {raw.shape}.')
    h, w = raw.shape
    raw = raw[:h - h % 2, :w - w % 2]
    r = raw[0::2, 0::2]
    gr = raw[0::2, 1::2]
    gb = raw[1::2, 0::2]
    b = raw[1::2, 1::2]
    rgb = np.stack((r, 0.5 * (gr + gb), b), axis=-1)
    cropped = rgb[psf_crop_top:psf_crop_bottom, psf_crop_left:psf_crop_right]
    if cropped.shape[:2] != SIZE_ORIGINAL:
        raise ValueError(f'Measurement crop shape is {cropped.shape[:2]}, expected {SIZE_ORIGINAL}; check raw resolution and crop coordinates.')
    return cropped


class PhaseMaskDataset(Dataset):
    def __init__(self, mode='train', max_len=None, root=DEFAULT_ROOT):
        super().__init__()
        if mode not in ('train', 'val', 'test'):
            raise ValueError('mode must be train, val, or test')
        self.mode = mode
        self.image_dir = Path(root).expanduser()
        self.max_len = max_len
        self.source_paths, self.target_paths = self._load_dataset()
        if len(self.source_paths) != len(self.target_paths):
            raise ValueError(f'Unpaired lists: {len(self.source_paths)} measurements vs {len(self.target_paths)} targets.')

    def _glob_images(self, file_list):
        file_list = Path(file_list)
        with file_list.open(encoding='utf-8-sig') as f:
            entries = [line.strip().strip('"') for line in f if line.strip()]
        paths = []
        for entry in entries:
            p = Path(entry)
            if p.is_absolute():
                paths.append(p)
                continue
            # Some split files use paths relative to the dataset root, others
            # already include the project-relative "dataset/phlatcam/..." prefix.
            candidates = [self.image_dir / p, self.image_dir.parent / p, Path.cwd() / p]
            paths.append(next((candidate for candidate in candidates if candidate.exists()), candidates[0]))
        return paths

    def _load_dataset(self):
        text_dir = self.image_dir / 'text_files'
        prefix = 'train' if self.mode == 'train' else 'val'
        source_file = text_dir / f'{prefix}_source_imagenet_384_384_Feb_19.txt'
        target_file = text_dir / f'{prefix}_target.txt'
        return (self._glob_images(source_file)[:self.max_len],
                self._glob_images(target_file)[:self.max_len])

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        source_path = self.source_paths[index]
        target_path = self.target_paths[index]
        raw = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f'Cannot load PhlatCam measurement: {source_path}')
        raw = raw.astype(np.float32) / 4096.0
        source = get_img_from_raw(raw)
        # 960 x 1280 -> 480 x 640 (one crop only).
        source = cv2.resize(source, (SIZE_DOWN[1], SIZE_DOWN[0]), interpolation=cv2.INTER_AREA)

        target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
        if target is None:
            raise FileNotFoundError(f'Cannot load PhlatCam target: {target_path}')
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        target = cv2.resize(target, (SIZE[1], SIZE[0]), interpolation=cv2.INTER_LINEAR)
        return (torch.from_numpy(np.ascontiguousarray(source.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(target.transpose(2, 0, 1))),
                source_path.name)
