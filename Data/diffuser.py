from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import (
    to_tensor,
    resize,
)

SIZE = 270, 480

def region_of_interest(x):
    return x[..., 60:270, 60:440]


def downsample_ax(img, factor):
    n = int(np.log2(factor))
    for i in range(n):
        if len(img.shape) == 2:
            img = .25 * (img[::2, ::2] + img[1::2, ::2]
                + img[::2, 1::2] + img[1::2, 1::2])
        else:
            img = .25 * (img[::2, ::2, :] + img[1::2, ::2, :]
                + img[::2, 1::2, :] + img[1::2, 1::2, :])
    return(img)


def transform(image, gray=False):
    image = np.flipud(image)
    image = image.copy()
    image = to_tensor(image)
    image = resize(image, SIZE)
    return image


def load_psf(path):
    psf = np.array(Image.open(path))
    return transform(psf)


class Lensless(Dataset):
    def __init__(self, diffuser_images, ground_truth_images):
        """
        Everything is upside-down, and the colors are BGR...
        """
        self.xs = diffuser_images
        self.ys = ground_truth_images

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        diffused = self.xs[idx]
        ground_truth = self.ys[idx]
        x = transform(np.load(diffused))
        y = transform(np.load(ground_truth))
        return x.float(), y.float()


def load_manifest(path, csv_filename):
    with open(path / csv_filename) as f:
        manifest = f.read().split()

    xs, ys = [], []
    for filename in manifest:
        x = path / 'diffuser_images' / filename.replace(".jpg.tiff", ".npy")
        y = path / 'ground_truth_lensed' / filename.replace(".jpg.tiff", ".npy")
        if x.exists() and y.exists():
            xs.append(x)
            ys.append(y)
        else:
            print(f"No file named {x}")
    return xs, ys




class Collection:

    def __init__(self, path):
        path = Path(path)

        self.psf = load_psf(path / 'psf.tiff')

        train_diffused, train_ground_truth = load_manifest(path, 'dataset_train.csv')
        val_diffused, val_ground_truth = load_manifest(path, 'dataset_test.csv')
        self.train_dataset = Lensless(train_diffused, train_ground_truth)
        self.val_dataset = Lensless(val_diffused, val_ground_truth)
        self.region_of_interest = region_of_interest