import argparse
from pathlib import Path

from torch.utils.data import DataLoader, Subset

if __package__:
    from .diffuser import Collection
    from .phlatcam import PhaseMaskDataset
else:
    from diffuser import Collection
    from phlatcam import PhaseMaskDataset


def build_dataloaders(dataset, data_root, batch_size=4, num_workers=2,
                      pin_memory=True, max_len=None):
    """Return (train_loader, val_loader); select the dataset via `dataset`.

    DiffuserCam batches: (measurement, ground_truth).
    PhlatCam batches: (measurement, ground_truth, filename).
    Dataset-specific preprocessing and input/target sizes are preserved.
    """
    name = dataset.lower().replace('-', '').replace('_', '')
    data_root = Path(data_root).expanduser()

    if name == 'diffusercam':
        collection = Collection(data_root)
        train_dataset, val_dataset = collection.train_dataset, collection.val_dataset
        if max_len is not None:
            train_dataset = Subset(train_dataset, range(min(max_len, len(train_dataset))))
            val_dataset = Subset(val_dataset, range(min(max_len, len(val_dataset))))
    elif name == 'phlatcam':
        train_dataset = PhaseMaskDataset(mode='train', root=data_root, max_len=max_len)
        val_dataset = PhaseMaskDataset(mode='val', root=data_root, max_len=max_len)
    else:
        raise ValueError(f'Unknown dataset: {dataset!r}; use diffusercam or phlatcam.')

    if not train_dataset or not val_dataset:
        raise ValueError(f'Found empty split(s): train={len(train_dataset)}, val={len(val_dataset)}')

    loader_args = dict(batch_size=batch_size, num_workers=num_workers,
                       pin_memory=pin_memory, persistent_workers=num_workers > 0,
                       drop_last=False)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    return train_loader, val_loader


def build_dataloaders_from_cfg(cfg):
    """Expected cfg: dataset, data_root, batch_size, num_workers.

    Optional cfg: pin_memory (default True), max_len (default None).
    """
    return build_dataloaders(
        dataset=cfg.dataset,
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=getattr(cfg, 'pin_memory', True),
        max_len=getattr(cfg, 'max_len', None),
    )


def main():
    parser = argparse.ArgumentParser(description='Check DiffuserCam/PhlatCam data loading')
    parser.add_argument('--dataset', required=True, choices=['diffusercam', 'phlatcam'])
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_len', type=int, default=None)
    args = parser.parse_args()

    train_loader, val_loader = build_dataloaders(
        dataset=args.dataset, data_root=args.data_root,
        batch_size=args.batch_size, num_workers=args.num_workers, max_len=args.max_len)
    print(f'Dataset: {args.dataset}')
    print(f'Train: {len(train_loader.dataset)} samples / {len(train_loader)} batches')
    print(f'Val:   {len(val_loader.dataset)} samples / {len(val_loader)} batches')
    for split_name, loader in [('train', train_loader), ('val', val_loader)]:
        batch = next(iter(loader))
        x, y = batch[:2]
        print(f'{split_name}: measurement={tuple(x.shape)}, ground_truth={tuple(y.shape)}')
        if len(batch) > 2:
            print(f'{split_name}: first filename={batch[2][0]}')


if __name__ == '__main__':
    main()
