from pathlib import Path

import numpy as np

from torchvision.datasets import DatasetFolder
from torchvision.transforms import ToTensor

_allowed_classes = ["ak", "amh", "bcc", "mb", "mm", "scc", "sk"]


def prepare_dataset(root_dir: Path, bands: slice):
    def is_valid_hsi_file(file_path: str):
        file_path = Path(file_path)
        return file_path.parent.name in _allowed_classes and file_path.suffix == ".npy"

    dataset = DatasetFolder(
        root=str(root_dir),
        loader=image_loader(bands),
        is_valid_file=is_valid_hsi_file,
        transform=ToTensor(),
    )

    return dataset


def image_loader(bands: slice):
    def hsi_loader(file_path: str):
        hsi = np.load(file_path)
        hsi = hsi[..., bands]
        # hsi = np.transpose(hsi, axes=[2, 0, 1])
        hsi = np.clip(hsi, 0, 1)
        return hsi

    return hsi_loader
