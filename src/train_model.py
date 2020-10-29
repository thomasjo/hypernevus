import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchsummary
import torchvision as vision

from hypernevus.models import Autoencoder, Autoencoder2
from hypernevus.utils import show_image_grid


class ECSLoss(nn.Module):
    __constants__ = ["reduction"]

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        n_channels = input.size(1)
        cumulative_diff = torch.cumsum(input, dim=1) - torch.cumsum(target, dim=1)
        spatial_ecs = torch.sqrt(torch.sum(torch.square(cumulative_diff), dim=1))
        spatial_ecs = spatial_ecs / (2 * n_channels)
        mean_ecs = torch.mean(spatial_ecs)

        return mean_ecs


def ensure_reproducibility(*, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def image_loader(bands):
    def _loader(file_path):
        hsi = np.load(file_path)
        hsi = hsi[..., bands]
        hsi = np.transpose(hsi, axes=[2, 0, 1])
        hsi = np.clip(hsi, 0, 1)
        return hsi

    return _loader


def prepare_dataset(root_dir, bands):
    dataset = vision.datasets.DatasetFolder(str(root_dir), image_loader(bands), extensions=".npy")
    return dataset


def save_checkpoint(output_dir, model, optimizer, loss, epoch):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, str(output_dir / f"epoch-{epoch + 1}.ckpt"))


def plot_image_grid(axes, image, *, band, vmin=0, vmax=1, cmap="viridis"):
    # image_grid = vision.utils.make_grid(torch.unsqueeze(image[:, band].detach(), 1))
    # axes.imshow(np.transpose(image_grid, axes=[1, 2, 0])[..., 0], vmin=0, vmax=1)
    show_image_grid(image, axes, band, vmin, vmax, cmap)


def save_reconstruction_vizualization(output_dir, model, test_image, epoch, device):
    with torch.no_grad():
        image = test_image.to(device=device, non_blocking=True)
        reconstructed_image = model(image)

    image = image.cpu()
    reconstructed_image = reconstructed_image.cpu()
    difference_image = image - reconstructed_image

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=600)
    plot_image_grid(ax1, image, band=50)
    plot_image_grid(ax2, reconstructed_image, band=50)
    plot_image_grid(ax3, difference_image, band=50, vmin=None, vmax=None, cmap="gray")
    fig.tight_layout()
    fig.savefig(output_dir / f"epoch-{epoch + 1}.png", dpi=600)


def main(args):
    # TODO(thomasjo): Make this configurable?
    bands, max_bands = slice(0, 115), 120
    num_bands = len(range(*bands.indices(max_bands)))

    # TODO(thomasjo): Make this configurable?
    ensure_reproducibility(seed=42)

    dataset = prepare_dataset(args.data_dir, bands)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)

    # Grab a batch of images that will be used for visualizing epoch results.
    test_image, _ = next(iter(dataloader))
    test_image = test_image[:64]

    device = torch.device(args.device)
    autoencoder = Autoencoder(num_bands)
    autoencoder = autoencoder.to(device=device)
    optimizer = optim.Adam(autoencoder.parameters())
    # criterion = nn.BCELoss()
    criterion = ECSLoss()

    torchsummary.summary(autoencoder, input_size=test_image.shape[1:], batch_size=args.batch_size, device=args.device)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training model using dataset '{args.data_dir}'...")
    for epoch in range(0, 50):
        sys.stdout.write("epoch: {}\n".format(epoch + 1))
        sys.stdout.flush()

        train_loss = 0
        autoencoder.train()
        for batch_idx, (image, _) in enumerate(dataloader):
            sys.stdout.write("==> batch: {}\n".format(batch_idx + 1))
            sys.stdout.flush()

            optimizer.zero_grad()
            image = image.to(device=device, non_blocking=True)
            reconstructed_image = autoencoder(image)
            loss = criterion(reconstructed_image, image)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        sys.stdout.write(str(train_loss / len(dataloader.dataset)) + "\n")
        sys.stdout.flush()

        save_checkpoint(output_dir, autoencoder, optimizer, loss, epoch)
        save_reconstruction_vizualization(output_dir, autoencoder, test_image, epoch, device)


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, metavar="PATH", help="dataset directory to use for training and evaluation")
    parser.add_argument("--output-dir", type=Path, required=True, metavar="PATH", help="path to output directory")
    parser.add_argument("--device", type=str, default="cuda", help="target device for PyTorch operations")
    parser.add_argument("--batch-size", type=int, default=512, help="number of examples per mini-batch.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
