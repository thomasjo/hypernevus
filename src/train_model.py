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

from torchvision.datasets import DatasetFolder
from torchvision.utils import make_grid

from hypernevus.models import Autoencoder


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

    autoencoder = Autoencoder(num_bands).to(device=args.device)
    optimizer = optim.Adam(autoencoder.parameters())
    criterion = nn.BCELoss()

    torchsummary.summary(autoencoder, input_size=test_image.shape[1:], batch_size=args.batch_size, device=str(args.device))

    # Create timestamped output directory.
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
            image = image.to(device=args.device, non_blocking=True)
            reconstructed_image = autoencoder(image)
            loss = criterion(reconstructed_image, image)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        sys.stdout.write(str(train_loss / len(dataloader.dataset)) + "\n")
        sys.stdout.flush()

        save_checkpoint(output_dir, autoencoder, optimizer, loss, epoch)
        save_reconstruction_vizualization(output_dir, autoencoder, test_image, epoch, args.device)


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
    dataset = DatasetFolder(str(root_dir), image_loader(bands), extensions=".npy")
    return dataset


def save_checkpoint(output_dir, model, optimizer, loss, epoch):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, str(output_dir / f"epoch-{epoch + 1}.ckpt"))


def plot_image_grid(axes, image, *, band):
    image_grid = make_grid(torch.unsqueeze(image[:, band].detach(), 1))
    axes.imshow(np.transpose(image_grid, axes=[1, 2, 0])[..., 0], vmin=0, vmax=1)


def save_reconstruction_vizualization(output_dir, model, test_image, epoch, device):
    with torch.no_grad():
        image = test_image.to(device=device, non_blocking=True)
        reconstructed_image = model(image)

    image = image.cpu()
    reconstructed_image = reconstructed_image.cpu()

    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=300)
    plot_image_grid(ax1, image, band=50)
    plot_image_grid(ax2, reconstructed_image, band=50)
    fig.savefig(output_dir / f"epoch-{epoch + 1}.png", dpi=300)


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data-dir", type=Path, required=True, metavar="PATH", help="dataset directory to use for training and evaluation")
    parser.add_argument("--output-dir", type=Path, required=True, metavar="PATH", help="path to output directory")
    parser.add_argument("--device", type=torch.device, default="cuda", help="target device for PyTorch operations")
    parser.add_argument("--batch-size", type=int, default=512, help="number of examples per mini-batch.")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
