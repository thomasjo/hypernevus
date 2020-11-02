from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchsummary

# from ignite.contrib.handlers import WandBLogger
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.utils import convert_tensor, setup_logger
from torchvision.datasets import DatasetFolder
from torchvision.utils import make_grid

from hypernevus.models import Autoencoder


def main(args):
    # TODO(thomasjo): Make this configurable?
    ensure_reproducibility(seed=42)

    # TODO(thomasjo): Make this configurable?
    bands = slice(0, 115)

    dataset = prepare_dataset(args.data_dir, bands)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)

    # Grab a batch of images that will be used for visualizing epoch results.
    test_batch, _ = next(iter(dataloader))
    test_batch = test_batch[:16]
    test_batch = test_batch.to(device=args.device)

    input_size = test_batch.shape[1:]

    model = Autoencoder(input_size).to(device=args.device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    torchsummary.summary(model, input_size=input_size, batch_size=args.batch_size, device=str(args.device))

    # Create timestamped output directory.
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M")
    args.output_dir = args.output_dir / timestamp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y = prepare_batch(batch, args.device)
        x_hat = model(x)
        loss = criterion(x_hat, x)

        loss.backward()
        optimizer.step()

        return x, x_hat, loss.item()

    trainer = Engine(train_step)
    trainer.logger = setup_logger("trainer")

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_epoch_metrics(engine: Engine):
        _, _, loss = engine.state.output
        engine.logger.info(
            "Epoch [{}] Iteration [{}/{}] Loss: {:.4f}".format(
                engine.state.epoch,
                engine.state.iteration,
                engine.state.max_epochs * engine.state.epoch_length,
                loss,
            )
        )

    # Configure model checkpoints.
    checkpoint_handler = ModelCheckpoint(str(args.output_dir), filename_prefix="ckpt", n_saved=None)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {"model": model})

    # Visualize training progress using test patches.
    @trainer.on(Events.EPOCH_COMPLETED)
    def vizualize_reconstruction(engine: Engine):
        x, x_hat = test_batch, model(test_batch)
        fig, (ax1, ax2) = plt.subplots(1, 2, dpi=300)
        plot_image_grid(ax1, x, band=50, nrow=4)
        plot_image_grid(ax2, x_hat, band=50, nrow=4)
        fig.savefig(args.output_dir / f"epoch-{engine.state.epoch}.png", dpi=300)

    # Start model optimization.
    trainer.run(dataloader, max_epochs=50)


def prepare_batch(batch, device, non_blocking=True):
    x, y = batch
    return (
        convert_tensor(x, device, non_blocking),
        convert_tensor(y, device, non_blocking),
    )


def prepare_dataset(root_dir, bands):
    def is_valid_hsi_file(file_path: str):
        file_path = Path(file_path)
        allowed_parents = ["ak", "amh", "bcc", "mb", "mm", "scc", "sk"]
        return file_path.parent.name in allowed_parents and file_path.suffix == ".npy"

    dataset = DatasetFolder(str(root_dir), image_loader(bands), is_valid_file=is_valid_hsi_file)

    return dataset


def image_loader(bands):
    def _loader(file_path):
        hsi = np.load(file_path)
        hsi = hsi[..., bands]
        hsi = np.transpose(hsi, axes=[2, 0, 1])
        hsi = np.clip(hsi, 0, 1)
        return hsi

    return _loader


def plot_image_grid(axes, image, *, band, nrow=8):
    image = image.detach().cpu()
    image_grid = make_grid(torch.unsqueeze(image[:, band], 1), nrow=nrow)
    axes.imshow(np.transpose(image_grid, axes=[1, 2, 0])[..., 0], vmin=0, vmax=1)
    axes.axis("off")


def ensure_reproducibility(*, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data-dir", type=Path, required=True, metavar="PATH", help="dataset directory to use for training and evaluation")
    parser.add_argument("--output-dir", type=Path, required=True, metavar="PATH", help="path to output directory")
    parser.add_argument("--device", type=torch.device, default="cuda", help="target device for PyTorch operations")
    parser.add_argument("--batch-size", type=int, default=512, help="number of examples per mini-batch.")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
