import logging

from argparse import ArgumentParser, HelpFormatter, Namespace
from datetime import datetime
from itertools import combinations, groupby
from math import ceil
from pathlib import Path

import joblib
import matplotlib.cm
import numpy as np
import torch

from PIL import Image
from sklearn.metrics.cluster import normalized_mutual_info_score  # noqa
from torch.utils.data import DataLoader

from hypernevus.datasets import image_loader, prepare_dataset
from hypernevus.models import Autoencoder
from hypernevus.utils import ensure_reproducibility

BACKGROUND_COLOR = (0, 0, 0, 255)
LABEL_COLORS = {
    0: (200, 200, 200, 0),
    1: (0, 255, 255, round(0.33 * 255)),
}


def main(args: Namespace):
    # logger = setup_logging()

    # TODO(thomasjo): Make this configurable?
    ensure_reproducibility(seed=42)

    # Create timestamped output directory.
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M")
    args.output_dir = args.output_dir / timestamp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare dataloader.
    # dataloader = prepare_dataloader(args)

    image_paths = sorted(list(args.data_dir.rglob("*.npy")))
    image_prefixes = sorted([k for k, g in groupby(image_paths, key=lambda x: x.stem.partition("--")[0])])

    # Find run directories.
    # run_dirs = sorted(list(args.state_dir.glob("run*/")))
    run_dirs = [args.state_dir]

    bands = slice(0, 115)
    load_image = image_loader(bands)

    for prefix in image_prefixes:
        print(prefix)

        output_dir = args.output_dir / prefix
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = sorted(list(args.data_dir.rglob(f"{prefix}*.npy")))
        patches = np.stack([load_image(str(path)) for path in paths]).astype(np.double)
        n, h, w, c = patches.shape

        patches = torch.tensor(patches)

        # all_labels = []
        # all_metrics = []

        # Iterate over each run.
        for run_dir in run_dirs:
            print(f" -> {run_dir.stem}")

            print(np.transpose(patches, (0, 3, 1, 2)).shape)

            input_size = (115, 32, 32)
            model = Autoencoder(input_size)
            model.load_state_dict(torch.load(args.model_ckpt, map_location=args.device))
            with torch.no_grad():
                model.eval()
                km = joblib.load(run_dir / "kmeans.joblib")
                labels = km.predict(model.encode(patches.permute(0, 3, 1, 2).to(dtype=torch.float)))
                # labels = km.predict(model.encode(np.transpose(patches, (0, 3, 1, 2))))
            # labels = km.predict(patches.reshape((n, h * w * c)))

            # Re-assign cluster labels to ensure consistency between run predictions.
            # labels = find_consistent_labels(labels, all_labels, km.n_clusters)

            # Save cluster predictions.
            # all_labels.append(labels)
            save_cluster_image(patches.reshape(n, h, w, c), labels, paths, output_dir / "{}.png".format(run_dir.stem))

        # Create visualization of cluster variations.
        # save_change_image(patches.reshape(n, h, w, c), all_labels, paths, output_dir / "changes.png")

        # Compute pair-wise clustering scores for all run combinations.
        # nmi = np.stack([normalized_mutual_info_score(a, b) for a, b in combinations(all_labels, 2)])
        # all_metrics.append(nmi)
        # print("- mean:", np.mean(nmi, axis=0))
        # print("- std: ", np.var(nmi, axis=0))

    print("-" * 80)

    # Calculate statistics for metrics from all runs.
    # nmi = np.stack(all_metrics, axis=0)
    # print("mean:", np.mean(nmi))
    # print("std: ", np.var(nmi))


def find_consistent_labels(labels, all_labels, n_clusters):
    if not all_labels:
        return labels

    # Grab previous cluster predictions.
    prev_labels = all_labels[-1]
    prev_clusters = [prev_labels == label_id for label_id in range(n_clusters)]

    # Compare each cluster against all clusters from previous run.
    new_labels = []
    for label_id in range(n_clusters):
        cluster = labels == label_id
        new_label_id = np.argmax([np.mean(np.equal(cluster, prev_cluster)) for prev_cluster in prev_clusters])
        new_labels.append((new_label_id, cluster))

    # Re-assign cluster labels to ensure consistency between run predictions.
    for new_label_id, cluster in new_labels:
        labels[cluster] = new_label_id

    return labels


def save_cluster_image(images, labels, paths, output_file, label_colors=LABEL_COLORS, bg_color=BACKGROUND_COLOR):
    cm = matplotlib.cm.get_cmap("gray")
    band_idx = 50

    pad = 1
    img_shape = ((32 + pad) * 32 + pad, (32 + pad) * 32 + pad)
    img = Image.new("RGBA", img_shape, bg_color)

    for image, label, image_file in zip(images, labels, paths):
        # Read row and column info from filename.
        xi, yi = image_file.stem.split("--")[1].split("-")
        row_idx, col_idx = int(yi), int(xi)

        # Make a single-band image patch.
        band_image = np.clip(cm(image[..., band_idx]) * 255, 0, 255).astype(np.uint8)
        temp = Image.new("RGBA", ((32 + pad, 32 + pad)), bg_color)
        temp.paste(Image.fromarray(band_image, mode="RGBA"), (ceil(pad / 2), ceil(pad / 2)))

        # Overlay patch image with label color.
        overlay = Image.new("RGBA", temp.size, label_colors[label])
        temp = Image.alpha_composite(temp, overlay)

        # Add patch image to the full image.
        img.paste(temp, ((32 + pad) * col_idx + (pad // 2), (32 + pad) * row_idx + (pad // 2)))

    img.save(output_file)


def save_change_image(images, all_labels, paths, output_file):
    # Colors for changed cluster labels.
    label_colors = {
        0: (0, 0, 0, 0),  # The "no change" label has a transparent color
        1: (255, 0, 0, round(0.3 * 255)),
    }

    label_changes = np.sum([np.not_equal(a, b) for a, b in combinations(all_labels, 2)], axis=0)
    change_labels = np.clip(label_changes, 0, 1)
    print("  # changes:", np.count_nonzero(change_labels))

    save_cluster_image(images, change_labels, paths, output_file, label_colors)


def prepare_dataloader(args: Namespace):
    dataset = prepare_dataset(args.data_dir, bands=slice(0, 115))
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.workers)

    return dataloader


def setup_logging():
    logger = logging.Logger("baseline", level=logging.INFO)

    handler = logging.StreamHandler()
    logger.addHandler(handler)

    # TODO(thomasjo): Configure formatting with timestamps, etc.

    return logger


def parse_args():
    parser = ArgumentParser(formatter_class=lambda prog: HelpFormatter(prog, max_help_position=40))

    parser.add_argument("--data-dir", metavar="PATH", type=Path, required=True, help="path to dataset directory")
    parser.add_argument("--output-dir", metavar="PATH", type=Path, required=True, help="path to output directory")
    parser.add_argument("--state-dir", metavar="PATH", type=Path, required=True, help="path to directory containing model state")
    parser.add_argument("--model-ckpt", metavar="PATH", type=Path, required=True, help="path to model checkpoint file")

    parser.add_argument("--batch-size", metavar="NUM", type=int, default=256, help="batch sample size used during training")
    parser.add_argument("--device", type=torch.device, default="cuda", help="target device for PyTorch operations")
    parser.add_argument("--workers", metavar="NUM", type=int, default=1, help="number of processes used to load data")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
