import logging
import warnings

from argparse import ArgumentParser, HelpFormatter, Namespace
from datetime import datetime
from itertools import combinations, groupby
from math import ceil
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.cm
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader

from hypernevus.datasets import image_loader, prepare_dataset
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
    run_dirs = sorted(list(args.state_dir.glob("run*/")))

    bands = slice(0, 115)
    load_image = image_loader(bands)

    for prefix in image_prefixes:
        print(prefix)

        output_dir = args.output_dir / prefix
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = sorted(list(args.data_dir.rglob(f"{prefix}*.npy")))
        patches = np.stack([load_image(str(path)) for path in paths]).astype(np.double)
        n, h, w, c = patches.shape

        all_labels = []
        all_labels_pca = []

        # Iterate over each run.
        for run_dir in run_dirs:
            print(f" -> {run_dir.stem}")

            pca = joblib.load(run_dir / "pca.joblib")
            patches_pca = pca.transform(patches.reshape((n, h * w * c)))

            km, km_pca = load_kmeans(run_dir)
            labels = km.predict(patches.reshape((n, h * w * c)))
            labels_pca = km_pca.predict(patches_pca)

            # Re-assign cluster labels to ensure consistency between run predictions.
            labels = find_consistent_labels(labels, all_labels, km.n_clusters)
            labels_pca = find_consistent_labels(labels_pca, all_labels_pca, km_pca.n_clusters)

            # Save cluster predictions.
            all_labels.append(labels)
            all_labels_pca.append(labels_pca)

            save_cluster_image(patches.reshape(n, h, w, c), labels, paths, output_dir / "{}.png".format(run_dir.stem))
            save_cluster_image(patches.reshape(n, h, w, c), labels_pca, paths, output_dir / "pca--{}.png".format(run_dir.stem))

        # Create visualization of cluster variations.
        save_change_image(patches.reshape(n, h, w, c), all_labels, paths, output_dir / "changes.png")
        save_change_image(patches.reshape(n, h, w, c), all_labels_pca, paths, output_dir / "pca--changes.png")

        # break  # HACK(thomasjo): DEBUG

        # Iterate over all pair-wise combinations.
        # for run_a, run_b in combinations(run_dir, 2):


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


def load_kmeans(state_dir: Path):
    kmeans = joblib.load(state_dir / "kmeans.joblib")
    kmeans_pca = joblib.load(state_dir / "kmeans_pca.joblib")

    return kmeans, kmeans_pca


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

    parser.add_argument("--state-dir", metavar="PATH", type=Path, required=True, help="path to directory containing model state")
    parser.add_argument("--data-dir", metavar="PATH", type=Path, required=True, help="path to dataset directory")
    parser.add_argument("--output-dir", metavar="PATH", type=Path, required=True, help="path to output directory")

    parser.add_argument("--batch-size", metavar="NUM", type=int, default=256, help="batch sample size used during training")
    parser.add_argument("--workers", metavar="NUM", type=int, default=1, help="number of processes used to load data")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
