import logging
import warnings

from argparse import ArgumentParser, HelpFormatter, Namespace
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from torch.utils.data import DataLoader

from hypernevus.datasets import prepare_dataset
from hypernevus.utils import ensure_reproducibility


def main(args: Namespace):
    logger = setup_logging()

    # TODO(thomasjo): Make this configurable?
    ensure_reproducibility(seed=42)

    # Create timestamped output directory.
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M")
    args.output_dir = args.output_dir / timestamp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.restart_dir:
        logger.info("Running in restart mode: {}".format(args.restart_dir))

    # Prepare dataloader.
    dataloader = prepare_dataloader(args)

    # Initialize PCA for dataset.
    pca = compute_pca(dataloader, args, logger)

    # Save PCA state to allow both reuse and restart.
    joblib.dump(pca, args.output_dir / "pca.joblib")

    explained_variance = np.sum(pca.explained_variance_ratio_)
    logger.info("Explained variance with PCA: {:.2f}".format(explained_variance * 100))

    # Cluster dataset using (PCA transformed) batches of data.
    logger.info("Clustering dataset...")
    kmeans = MiniBatchKMeans(n_clusters=args.clusters)
    kmeans_pca = MiniBatchKMeans(n_clusters=args.clusters)
    for batch_num, (x, y) in enumerate(dataloader, start=1):
        logger.info("Batch [{}]".format(batch_num))
        kmeans.partial_fit(x.flatten(start_dim=1))
        kmeans_pca.partial_fit(pca.transform(x.flatten(start_dim=1)))

    # Save model state.
    joblib.dump(kmeans, args.output_dir / "kmeans.joblib")
    joblib.dump(kmeans_pca, args.output_dir / "kmeans_pca.joblib")


def compute_pca(dataloader: DataLoader, args: Namespace, logger: logging.Logger):
    saved_pca = load_saved_pca(args.restart_dir, logger)
    if saved_pca:
        logger.info("Loaded PCA from saved state.")
        return saved_pca

    logger.info("Computing {} PCA components...".format(args.components))
    pca = IncrementalPCA(args.components)
    max_pca_batches = len(dataloader) * 0.2  # Only use a subset of the dataset to improve effiency
    for batch_num, (x, y) in enumerate(dataloader, start=1):
        logger.info("Batch [{}]".format(batch_num))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"Mean of empty slice", category=RuntimeWarning)
            warnings.filterwarnings("ignore", r"invalid value encountered", category=RuntimeWarning)
            pca.partial_fit(x.flatten(start_dim=1))
        if batch_num == max_pca_batches:
            break

    return pca


def load_saved_pca(restart_dir: Optional[Path], logger: logging.Logger) -> Optional[IncrementalPCA]:
    if not restart_dir:
        return None

    state_file = restart_dir / "pca.joblib"
    if state_file.exists():
        return joblib.load(state_file)

    return None


def prepare_dataloader(args: Namespace):
    dataset = prepare_dataset(args.data_dir, bands=slice(0, 115))
    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.workers)

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
    parser.add_argument("--restart-dir", metavar="PATH", type=Path, help="path to directory containing model state")

    parser.add_argument("--batch-size", metavar="NUM", type=int, default=256, help="batch sample size used during training")
    parser.add_argument("--clusters", metavar="NUM", type=int, default=2, help="number of target clusters")
    parser.add_argument("--components", metavar="NUM", type=int, default=256, help="number of PCA components to use")
    parser.add_argument("--repetitions", metavar="NUM", type=int, default=1, help="number of times to repeat the training to collect statistics")
    parser.add_argument("--workers", metavar="NUM", type=int, default=1, help="number of processes used to load data")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
