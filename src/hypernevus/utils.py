from math import ceil

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def make_grid(tensor, ncols=8, vmin=0, vmax=1, cmap="viridis"):
    nrows = ceil(tensor.shape[0] / ncols)

    fig = Figure(figsize=(8, 8), dpi=600)
    canvas = FigureCanvasAgg(fig)

    for image_num, image in enumerate(tensor, start=1):
        ax = fig.add_subplot(nrows, ncols, image_num)
        ax.imshow(image, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.axis("off")

    fig.tight_layout()
    canvas.draw()
    buf = canvas.buffer_rgba()
    image_grid = np.asarray(buf)

    return image_grid


def show_image_grid(image, axes, band=50, vmin=0, vmax=1, cmap="viridis"):
    temp_image = image.detach().cpu()
    temp_image = temp_image[:, band]
    image_grid = make_grid(temp_image, vmin=vmin, vmax=vmax, cmap=cmap)
    axes.imshow(image_grid)
    axes.axis("off")
