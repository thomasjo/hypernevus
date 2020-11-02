import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        c, h, w = input_size
        assert h == w, "spatial dimensions of input must be square"
        self.input_size = input_size

        self.encoder = nn.Sequential(
            # conv1
            nn.Conv2d(c, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            # conv2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            # conv3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            # fc
            nn.Flatten(),
            nn.Linear(4096, 256),
        )

        self.decoder = nn.Sequential(
            # fc
            nn.Linear(256, 4096),
            nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)),
            # upconv3
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            # upconv2
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            # upconv1
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, c, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)

        return x_hat
