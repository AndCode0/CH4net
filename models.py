import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Base MLP module with ReLU activation
    Parameters:
    -----------
    in_channels: Int
        Number of input channels
    out_channels: Int
        Number of output channels
    h_channels: Int
        Number of hidden channels
    h_layers: Int
        Number of hidden layers
    """

    def __init__(self, in_channels, out_channels, h_channels=64, h_layers=4):

        super().__init__()

        def hidden_block(h_channels):
            h = nn.Sequential(nn.Linear(h_channels, h_channels), nn.ReLU())
            return h

        # Model

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, h_channels),
            nn.ReLU(),
            *[hidden_block(h_channels) for _ in range(h_layers)],
            nn.Linear(h_channels, out_channels)
        )

    def forward(self, x):
        return self.mlp(x)


class Unet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        div_factor=8,
        prob_output=True,
        class_output=False,
    ):
        super(Unet, self).__init__()

        self.n_channels = in_channels
        self.bilinear = True
        self.sigmoid = nn.Sigmoid()
        self.prob_output = prob_output
        self.class_output = class_output

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2), double_conv(in_channels, out_channels)
            )

        class up(nn.Module):
            def __init__(self, in_channels, out_channels, bilinear=True):
                super().__init__()

                if bilinear:
                    # Bilinear upsampling doesn't change channel count
                    # After concat: in_channels + skip_channels = in_channels + in_channels//2
                    self.up = nn.Upsample(
                        scale_factor=2, mode="bilinear", align_corners=True
                    )
                else:
                    raise NotImplementedError(
                        "bilinear is set to False, which means that the upscaling would be done with ConvTranspose2d. The dimensions in forward *might* not add up"
                    )
                    self.up = nn.ConvTranspose2d(
                        in_channels // 2, in_channels // 2, kernel_size=2, stride=2
                    )

                self.conv = double_conv(in_channels, out_channels)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                # [?, C, H, W]
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(
                    x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
                )
                x = torch.cat([x2, x1], dim=1)  ## concatenate along channel dimension
                return self.conv(x)

        base_channels = 64 // div_factor

        self.inc = double_conv(self.n_channels, base_channels)
        self.down1 = down(base_channels, base_channels * 2)
        self.down2 = down(base_channels * 2, base_channels * 4)
        self.down3 = down(base_channels * 4, base_channels * 8)
        self.down4 = down(base_channels * 8, base_channels * 8)

        self.up1 = up(base_channels * 16, base_channels * 4)
        self.up2 = up(base_channels * 8, base_channels * 2)
        self.up3 = up(base_channels * 4, base_channels)
        self.up4 = up(base_channels * 2, base_channels * 2)
        self.out = nn.Conv2d(base_channels * 2, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if self.prob_output:
            x = self.out(x)
            return self.sigmoid(x).permute(0, 2, 3, 1)
        else:
            return self.out(x).permute(0, 2, 3, 1)
