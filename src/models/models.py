import torch
from torch import nn
import timm


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        total_in_channels = in_channels + skip_channels
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=total_in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        return x


class MonocularDepthModel(nn.Module):
    def __init__(self, encoder_name="resnet34", pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained, features_only=True)
        ch = self.encoder.feature_info.channels()
        self.block5 = DecoderBlock(ch[4], ch[3], ch[3])
        self.block4 = DecoderBlock(ch[3], ch[2], ch[2])
        self.block3 = DecoderBlock(ch[2], ch[1], ch[1])
        self.block2 = DecoderBlock(ch[1], ch[0], ch[0] // 2)
        self.block1 = DecoderBlock(ch[0] // 2, 0, ch[0] // 4)

        self.head = nn.Conv2d(ch[0] // 4, 1, kernel_size=1)

    def forward(self, x):
        f0, f1, f2, f3, f4 = self.encoder(x)
        x = self.block5(f4, skip=f3)
        x = self.block4(x, skip=f2)
        x = self.block3(x, skip=f1)
        x = self.block2(x, skip=f0)
        x = self.block1(x, skip=None)

        return self.head(x)
