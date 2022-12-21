import torch
import torch.nn as nn


class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpConvBlock, self).__init__()
        self.blk = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.blk(x)


#------------------------------------------------#
#   类似Unet的解码器
#
#   f0 ─── conv ──── cat ── upconv2mask ── final_conv ──>
#                     │
#                     up
#   f1 ───────── cat ─┘
#                 │
#                 up
#   f2 ───── cat ─┘
#             │
#             up
#   f3 ─ cat ─┘
#         │
#         up
#         │
#   encoder_output
#------------------------------------------------#
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.upconv3 = UpConvBlock(512, 256)
        self.upconv2 = UpConvBlock(512, 128)
        self.upconv1 = UpConvBlock(256, 64)
        self.upconv0 = UpConvBlock(128, 48)
        self.conv = nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1)

        self.upconv2mask = UpConvBlock(96, 48)

        self.final_conv = nn.Conv2d(48, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, encoder_output, concat_features):
        # encoder_output  = [B, 512, 8, 8]
        # concat_features = [[B, 64, 128, 128], [B, 64, 64, 64], [B, 128, 32, 32], [B, 256, 16, 16]]
        f0, f1, f2, f3 = concat_features

        x_up3 = self.upconv3(encoder_output)      # [B, 512, 8, 8] -> [B, 256, 16, 16]
        x_up3 = torch.cat([x_up3, f3], dim=1)     # [B, 256, 16, 16] cat [B, 256, 16, 16] = [B, 512, 16, 16]

        x_up2 = self.upconv2(x_up3)               # [B, 512, 16, 16] -> [B, 128, 32, 32]
        x_up2 = torch.cat([x_up2, f2], dim=1)     # [B, 128, 32, 32] cat [B, 128, 32, 32] = [B, 256, 32, 32]

        x_up1 = self.upconv1(x_up2)               # [B, 256, 32, 32] -> [B, 64, 64, 64]
        x_up1 = torch.cat([x_up1, f1], dim=1)     # [B, 64, 64, 64] cat [B, 64, 64, 64] = [B, 128, 64, 64]

        x_up0 = self.upconv0(x_up1)               # [B, 128, 64, 64]  -> [B, 48, 128, 128]
        f0 = self.conv(f0)                        # [B, 64, 128, 128] -> [B, 48, 128, 128]
        x_up2mask = torch.cat([x_up0, f0], dim=1) # [B, 48, 128, 128] cat [B, 48, 128, 128] = [B, 96, 128, 128]

        x_mask = self.upconv2mask(x_up2mask)      # [B, 96, 128, 128] -> [B, 48, 256, 256]

        x_mask = self.final_conv(x_mask)          # [B, 48, 256, 256] -> [B, 2, 256, 256]

        return x_mask
