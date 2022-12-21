import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .coordatt import CoordAtt


#---------------------------------------#
#   Multi-scale Feature Fusion Module
#   使用CoordAtt注意力
#---------------------------------------#
class MSFFBlock(nn.Module):
    def __init__(self, in_channel):
        super(MSFFBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.attn = CoordAtt(in_channel, in_channel)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel // 2, in_channel // 2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x_conv = self.conv1(x) # [B, 128, 64, 64] -> [B, 128, 64, 64]
        x_att = self.attn(x)   # [B, 128, 64, 64] -> [B, 128, 64, 64]

        x = x_conv * x_att     # [B, 128, 64, 64] * [B, 128, 64, 64] = [B, 128, 64, 64]
        x = self.conv2(x)      # [B, 128, 64, 64] -> [B, 64, 64, 64]
        return x


#-----------------------------------------------------------#
#   Multi-scale Feature Fusion Module + Spatial Attention
#-----------------------------------------------------------#
class MSFF(nn.Module):
    def __init__(self):
        super(MSFF, self).__init__()
        self.blk1 = MSFFBlock(128)
        self.blk2 = MSFFBlock(256)
        self.blk3 = MSFFBlock(512)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.upconv32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        )
        self.upconv21 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, features):
        # features = [[B, 128, 64, 64], [B, 256, 32, 32], [B, 512, 16, 16]]
        f1, f2, f3 = features

        # MSFF Module
        f1_k = self.blk1(f1) # [B, 128, 64, 64] -> [B,  64, 64, 64]
        f2_k = self.blk2(f2) # [B, 256, 32, 32] -> [B, 128, 32, 32]
        f3_k = self.blk3(f3) # [B, 512, 16, 16] -> [B, 256, 16, 16]

        f2_f = f2_k + self.upconv32(f3_k) # [B, 128, 32, 32] + ([B, 256, 16, 16] -> [B, 128, 32, 32]) = [B, 128, 32, 32]
        f1_f = f1_k + self.upconv21(f2_f) # [B,  64, 64, 64] + ([B, 128, 32, 32] -> [B,  64, 64, 64]) = [B,  64, 64, 64]

        # spatial attention
        # mask 只要通道后面一半的数据
        m3 = f3[:,256:,...].mean(dim=1, keepdim=True)                     # [B, 512, 16, 16] get [B, 256, 16, 16] -> [B, 1, 16, 16]
        m2 = f2[:,128:,...].mean(dim=1, keepdim=True) * self.upsample(m3) # [B, 256, 32, 32] get [B, 128, 32, 32] -> [B, 1, 32, 32] * ([B, 1, 16, 16] -> [B, 1, 32, 32]) -> [B, 1, 32, 32]
        m1 = f1[:, 64:,...].mean(dim=1, keepdim=True) * self.upsample(m2) # [B, 128, 64, 64] get [B, 64, 64, 64] -> [B, 1, 64, 64] * ([B, 1, 32, 32] -> [B, 1, 64, 64]) -> [B, 1, 64, 64]
        # features * mask
        f1_out = f1_f * m1 # [B,  64, 64, 64] * [B, 1, 64, 64] = [B,  64, 64, 64]
        f2_out = f2_f * m2 # [B, 128, 32, 32] * [B, 1, 32, 32] = [B, 128, 32, 32]
        f3_out = f3_k * m3 # [B, 256, 16, 16] * [B, 1, 16, 16] = [B, 256, 16, 16]

        return [f1_out, f2_out, f3_out] # [B, 64, 64, 64], [B, 128, 32, 32], [B, 256, 16, 16]