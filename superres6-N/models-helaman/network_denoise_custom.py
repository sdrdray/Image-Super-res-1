import torch
import torch.nn as nn
import numpy as np
import cv2

class ChannelMixer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, in_ch, 3, 1, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class LKConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 17, 1, 8)
    def forward(self, x):
        return self.conv(x)

class Attn(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 1, 1)
    def forward(self, x):
        return self.conv(x)

class Refine(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 1)
    def forward(self, x):
        return self.conv(x)

class Norm(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.LayerNorm([ch, 1, 1])
    def forward(self, x):
        # LayerNorm expects (N, C, H, W), but normalizes over last dims
        return self.norm(x)

class DenoiseCustomNet(nn.Module):
    def __init__(self, num_blocks=30, in_ch=3, feat_ch=64, mid_ch=128):
        super().__init__()
        self.feats = nn.ModuleList()
        self.feats.append(nn.Conv2d(in_ch, feat_ch, 3, 1, 1))  # feats.0
        for i in range(1, num_blocks+1):
            block = nn.ModuleDict({
                'channel_mixer': ChannelMixer(feat_ch, mid_ch),
                'lk': LKConv(16),
                'attn': Attn(feat_ch),
                'refine': Refine(feat_ch),
                'norm': nn.BatchNorm2d(feat_ch)
            })
            self.feats.append(block)
        self.out_conv = nn.Conv2d(feat_ch, in_ch, 3, 1, 1)  # feats.30
    def forward(self, x):
        x = self.feats[0](x)
        for block in self.feats[1:]:
            x = block['channel_mixer'](x)
            # LKConv expects 16 channels, so only apply if shape matches
            if x.shape[1] == 16:
                x = block['lk'](x)
            x = block['attn'](x)
            x = block['refine'](x)
            x = block['norm'](x)
        x = self.out_conv(x)
        return x 