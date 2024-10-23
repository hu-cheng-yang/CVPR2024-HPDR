import torch
import torch.nn.functional as F
from torch import nn

import hyptorch.nn as hypnn
from hyptorch.pmath import dist_matrix


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias)


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, mode="BN"):
        super(Conv_block, self).__init__()
        self.conv = conv3x3(in_channels, out_channels)
        if mode == "BN":
            self.bn = nn.BatchNorm2d(out_channels)
        elif mode == "IN":
            print("Use IN")
            self.bn = nn.InstanceNorm2d(out_channels)
            print(self.bn)
        else:
            raise (Exception("mode must be BN or IN"))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Basic_block(nn.Module):

    def __init__(self, in_channels, out_channels, mode="BN"):
        super(Basic_block, self).__init__()
        self.conv_block1 = Conv_block(in_channels, 128, mode=mode)
        self.conv_block2 = Conv_block(128, 196, mode=mode)
        self.conv_block3 = Conv_block(196, out_channels, mode=mode)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.max_pool(x)
        return x


class FeatExtractor(nn.Module):
    def __init__(self, in_channels=6, mode="BN"):
        super(FeatExtractor, self).__init__()
        self.inc = Conv_block(in_channels, 64, mode=mode)
        self.down1 = Basic_block(64, 128, mode=mode)
        self.down2 = Basic_block(128, 128, mode=mode)
        self.down3 = Basic_block(128, 128, mode=mode)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        dx1 = self.inc(x)
        dx2 = self.down1(dx1)
        dx3 = self.down2(dx2)
        dx4 = self.down3(dx3)

        re_dx2 = F.adaptive_avg_pool2d(dx2, 32)
        re_dx3 = F.adaptive_avg_pool2d(dx3, 32)
        catfeat = torch.cat([re_dx2, re_dx3, dx4], 1)
        feat = self.global_pool(catfeat)
        feat = feat.view(feat.shape[0], -1)

        return catfeat, feat


class FeatEmbedder(nn.Module):
    def __init__(self, in_channels=384, out_channels=8):
        super(FeatEmbedder, self).__init__()

        self.conv_block1 = Conv_block(in_channels, 128)
        self.conv_block2 = Conv_block(128, 256)
        self.conv_block3 = Conv_block(256, out_channels)
        self.max_pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(out_channels)
        self.projector = hypnn.ToPoincare(c=0.01, ball_dim=out_channels, clip_r=2.3, riemannian=True)
        self.fc = hypnn.HyperbolicMLR(ball_dim=out_channels, n_classes=2, c=0.01)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.max_pool(x)
        x = self.conv_block2(x)
        x = self.max_pool(x)
        x = self.conv_block3(x)
        x = self.global_pool(x)
        p = x.view(x.size(0), -1)
        p = self.norm(p)
        p = self.projector(p)
        x = self.fc(p)
        return x, p


class Encoder(nn.Module):
    def __init__(self, in_channels=3, mode="IN"):
        super(Encoder, self).__init__()
        self.extractor = FeatExtractor(in_channels, mode)
        self.classifier = FeatEmbedder(out_channels=256)

    def forward(self, x):
        feat, _ = self.extractor(x)
        c, p = self.classifier(feat)

        return c, None, p