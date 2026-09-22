import torch
import torch.nn as nn

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4):
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        # upsampling layers
        if scale == 4:
            self.upsample = nn.Sequential(
                nn.Conv2d(nf, nf, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(nf, nf, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        elif scale == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(nf, nf, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.upconv2 = None
        else:
            raise NotImplementedError('Only scale=2 or 4 is supported.')
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.conv_body(self.body(fea))
        fea = fea + trunk
        if self.scale == 4:
            fea = self.lrelu(self.upconv1(torch.nn.functional.interpolate(fea, scale_factor=2, mode='nearest')))
            fea = self.lrelu(self.upconv2(torch.nn.functional.interpolate(fea, scale_factor=2, mode='nearest')))
        elif self.scale == 2:
            fea = self.lrelu(self.upconv1(torch.nn.functional.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))
        return out 