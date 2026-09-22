import torch
import torch.nn as nn

class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=1, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        for _ in range(num_conv - 2):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            if act_type == 'prelu':
                self.body.append(nn.PReLU())
            elif act_type == 'relu':
                self.body.append(nn.ReLU(inplace=True))
            else:
                raise NotImplementedError(f'Activation {act_type} not implemented')
        self.body.append(nn.Conv2d(num_feat, num_out_ch * (upscale ** 2), 3, 1, 1) if upscale > 1 else nn.Conv2d(num_feat, num_out_ch, 3, 1, 1))
        self.act_type = act_type

    def forward(self, x):
        for layer in self.body:
            x = layer(x)
        if self.upscale > 1:
            x = torch.nn.functional.pixel_shuffle(x, self.upscale)
        return x 