import torch
import torch.nn as nn
import torch.nn.functional as F


class BS_Branch(nn.Module):
    def __init__(self, visual_dim=128, text_dim=512, proj_dim=128, leaky=True):
        super(BS_Branch, self).__init__()

        self.mapping_visu = ConvBatchNormReLU(visual_dim, proj_dim, 1, 1, 0, 1, leaky=leaky)

        self.text_gate = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.Sigmoid()  
        )

    def forward(self, visual_feat, lang_feat):
        vis = self.mapping_visu(visual_feat)
        if lang_feat.dim() == 3: 
             lang_feat = lang_feat.squeeze(1) 
             
        gate = self.text_gate(lang_feat)
        gate = gate.unsqueeze(-1).unsqueeze(-1) 

        return vis * gate




class ConvBatchNormReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        leaky=False,
        relu=True,
        instance=False,
    ):
        super(ConvBatchNormReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False)

        if instance:
            self.bn = nn.InstanceNorm2d(num_features=out_channels)
        else:
            self.bn = nn.BatchNorm2d(
                    num_features=out_channels, eps=1e-5, momentum=0.999, affine=True
                )

        if leaky:
            self.relu = nn.LeakyReLU(0.1)
        elif relu:
            self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x