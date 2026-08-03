import torch
import torch.nn as nn
import torch.nn.functional as F
from .DualStreamFusion import DualStreamFusion

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        
        if self.training:   
            avg_out = self.fc2(self.relu1(self.fc1(nn.functional.avg_pool2d(x, kernel_size=(x.shape[2], x.shape[3])))))
            max_out = self.fc2(self.relu1(self.fc1(nn.functional.max_pool2d(x, kernel_size=(x.shape[2], x.shape[3])))))
        else:
            avg_out = self.fc2(self.relu1(self.fc1(x.flatten(2).mean(dim=2, keepdim=True).unsqueeze(3))))
            max_out = self.fc2(self.relu1(self.fc1(x.flatten(2).max(dim=2, keepdim=True)[0].unsqueeze(3))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out
        
        out += residual

        out = self.relu(out)
        return out




class ADGNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_block, # Res_CBAM_block, 
                 num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256]):
        super(ADGNet, self).__init__()
        self.relu = nn.ReLU(inplace = True)

        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2)
        self.up4   = nn.Upsample(scale_factor=4)
        self.up8   = nn.Upsample(scale_factor=8)
        self.up16  = nn.Upsample(scale_factor=16)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv4_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv3_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.conv2_1 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

        
        dim = nb_filter
        # Cross-modal Alignment Branch
        self.dualFusion0 = DualStreamFusion(
                        visual_dim=dim[0],  # both the visual input and for combining, num of channels
                        text_dim=512,  # l_in
                        fusion_dim=dim[0],  # key and value
                        dropout_global=0.1,
                        dropout_local=0.0
        )

        self.dualFusion1 = DualStreamFusion(
                        visual_dim=dim[1],  # both the visual input and for combining, num of channels
                        text_dim=512,  # l_in
                        fusion_dim=dim[1],  # key and value
                        dropout_global=0.1,
                        dropout_local=0.0
        )

        self.dualFusion2 = DualStreamFusion(
                        visual_dim=dim[2],  # both the visual input and for combining, num of channels
                        text_dim=512,  # l_in
                        fusion_dim=dim[2],  # key and value
                        dropout_global=0.1,
                        dropout_local=0.0
        )

        self.dualFusion3 = DualStreamFusion(
                        visual_dim=dim[3],  # both the visual input and for combining, num of channels
                        text_dim=512,  # l_in
                        fusion_dim=dim[3],  # key and value
                        dropout_global=0.1,
                        dropout_local=0.0
        )

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input, text_sequence,l_mask_fg,text_eot,l_mask_bg):
        x0_0 = self.conv0_0(input)#[b, c, h, w]
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))


        # Fuse local/global text features into visual skip connections
        x0_0_skip = self.dualFusion0(x0_0, text_sequence, text_eot, l_mask_fg,l_mask_bg)#[b, c, h, w]
        x1_0_skip = self.dualFusion1(x1_0, text_sequence, text_eot, l_mask_fg,l_mask_bg)#[b, c, h, w]
        x2_0_skip = self.dualFusion2(x2_0, text_sequence, text_eot, l_mask_fg,l_mask_bg)#[b, c, h, w]
        x3_0_skip = self.dualFusion3(x3_0, text_sequence, text_eot, l_mask_fg,l_mask_bg)#[b, c, h, w]


        x4_0 = self.conv4_0(self.pool(x3_0))

        xu3 = self.conv0_4_1x1(self.up16(x4_0))

        x3_1 = self.conv4_1(torch.cat([x3_0_skip, self.up(x4_0)], 1))
        xu2 = self.conv0_3_1x1(self.up8(x3_1))

        x2_1 = self.conv3_1(torch.cat([x2_0_skip, self.up(x3_1)], 1))
        xu1 = self.conv0_2_1x1(self.up4(x2_1))

        x1_1 = self.conv2_1(torch.cat([x1_0_skip, self.up(x2_1)], 1))
        xu0 = self.conv0_1_1x1(self.up(x1_1))

        x0_1 = self.conv1_1(torch.cat([x0_0_skip, self.up(x1_1)], 1))

        
        xf = self.conv0_1(xu3 + xu2 + xu1 + xu0 + x0_1)

        output = self.final(xf).sigmoid()
        return output

