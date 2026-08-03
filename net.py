from torch import nn
from model.ADGNet import ADGNet
from loss import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cal_loss = SoftIoULoss()
        self.model = ADGNet()

    def forward(self, img, text_sequence, l_mask_fg, text_eot, l_mask_bg):
        return self.model(img, text_sequence, l_mask_fg, text_eot, l_mask_bg)

    def loss(self, pred, gt_mask, epoch=None):
        if epoch is not None:
            loss = self.cal_loss(pred, gt_mask, epoch)
        else:
            loss = self.cal_loss(pred, gt_mask)
        return loss
