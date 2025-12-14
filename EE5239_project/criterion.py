import torch
from torch.nn.modules.loss import _Loss
from config import Config

class SegmentationLoss(_Loss):

    def __init__(self, dice_weight = 1, bce_weight = 0):
        super(SegmentationLoss, self).__init__()
        cfg = Config().parse()
        self.BCE_loss = torch.nn.BCEWithLogitsLoss()
        self.dice_weight = cfg.dice_weight
        self.bce_weight = cfg.bce_weight

    def dice_loss(self,y_pred, y_true, eps=1e-8):
        intersection = torch.sum(torch.mul(y_pred, y_true)) 
        union = torch.sum(torch.mul(y_pred, y_pred)) + torch.sum(torch.mul(y_true, y_true)) + eps
        dice = 2 * intersection / union 
        dice_loss = 1 - dice
        return dice_loss

    def forward(self, y_pred, y_true, eps=1e-8):
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        y_pred_prob = torch.sigmoid(y_pred)
        
        dice = self.dice_loss(y_pred_prob, y_true, eps)
        bce = self.BCE_loss(y_pred, y_true)

        return self.dice_weight * dice + self.bce_weight * bce