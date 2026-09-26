"""Official equal-weight probability BCE + per-image soft Dice (smooth=1)."""
import torch
from torch import nn
from torch.nn import functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
        super().__init__()
        self.bce_weight, self.dice_weight, self.smooth = bce_weight, dice_weight, smooth

    def forward(self, logits, target):
        # Run outside autocast in float32: BCELoss on sigmoid matches source code.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            prob, target = logits.float().sigmoid(), target.float()
            bce = F.binary_cross_entropy(prob, target)
            p, y = prob.flatten(1), target.flatten(1)
            dice = (2 * (p * y).sum(1) + self.smooth) / (p.sum(1) + y.sum(1) + self.smooth)
            return self.bce_weight * bce + self.dice_weight * (1 - dice.mean())
