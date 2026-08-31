import torch
from torch import nn


class DepthLoss(nn.Module):
    def __init__(self, alpha=0.5, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        pred_log = torch.log1p(torch.clamp(pred, min=self.eps))
        target_log = torch.log1p(torch.clamp(target, min=self.eps))
        loss = self.l1(pred_log, target_log)

        if self.alpha == 0:
            return loss
        else:
            pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

            target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
            target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

            loss_grad = self.l1(pred_dx, target_dx) + self.l1(pred_dy, target_dy)
            return loss + self.alpha * loss_grad
