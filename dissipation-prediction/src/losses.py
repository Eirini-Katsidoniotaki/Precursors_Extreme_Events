import torch
import torch.nn as nn


class OutputWeightedMAE(nn.Module):
    """
    Output-weighted Mean Absolute Error (MAE_OW) as used in Barthel & Sapsis (AIAA 2023).
    Errors on rare outputs (low PDF values) are weighted higher.

    Args:
        pdf_func (callable): A function f(y) that returns the estimated PDF at y (torch.Tensor).
                             Should output nonzero values (use smoothing or epsilon).
        eps (float): Small constant to avoid division by zero.
    """
    def __init__(self, pdf_func, eps=1e-6):
        super(OutputWeightedMAE, self).__init__()
        self.pdf_func = pdf_func
        self.eps = eps

    def forward(self, y_pred, y_true):
        # y_pred, y_true: (batch, horizon) or (batch, horizon, 1)
        if y_true.dim() > 2:
            y_true = y_true.squeeze(-1)
            y_pred = y_pred.squeeze(-1)

        # PDF evaluated at the true values
        pdf_vals = self.pdf_func(y_true.detach())
        # Prevent division by zero
        weights = 1.0 / (pdf_vals + self.eps)

        # Weighted MAE
        loss = torch.mean(weights * torch.abs(y_pred - y_true))
        return loss



