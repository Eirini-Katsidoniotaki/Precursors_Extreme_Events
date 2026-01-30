import numpy as np
from scipy.stats import gaussian_kde
import torch

def build_pdf_lookup(train_targets, bandwidth=None, grid_size=512):
    """Fit KDE on training targets and return a torch-friendly lookup function."""
    train_vals = train_targets.flatten()
    kde = gaussian_kde(train_vals, bw_method=bandwidth)

    # Build grid over range of training values
    ymin, ymax = np.min(train_vals), np.max(train_vals)
    grid = np.linspace(ymin, ymax, grid_size)
    pdf_vals = kde(grid)

    # Convert to tensors
    grid_t = torch.tensor(grid, dtype=torch.float32)
    pdf_t = torch.tensor(pdf_vals, dtype=torch.float32)

    def pdf_func(y):
        y_flat = y.detach().cpu().numpy().flatten()
        interp_vals = np.interp(y_flat, grid, pdf_vals)
        return torch.tensor(interp_vals, dtype=torch.float32, device=y.device).view_as(y)

    return pdf_func
