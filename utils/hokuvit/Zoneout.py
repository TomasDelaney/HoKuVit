import torch
import torch.nn as nn


class Zoneout(nn.Module):
    """Stateless zoneout for iterative updates"""

    def __init__(self, p=0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("Zoneout probability has to be between 0 and 1.")
        self.p = p

    def forward(self, x_old, x_new):
        """
        Args:
            x_old: previous state
            x_new: proposed new state
        Returns:
            mixture of old and new states
        """
        if not self.training or self.p == 0.0:
            return x_new

        mask = torch.rand_like(x_new) < self.p
        return torch.where(mask, x_old, x_new)