import torch
import torch.nn as nn

NOOP_ACTION = 4


class MarginLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1.0

    def forward(self, z_t, z_t1, a_t):
        sq_diff = torch.sum((z_t - z_t1) ** 2, dim=-1)

        noop_mask = a_t == NOOP_ACTION

        # NOOP: game state unchanged, only sqrl moved — pull encodings together
        noop_loss = sq_diff
        # Non-NOOP: true state changed — push encodings apart (standard margin)
        margin_loss = torch.relu(1 - sq_diff / self.eps ** 2)

        per_sample_loss = torch.where(noop_mask, noop_loss, margin_loss)
        return per_sample_loss.mean()