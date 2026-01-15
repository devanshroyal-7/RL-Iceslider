import torch
import torch.nn as nn

class MarginLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 0.1

    def margin_loss(self, z_t, z_t1):
        sq_diff = torch.sum((z_t - z_t1)**2, dim=-1)
        m_loss = torch.relu(1 - sq_diff/self.eps**2)

        return m_loss