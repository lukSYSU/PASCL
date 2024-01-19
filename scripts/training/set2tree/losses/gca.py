from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

class GCALoss(nn.Module):
    def __init__(self, temperature=0.4):
        super(GCALoss, self).__init__()
        self.temperature = temperature

    def forward(self, out: torch.Tensor, aug_out: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None):
        """Compute loss for model using loss function in paper
        GCA: Graph Contrastive Learning with Adaptive Augmentation.
        https://github.com/CRIPAC-DIG/GCA/blob/main/pGRACE/model.py#L67

        Returns:
            A loss scalar.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if batch_size is None:
            l1 = self.semi_loss(out, aug_out)
            l2 = self.semi_loss(aug_out, out)
        else:
            l1 = self.batched_semi_loss(out, aug_out, batch_size)
            l2 = self.batched_semi_loss(aug_out, out, batch_size)

        loss = (l1 + l2) * 0.5
        loss = loss.mean() if mean else loss.sum()

        return loss

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.temperature)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.temperature)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)
