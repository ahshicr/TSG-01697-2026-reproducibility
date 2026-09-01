from __future__ import annotations

import math

import torch
from torch import nn


class GraphGRUForecaster(nn.Module):
    """Compact graph recurrent forecaster for zone-level rollout states."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        horizon: int,
        adj,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        adj_tensor = torch.as_tensor(adj, dtype=torch.float32)
        self.register_buffer("adj", adj_tensor)
        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * n_features),
        )

    def graph_mix(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,bjf->bif", self.adj, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, nodes, _ = x.shape
        h = x.new_zeros(batch, nodes, self.input_proj.out_features)
        for step in range(x.shape[1]):
            z = self.input_proj(self.graph_mix(x[:, step]))
            h_in = self.graph_mix(h)
            h = self.gru(z.reshape(batch * nodes, -1), h_in.reshape(batch * nodes, -1))
            h = h.reshape(batch, nodes, -1)
            h = self.norm(h)
        out = self.head(self.dropout(h))
        out = out.reshape(batch, nodes, self.horizon, -1).permute(0, 2, 1, 3)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalized_laplacian_smoothness(pred: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    mixed = torch.einsum("ij,bhjf->bhif", adj, pred)
    return torch.mean((pred - mixed) ** 2)


def masked_mape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target) / torch.clamp(torch.abs(target), min=eps))


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
