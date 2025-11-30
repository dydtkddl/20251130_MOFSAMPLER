#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
models.py
- Equivariant GNN Encoder (QM9 구조 → latent z)
- Invariant Decoder (latent → 구조 descriptor)
- Residual MLP (latent → HOMO 예측)

설계 포인트
----------
- Encoder:
    * e3nn + PyG 기반
    * node irreps: configs.ENC_HIDDEN_IRREPS
    * 메시지패싱 블록은 Gate 없이 TP + Linear + ReLU + residual 구조 (버전 의존성 최소화)
- Decoder:
    * 단순 MLP로 invariant descriptor 재구성 (DESC_DIM과 호환)
- ResidualMLP:
    * latent_dim → hidden_dim → residual block 여러 개 → scalar 출력
"""

from typing import Optional

import torch
import torch.nn as nn

from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Linear

import configs


# ============================================================
# 1. Residual Block & Residual MLP (HOMO 예측용)
# ============================================================

class ResidualBlock(nn.Module):
    """단일 Residual MLP 블록: Linear -> ReLU -> skip connection."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        out = self.act(out)
        return x + out


class ResidualMLP(nn.Module):
    """
    구조 latent z를 입력으로 HOMO를 예측하는 Residual MLP.
    - 입력: (batch, latent_dim)
    - 출력: (batch,) HOMO 값
    """

    def __init__(
        self,
        in_dim: int = configs.LATENT_DIM,
        hidden_dim: int = configs.MLP_HIDDEN_DIM,
        num_layers: int = configs.MLP_LAYERS,
    ):
        super().__init__()

        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_layers)]
        )
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (batch, latent_dim)
        return: (batch,)
        """
        h = self.in_proj(z)
        h = torch.relu(h)

        for block in self.blocks:
            h = block(h)

        out = self.out_proj(h)  # (batch, 1)
        return out.squeeze(-1)


# ============================================================
# 2. Equivariant Message Passing Block
#   - E(3) equivariant TP + Linear + ReLU + Residual
# ============================================================

class EquivMPBlock(nn.Module):
    """
    단일 equivariant message passing 블록.
    - node_irreps -> node_irreps (residual)
    - edge_irreps: spherical harmonics(l<=1 등)
    """

    def __init__(
        self,
        node_irreps: o3.Irreps,
        edge_irreps: o3.Irreps,
    ):
        super().__init__()
        self.node_irreps = node_irreps
        self.edge_irreps = edge_irreps

        # 메시지 계산용 TensorProduct: h_i, Y_ij -> m_ij
        self.tp = FullyConnectedTensorProduct(
            node_irreps, edge_irreps, node_irreps
        )

        # 메시지 aggregation 이후 node-wise 업데이트용 선형 변환
        self.lin = Linear(node_irreps, node_irreps)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        sh: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (N, node_irreps.dim)
        edge_index: (2, E)
        sh: (E, edge_irreps.dim)
        """
        src, dst = edge_index

        # 메시지: m_ij = TP(x_i, Y_ij)
        m = self.tp(x[src], sh)  # (E, node_irreps.dim)

        # 수신 노드 dst에 sum aggregation
        N = x.shape[0]
        m_agg = x.new_zeros((N, self.node_irreps.dim))
        m_agg.index_add_(0, dst, m)

        # 선형 + ReLU
        h = self.lin(m_agg)
        h = torch.relu(h)

        # Residual
        return x + h


# ============================================================
# 3. radius_graph 대체 구현 (torch-cluster 의존 제거)
# ============================================================

def build_radius_graph(
    pos: torch.Tensor,
    batch: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """
    torch_cluster.radius_graph 대신 쓰는 간단한 구현.

    - pos:   (N, 3) 좌표
    - batch: (N,) 그래프 인덱스 (0 ~ B-1)
    - radius: cutoff 거리

    return:
        edge_index: (2, E) long tensor
    """
    device = pos.device
    batch = batch.to(device)
    radiussq = radius * radius

    edge_src_list = []
    edge_dst_list = []

    # 각 그래프별로 나눠서 O(n_i^2)로 처리 (QM9라 괜찮음)
    unique_batches = torch.unique(batch)
    for b in unique_batches:
        mask = (batch == b)
        idx = torch.nonzero(mask, as_tuple=False).view(-1)  # (n_b,)
        n_b = idx.numel()
        if n_b <= 1:
            continue

        coords = pos[idx]  # (n_b, 3)

        # pairwise 거리 제곱 (cdist 사용)
        dists = torch.cdist(coords, coords, p=2)  # (n_b, n_b)
        dists_sq = dists * dists

        # 자기 자신(i==j) 제외하고, radius 이하인 edge 선택
        mask_edge = (dists_sq <= radiussq) & (dists_sq > 0.0)
        src_rel, dst_rel = torch.nonzero(mask_edge, as_tuple=True)

        if src_rel.numel() == 0:
            continue

        src_abs = idx[src_rel]
        dst_abs = idx[dst_rel]

        edge_src_list.append(src_abs)
        edge_dst_list.append(dst_abs)

    if len(edge_src_list) == 0:
        return torch.empty(2, 0, dtype=torch.long, device=device)

    edge_src = torch.cat(edge_src_list)
    edge_dst = torch.cat(edge_dst_list)
    edge_index = torch.stack([edge_src, edge_dst], dim=0)  # (2, E)
    return edge_index


# ============================================================
# 4. Equivariant GNN Encoder (QM9 구조 → latent z)
# ============================================================

class EquivGNNEncoder(nn.Module):
    """
    QM9 분자 구조용 E(3)-equivariant encoder.

    입력:
        z: (N,) long, atomic numbers
        pos: (N, 3) float, 3D coordinates
        batch: (N,) long, PyG graph index (0 ~ B-1)

    출력:
        z_graph: (B, latent_dim) graph-level latent
    """

    def __init__(
        self,
        latent_dim: int = configs.LATENT_DIM,
        radius: float = 5.0,
        max_atomic_num: int = 100,
        num_layers: int = 3,
        node_irreps_str: str = configs.ENC_HIDDEN_IRREPS,
        lmax_edge: int = 1,
    ):
        super().__init__()
        self.radius = radius
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # ----------------------------------------------------
        # irreps 설정
        # ----------------------------------------------------
        self.node_irreps = o3.Irreps(node_irreps_str)
        self.edge_irreps = o3.Irreps.spherical_harmonics(lmax=lmax_edge)

        # ----------------------------------------------------
        # 원자 타입 embedding (scalar 0e) -> node_irreps로 사상
        # ----------------------------------------------------
        scalars_dim = 32
        self.atom_emb = nn.Embedding(max_atomic_num, scalars_dim)
        self.scalar_irreps = o3.Irreps(f"{scalars_dim}x0e")
        self.scalar2node = Linear(self.scalar_irreps, self.node_irreps)

        # ----------------------------------------------------
        # Equivariant MP blocks (multi-layer + residual)
        # ----------------------------------------------------
        self.layers = nn.ModuleList([
            EquivMPBlock(self.node_irreps, self.edge_irreps)
            for _ in range(num_layers)
        ])

        # ----------------------------------------------------
        # graph-level pooling 후 latent로 MLP 투사
        # ----------------------------------------------------
        self.readout = nn.Sequential(
            nn.Linear(self.node_irreps.dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        z: (N,) long
        pos: (N, 3)
        batch: (N,)
        """
        # 1) atomic number → scalar embedding
        x_scalar = self.atom_emb(z)  # (N, scalars_dim)

        # scalar irreps 텐서로 보고 node irreps로 사상
        x = self.scalar2node(x_scalar)  # (N, node_irreps.dim)

        # 2) radius graph 생성 (torch-cluster 없이)
        edge_index = build_radius_graph(pos, batch, self.radius)
        src, dst = edge_index

        # 3) edge 방향 기반 spherical harmonics (한 번 계산)
        rij = pos[dst] - pos[src]  # (E, 3)
        sh = o3.spherical_harmonics(
            self.edge_irreps,
            rij,
            normalize=True,
            normalization='component'
        )  # (E, edge_irreps.dim)

        # 4) multi-layer equivariant message passing
        for layer in self.layers:
            x = layer(x, edge_index, sh)

        # 5) batch-wise pooling (sum)
        B = int(batch.max().item()) + 1
        hg = x.new_zeros((B, self.node_irreps.dim))
        hg.index_add_(0, batch, x)

        # 6) latent projection
        z_graph = self.readout(hg)  # (B, latent_dim)
        return z_graph


# ============================================================
# 5. Invariant Decoder (latent → 구조 descriptor 재구성)
# ============================================================

class EquivDecoder(nn.Module):
    """
    구조-only self-supervised 학습용 invariant decoder.

    - 입력: graph-level latent (B, latent_dim)
    - 출력: 재구성 타깃 descriptor (B, desc_dim)

    desc_dim은 train_encoder_qm9.py에서 정의하는
    구조 descriptor (예: distance histogram) dimension과 맞춰야 한다.
    """

    def __init__(
        self,
        latent_dim: int = configs.LATENT_DIM,
        desc_dim: int = configs.DESC_DIM,
        hidden_dim: int = configs.DEC_HIDDEN_DIM,
        num_layers: int = 3,
    ):
        super().__init__()
        self.desc_dim = desc_dim

        layers = []
        in_dim = latent_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, desc_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, z_graph: torch.Tensor) -> torch.Tensor:
        """
        z_graph: (B, latent_dim)
        return: (B, desc_dim)
        """
        return self.net(z_graph)


__all__ = [
    "ResidualBlock",
    "ResidualMLP",
    "EquivMPBlock",
    "EquivGNNEncoder",
    "EquivDecoder",
]
