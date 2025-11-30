#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_encoder_qm9.py
- QM9 구조-only self-supervised 학습
- EquivGNNEncoder + Invariant Decoder 로 구조 descriptor 재구성
- 학습 완료 후 전체 QM9에 대한 latent z 및 HOMO y를 npz로 저장

출력:
- encoder/decoder 체크포인트: checkpoints/encoder_struct.pt, decoder_struct.pt
- latent npz: configs.LATENT_NPZ_PATH (latents_qm9.npz)
    * z_all: (N_mol, LATENT_DIM)
    * y_all: (N_mol,)
    * idx_train, idx_val, idx_test: split index
"""

import os
import argparse
from typing import Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Subset

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader as PyGDataLoader

import configs
import utils
from models import EquivGNNEncoder, EquivDecoder


# ============================================================
# 구조 descriptor: pairwise distance histogram 기반
# ============================================================

def build_structural_descriptor(
    pos: torch.Tensor,
    num_bins: int,
    r_max: float,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    단일 분자에 대한 구조 descriptor 생성.

    아이디어:
    - 모든 atom pair i<j 에 대해 거리 d_ij 계산
    - [0, r_max] 구간을 num_bins로 등분한 histogram 생성
    - histogram 값을 정규화 후, hist^2와 concat → 총 2 * num_bins 차원
    - configs.DESC_DIM == 2 * num_bins 인지 체크 (불일치면 assert)

    pos: (N, 3), float
    return: (DESC_DIM,) float
    """
    num_bins_cfg = configs.DESC_NUM_BINS
    desc_dim_cfg = configs.DESC_DIM
    assert desc_dim_cfg == 2 * num_bins_cfg, (
        f"DESC_DIM({desc_dim_cfg}) must be 2 * DESC_NUM_BINS({num_bins_cfg})"
    )

    pos = pos.to(device=device, dtype=torch.float32)
    N = pos.size(0)

    if N < 2:
        # atom이 1개인 edge case → 0벡터 반환
        hist = torch.zeros(num_bins, device=device)
    else:
        # (N,3) → pairwise 거리 (N*(N-1)/2,)
        # torch.pdist는 CPU에서 동작하므로 device 강제시 CPU로 보내도 됨.
        dists = torch.pdist(pos, p=2)

        # [0, r_max] 구간에서 histogram
        # torch.histc는 [min, max] 포함, bins 개
        hist = torch.histc(
            dists,
            bins=num_bins,
            min=0.0,
            max=r_max
        )

    # 정규화 (합 1) + 제곱 특징
    if hist.sum() > 0:
        hist = hist / hist.sum()

    hist_sq = hist ** 2
    desc = torch.cat([hist, hist_sq], dim=0)  # (2 * num_bins,)
    return desc


# ============================================================
# 학습 루프 (encoder + decoder)
# ============================================================

def train_one_epoch(
    encoder: EquivGNNEncoder,
    decoder: EquivDecoder,
    loader: PyGDataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    encoder.train()
    decoder.train()

    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Train encoder", ncols=100):
        batch = batch.to(device)
        optimizer.zero_grad()

        # 구조 latent
        z_graph = encoder(batch.z, batch.pos, batch.batch)
        # 구조 descriptor target (graph-wise)
        target_desc = batch.struct_desc.to(device)

        pred_desc = decoder(z_graph)

        loss = criterion(pred_desc, target_desc)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def eval_one_epoch(
    encoder: EquivGNNEncoder,
    decoder: EquivDecoder,
    loader: PyGDataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Val encoder", ncols=100):
        batch = batch.to(device)

        z_graph = encoder(batch.z, batch.pos, batch.batch)
        target_desc = batch.struct_desc.to(device)
        pred_desc = decoder(z_graph)

        loss = criterion(pred_desc, target_desc)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


# ============================================================
# 전체 latent 추출
# ============================================================

@torch.no_grad()
def extract_latents(
    encoder: EquivGNNEncoder,
    loader: PyGDataLoader,
    device: torch.device
) -> np.ndarray:
    encoder.eval()

    all_latents = []

    for batch in tqdm(loader, desc="Extract latents", ncols=100):
        batch = batch.to(device)
        z_graph = encoder(batch.z, batch.pos, batch.batch)  # (B, latent_dim)
        all_latents.append(z_graph.cpu().numpy())

    z_all = np.concatenate(all_latents, axis=0)  # (N_mol, latent_dim)
    return z_all


# ============================================================
# 메인
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train EquivGNN encoder on QM9")
    parser.add_argument("--epochs", type=int, default=configs.ENC_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=configs.ENC_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=configs.ENC_LR)
    parser.add_argument("--weight_decay", type=float, default=configs.ENC_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=configs.SEED)
    parser.add_argument("--latent_dim", type=int, default=configs.LATENT_DIM)
    args = parser.parse_args()

    # ----------------------------------------------
    # Seed & Device & Logger
    # ----------------------------------------------
    utils.set_seed(args.seed)
    device = utils.get_device()

    log_file = os.path.join(configs.RESULT_DIR, "train_encoder_qm9.log")
    logger = utils.get_logger("train_encoder_qm9", log_file=log_file)

    logger.info("===== Train EquivGNN Encoder on QM9 (self-supervised) =====")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {args.epochs}, BatchSize: {args.batch_size}, LR: {args.lr}")
    logger.info(f"Latent dim: {args.latent_dim}")

    # ----------------------------------------------
    # QM9 Dataset 로드
    # ----------------------------------------------
    qm9_root = os.path.join(configs.DATA_ROOT, "qm9")
    logger.info(f"Loading QM9 dataset from: {qm9_root}")
    dataset = QM9(root=qm9_root)

    N_data = len(dataset)
    logger.info(f"QM9 total molecules: {N_data}")

    # ----------------------------------------------
    # 구조 descriptor 사전 계산 + Data에 저장
    # ----------------------------------------------
    logger.info("Precomputing structural descriptors for all molecules...")

    for data in tqdm(dataset, desc="Build descriptors", ncols=100):
        # pos: (N_i, 3)
        desc = build_structural_descriptor(
            data.pos,
            num_bins=configs.DESC_NUM_BINS,
            r_max=configs.DESC_R_MAX,
            device=torch.device("cpu"),
        )
        # Data 객체에 바로 저장 (PyG가 graph-wise 속성으로 잘 collate 해줌)
        data.struct_desc = desc

    logger.info("Finished structural descriptor precomputation.")

    # ----------------------------------------------
    # Train / Val / Test split
    # ----------------------------------------------
    idx_train, idx_val, idx_test = utils.train_val_test_split_indices(
        N_data,
        configs.TRAIN_VAL_TEST_SPLIT,
        seed=args.seed
    )
    logger.info(
        f"Split sizes - train: {len(idx_train)}, "
        f"val: {len(idx_val)}, test: {len(idx_test)}"
    )

    train_subset = Subset(dataset, idx_train)
    val_subset = Subset(dataset, idx_val)
    full_loader = PyGDataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=configs.NUM_WORKERS, pin_memory=configs.PIN_MEMORY
    )

    train_loader = PyGDataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=configs.NUM_WORKERS, pin_memory=configs.PIN_MEMORY
    )

    val_loader = None
    if len(idx_val) > 0:
        val_loader = PyGDataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False,
            num_workers=configs.NUM_WORKERS, pin_memory=configs.PIN_MEMORY
        )

    # ----------------------------------------------
    # 모델/옵티마 초기화
    # ----------------------------------------------
    encoder = EquivGNNEncoder(latent_dim=args.latent_dim).to(device)
    decoder = EquivDecoder(latent_dim=args.latent_dim, desc_dim=configs.DESC_DIM).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = -1

    encoder_ckpt_path = os.path.join(configs.MODEL_DIR, "encoder_struct.pt")
    decoder_ckpt_path = os.path.join(configs.MODEL_DIR, "decoder_struct.pt")
    os.makedirs(configs.MODEL_DIR, exist_ok=True)

    # ----------------------------------------------
    # 학습 루프
    # ----------------------------------------------
    for epoch in range(1, args.epochs + 1):
        logger.info(f"=== Epoch {epoch}/{args.epochs} ===")

        train_loss = train_one_epoch(
            encoder, decoder, train_loader,
            optimizer, device, criterion
        )
        logger.info(f"Train loss: {train_loss:.6f}")

        if val_loader is not None:
            val_loss = eval_one_epoch(
                encoder, decoder, val_loader, device, criterion
            )
            logger.info(f"Val loss:   {val_loss:.6f}")
        else:
            val_loss = train_loss

        # Best 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(encoder.state_dict(), encoder_ckpt_path)
            torch.save(decoder.state_dict(), decoder_ckpt_path)
            logger.info(
                f"  >> New best model saved (epoch {epoch}, val_loss={val_loss:.6f})"
            )

    logger.info(
        f"Training finished. Best epoch: {best_epoch}, best val_loss: {best_val_loss:.6f}"
    )

    # ----------------------------------------------
    # Best encoder 로드 후 전체 latent 추출
    # ----------------------------------------------
    logger.info("Loading best encoder checkpoint and extracting latents...")
    encoder.load_state_dict(torch.load(encoder_ckpt_path, map_location=device))

    z_all = extract_latents(encoder, full_loader, device)  # (N, latent_dim)

    # ----------------------------------------------
    # QM9 HOMO 값 y_all 추출
    # ----------------------------------------------
    logger.info("Extracting HOMO values from QM9 dataset...")
    target_idx = configs.QM9_HOMO_TARGET_INDEX

    y_all_list = []
    for data in dataset:
        # PyG QM9: data.y shape (1, 19) or (19,)
        y = data.y.view(-1)
        y_homo = y[target_idx].item()
        y_all_list.append(y_homo)

    y_all = np.array(y_all_list, dtype=np.float32)

    assert z_all.shape[0] == y_all.shape[0] == N_data, \
        "z_all / y_all length mismatch"

    # ----------------------------------------------
    # latent + target + split index 저장
    # ----------------------------------------------
    npz_path = configs.LATENT_NPZ_PATH
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)

    logger.info(f"Saving latents and splits to: {npz_path}")
    np.savez_compressed(
        npz_path,
        z_all=z_all,
        y_all=y_all,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )
    logger.info("All done. ✅")


if __name__ == "__main__":
    main()
