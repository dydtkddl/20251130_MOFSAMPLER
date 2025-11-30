#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sampling_mlp_experiment.py
- QM9 latent 공간에서 샘플링 전략 비교
    * Random sampling
    * Latent k-center (farthest-first) sampling
- 각 N (configs.SAMPLING_NS)에 대해:
    * 선택된 subset으로 ResidualMLP 학습
    * 고정 test set에서 MAE / RMSE / R2 평가
- 결과:
    * results_random.csv
    * results_kcenter.csv
    * curve_mae.png (N vs MAE, random vs kcenter)
"""

import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import configs
import utils
from models import ResidualMLP


# ============================================================
# k-center greedy (farthest-first) 샘플링
# ============================================================
def kcenter_greedy(
    X: np.ndarray,
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Latent 공간에서 farthest-first 방식의 k-center greedy.

    X: (N, d) latent vector 배열
    n_samples: 선택할 샘플 수
    return: 선택된 index 배열 (shape: (n_samples,))
    """
    N = X.shape[0]
    assert n_samples <= N, "n_samples must be <= N"

    rng = np.random.RandomState(seed)

    # 초기 center 랜덤 선택
    first_idx = rng.randint(0, N)
    centers = [first_idx]

    # 각 포인트가 가장 가까운 center까지의 거리 (초기에는 ∞)
    dist = np.full(N, np.inf, dtype=np.float64)

    # 첫 center에 대해 거리 갱신
    diff = X - X[first_idx]
    dist = np.minimum(dist, np.linalg.norm(diff, axis=1))

    for _ in tqdm(range(1, n_samples), desc="k-center greedy", ncols=100):
        # 현재 center들과 가장 먼 포인트 선택
        next_idx = int(np.argmax(dist))
        centers.append(next_idx)

        # 새 center 기준 거리 갱신
        diff = X - X[next_idx]
        dist = np.minimum(dist, np.linalg.norm(diff, axis=1))

    return np.array(centers, dtype=np.int64)


# ============================================================
# Numpy → Torch Dataset
# ============================================================
class NumpyDataset(Dataset):
    """단순 (X, y) numpy array용 Dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0]
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx]),
        )


# ============================================================
# ResidualMLP 학습 루프
#   - 여기서 y는 이미 (변환 + 스케일링된) space 상의 값
# ============================================================
def train_mlp(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    logger,
    desc: str = "MLP",
) -> ResidualMLP:
    """
    선택된 subset으로 ResidualMLP를 한 번 학습하고, val에서 best 모델 반환.
    (y는 이미 transform + scaling된 값이라고 가정)
    """
    device = utils.get_device()
    utils.set_seed(configs.SEED)  # 내부 seed 재고정 (재현성)

    input_dim = z_train.shape[1]
    model = ResidualMLP(in_dim=input_dim).to(device)

    train_dataset = NumpyDataset(z_train, y_train)
    val_dataset = NumpyDataset(z_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=configs.MLP_BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=configs.MLP_BATCH_SIZE,
        shuffle=False
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=configs.MLP_LR,
        weight_decay=configs.MLP_WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in tqdm(range(1, configs.MLP_EPOCHS + 1), desc=desc, ncols=100):
        # -----------------------------
        # Train
        # -----------------------------
        model.train()
        train_loss_sum = 0.0
        n_train_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            n_train_batches += 1

        train_loss = train_loss_sum / max(1, n_train_batches)

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                preds = model(xb)
                loss = criterion(preds, yb)

                val_loss_sum += loss.item()
                n_val_batches += 1

        val_loss = val_loss_sum / max(1, n_val_batches)

        if epoch % 50 == 0 or epoch == 1 or epoch == configs.MLP_EPOCHS:
            logger.info(
                f"[{desc}] Epoch {epoch}/{configs.MLP_EPOCHS} "
                f"TrainLoss={train_loss:.6f}  ValLoss={val_loss:.6f}"
            )

        # -----------------------------
        # Best 모델 갱신
        # -----------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    logger.info(f"[{desc}] Best ValLoss={best_val_loss:.6f}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model


# ============================================================
# Evaluation
#   - z_test는 (옵션) standardization된 latent
#   - y_test_raw는 "원래 HOMO 값" (변환/스케일링 전)
# ============================================================
@torch.no_grad()
def eval_mlp(
    model: ResidualMLP,
    z_test_proc: np.ndarray,
    y_test_raw: np.ndarray,
    y_scaler_params: dict,
    y_transform_mode: str,
) -> tuple:
    """
    model: ResidualMLP (transform+scaling된 space에서 학습된 상태)
    z_test_proc: (N_test, d) - (옵션) standardization 적용된 latent
    y_test_raw: (N_test,) - 원래 HOMO 값
    y_scaler_params: scale_y_fit 에서 얻은 params dict
    y_transform_mode: configs.Y_TRANSFORM 값
    """
    device = utils.get_device()
    model = model.to(device)
    model.eval()

    X_test = torch.from_numpy(z_test_proc.astype(np.float32)).to(device)
    y_true_raw = np.asarray(y_test_raw, dtype=np.float32)

    preds_scaled = model(X_test).cpu().numpy()  # (N_test, 1) or (N_test,)
    preds_scaled = preds_scaled.reshape(-1).astype(np.float32)

    # 1) scaling 역변환 (scaled → transformed space)
    y_pred_trans = utils.inverse_scale_y(preds_scaled, y_scaler_params)

    # 2) transform 역변환 (transformed → raw HOMO space)
    y_pred_raw = utils.inverse_transform_y(y_pred_trans, y_transform_mode)

    # metric은 "raw HOMO" 기준으로 계산
    mae = utils.mae(y_true_raw, y_pred_raw)
    rmse = utils.rmse(y_true_raw, y_pred_raw)
    r2 = utils.r2_numpy(y_true_raw, y_pred_raw)

    return mae, rmse, r2


# ============================================================
# 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Sampling experiment on QM9 latents (Random vs k-center)"
    )
    parser.add_argument("--seed", type=int, default=configs.RANDOM_SEED)
    parser.add_argument(
        "--npz_path",
        type=str,
        default=configs.LATENT_NPZ_PATH,
        help="Path to latents_qm9.npz"
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disables MAE curve plotting"
    )
    args = parser.parse_args()

    # Seed & Logger
    utils.set_seed(args.seed)
    log_file = os.path.join(configs.RESULT_DIR, "sampling_mlp_experiment.log")
    logger = utils.get_logger("sampling_mlp", log_file=log_file)

    logger.info("===== Sampling MLP Experiment (Random vs k-center) =====")
    logger.info(f"NPZ path: {args.npz_path}")
    logger.info(
        f"Y_TRANSFORM={configs.Y_TRANSFORM}, "
        f"Y_SCALING={configs.Y_SCALING}, "
        f"Z_STANDARDIZE={configs.Z_STANDARDIZE}"
    )

    # --------------------------------------------------------
    # Latent npz 로드
    # --------------------------------------------------------
    data = np.load(args.npz_path, allow_pickle=True)
    z_all: np.ndarray = data["z_all"]
    y_all: np.ndarray = data["y_all"]
    idx_train: np.ndarray = data["idx_train"]
    idx_val: np.ndarray = data["idx_val"]
    idx_test: np.ndarray = data["idx_test"]

    logger.info(
        f"z_all shape: {z_all.shape}, y_all shape: {y_all.shape} "
        f"(train/val/test = {len(idx_train)}/{len(idx_val)}/{len(idx_test)})"
    )

    # Target 분포 요약 (raw HOMO)
    hist_path = None
    if configs.Y_HIST_PLOT:
        hist_path = os.path.join(configs.RESULT_DIR, "homo_hist_raw.png")
    utils.describe_target(y_all, logger, name="HOMO", hist_out=hist_path, bins=configs.Y_HIST_BINS)

    # --------------------------------------------------------
    # Target 변환 (예: signed_log1p)
    # --------------------------------------------------------
    y_all_trans = utils.transform_y(y_all, configs.Y_TRANSFORM)

    # --------------------------------------------------------
    # train/val/test 분할 (변환된 y 기준)
    # --------------------------------------------------------
    z_train = z_all[idx_train]
    z_val = z_all[idx_val] if len(idx_val) > 0 else z_all[idx_train]
    z_test = z_all[idx_test]

    y_train_trans = y_all_trans[idx_train]
    y_val_trans = y_all_trans[idx_val] if len(idx_val) > 0 else y_all_trans[idx_train]
    y_test_trans = y_all_trans[idx_test]

    # raw HOMO 값 (metrics용)
    y_test_raw = y_all[idx_test]

    # --------------------------------------------------------
    # Latent Z standardization (옵션)
    # --------------------------------------------------------
    if configs.Z_STANDARDIZE:
        logger.info("Applying feature-wise standardization to latent Z (train-based).")
        z_train_proc, z_scaler_params = utils.standardize_features_fit(z_train)
        z_val_proc = utils.standardize_features_apply(z_val, z_scaler_params)
        z_test_proc = utils.standardize_features_apply(z_test, z_scaler_params)
    else:
        z_train_proc = z_train
        z_val_proc = z_val
        z_test_proc = z_test

    # --------------------------------------------------------
    # Target 스케일링 (train 기반)
    # --------------------------------------------------------
    y_train_scaled, y_scaler_params = utils.scale_y_fit(
        y_train_trans,
        mode=configs.Y_SCALING
    )
    y_val_scaled = utils.scale_y_apply(y_val_trans, y_scaler_params)
    y_test_scaled = utils.scale_y_apply(y_test_trans, y_scaler_params)

    logger.inf
