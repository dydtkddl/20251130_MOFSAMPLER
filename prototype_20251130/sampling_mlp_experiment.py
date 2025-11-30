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
# ============================================================

@torch.no_grad()
def eval_mlp(
    model: ResidualMLP,
    z_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple:
    device = utils.get_device()
    model = model.to(device)
    model.eval()

    X_test = torch.from_numpy(z_test.astype(np.float32)).to(device)
    y_true = torch.from_numpy(y_test.astype(np.float32)).to(device)

    preds = model(X_test)
    preds_np = preds.cpu().numpy()
    y_true_np = y_true.cpu().numpy()

    mae = utils.mae(y_true_np, preds_np)
    rmse = utils.rmse(y_true_np, preds_np)
    r2 = utils.r2_numpy(y_true_np, preds_np)

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

    # train/val/test 분할
    z_train = z_all[idx_train]
    y_train = y_all[idx_train]

    z_val = z_all[idx_val] if len(idx_val) > 0 else z_train
    y_val = y_all[idx_val] if len(idx_val) > 0 else y_train

    z_test = z_all[idx_test]
    y_test = y_all[idx_test]

    # 사용할 N 리스트 (train size 보다 큰 값은 제거)
    Ns = [n for n in configs.SAMPLING_NS if n <= len(z_train)]
    logger.info(f"Sampling Ns: {Ns}")

    # --------------------------------------------------------
    # k-center index 미리 계산 (가장 큰 N에 대해 한 번만)
    # --------------------------------------------------------
    max_N = max(Ns)
    logger.info(f"Precomputing k-center indices for max N={max_N}...")
    kcenter_indices_full = kcenter_greedy(
        z_train,
        n_samples=max_N,
        seed=args.seed + 1
    )

    # --------------------------------------------------------
    # Random / k-center 에 대해 실험
    # --------------------------------------------------------
    results_random = {
        "N": [],
        "MAE": [],
        "RMSE": [],
        "R2": [],
    }
    results_kcenter = {
        "N": [],
        "MAE": [],
        "RMSE": [],
        "R2": [],
    }

    rng = np.random.RandomState(args.seed + 123)

    # ---------------- Random ----------------
    logger.info("### Random sampling experiments ###")
    for N in Ns:
        logger.info(f"[Random] N = {N}")

        rand_idx = rng.choice(len(z_train), size=N, replace=False)
        z_sub = z_train[rand_idx]
        y_sub = y_train[rand_idx]

        model_r = train_mlp(
            z_sub, y_sub,
            z_val, y_val,
            logger,
            desc=f"Random N={N}"
        )

        mae_r, rmse_r, r2_r = eval_mlp(model_r, z_test, y_test)
        logger.info(
            f"[Random] N={N}  "
            f"MAE={mae_r:.5f}, RMSE={rmse_r:.5f}, R2={r2_r:.5f}"
        )

        results_random["N"].append(N)
        results_random["MAE"].append(mae_r)
        results_random["RMSE"].append(rmse_r)
        results_random["R2"].append(r2_r)

    # ---------------- k-center ----------------
    logger.info("### k-center sampling experiments ###")
    for N in Ns:
        logger.info(f"[k-center] N = {N}")

        kc_idx = kcenter_indices_full[:N]
        z_sub = z_train[kc_idx]
        y_sub = y_train[kc_idx]

        model_k = train_mlp(
            z_sub, y_sub,
            z_val, y_val,
            logger,
            desc=f"k-center N={N}"
        )

        mae_k, rmse_k, r2_k = eval_mlp(model_k, z_test, y_test)
        logger.info(
            f"[k-center] N={N}  "
            f"MAE={mae_k:.5f}, RMSE={rmse_k:.5f}, R2={r2_k:.5f}"
        )

        results_kcenter["N"].append(N)
        results_kcenter["MAE"].append(mae_k)
        results_kcenter["RMSE"].append(rmse_k)
        results_kcenter["R2"].append(r2_k)

    # --------------------------------------------------------
    # 결과 저장 (CSV + MAE 곡선)
    # --------------------------------------------------------
    os.makedirs(configs.RESULT_DIR, exist_ok=True)

    df_random = pd.DataFrame(results_random)
    df_kcenter = pd.DataFrame(results_kcenter)

    random_csv = os.path.join(configs.RESULT_DIR, "results_random.csv")
    kcenter_csv = os.path.join(configs.RESULT_DIR, "results_kcenter.csv")

    df_random.to_csv(random_csv, index=False)
    df_kcenter.to_csv(kcenter_csv, index=False)

    logger.info(f"Saved random results to: {random_csv}")
    logger.info(f"Saved k-center results to: {kcenter_csv}")

    # MAE 곡선 플롯
    if not args.no_plot:
        mae_dict = {
            "random": results_random["MAE"],
            "k-center": results_kcenter["MAE"],
        }
        mae_png = os.path.join(configs.RESULT_DIR, "curve_mae.png")
        utils.save_learning_curve(
            x_values=Ns,
            y_dict=mae_dict,
            out_png=mae_png,
            xlabel="# train samples",
            ylabel="MAE (HOMO)",
            title="Random vs k-center sampling (QM9 latents)",
        )
        logger.info(f"Saved MAE curve to: {mae_png}")

    logger.info("Sampling MLP experiment finished. ✅")


if __name__ == "__main__":
    main()
