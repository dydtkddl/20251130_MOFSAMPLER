#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sampling_mlp_experiment.py
- QM9 latent 공간에서 샘플링 전략 비교
    * Random sampling (uniform)
    * Latent k-center (farthest-first) sampling
    * Latent density-weighted random sampling (low-density 영역 우선 탐색)
- 각 N (configs.SAMPLING_NS)에 대해:
    * 선택된 subset으로 ResidualMLP 학습 (y: transform + scaling space)
    * 고정 test set에서 MAE / RMSE / R2 평가 (raw HOMO space 기준)
- 결과:
    * results_random.csv
    * results_kcenter.csv
    * results_density.csv
    * curve_mae.png (N vs MAE, random vs kcenter vs density)
"""

import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.neighbors import NearestNeighbors  # density용

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
# Latent density 기반 inverse-density weights
#  - low-density 영역일수록 샘플링 확률↑
# ============================================================
def compute_inverse_density_weights(
    Z: np.ndarray,
    k: int = None,
    alpha: float = None,
) -> np.ndarray:
    """
    Z: (N, d) latent (보통 z_train_proc: 표준화된 latent)
    k: k-NN 이웃 수 (None이면 configs.DENSITY_KNN_K 또는 10)
    alpha: sparsity 강조 정도 (None이면 configs.DENSITY_ALPHA 또는 1.0)

    return: probs (N,) - 합이 1인 확률 벡터
    """
    if k is None:
        k = getattr(configs, "DENSITY_KNN_K", 10)
    if alpha is None:
        alpha = getattr(configs, "DENSITY_ALPHA", 1.0)

    N = Z.shape[0]
    assert N > k, "N must be > k for density estimation"

    # k+1: 자기 자신 + k neighbors → 자기 자신 제외
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nbrs.fit(Z)
    dists, _ = nbrs.kneighbors(Z)  # (N, k+1)
    dists = dists[:, 1:]           # (N, k)

    # k-NN 평균 거리 → local sparsity proxy
    mean_d = dists.mean(axis=1)  # (N,)

    eps = 1e-8
    density = 1.0 / (mean_d + eps)      # local density
    inv_density = 1.0 / (density + eps) # low-density → 큰 값

    # sparsity 강조
    weights = inv_density ** alpha

    # 극단 outlier 보호용 clip (옵션)
    p99 = np.percentile(weights, 99.0)
    weights = np.minimum(weights, p99)

    probs = weights / weights.sum()
    return probs


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
#   - y는 이미 (변환 + 스케일링된) space 상의 값
#   - early stopping 포함
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
    Early stopping:
        - configs.MAX_EPOCHS 번까지 학습
        - configs.EARLY_STOP_PATIENCE 동안 val 개선 없으면 중단
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

    max_epochs = getattr(configs, "MAX_EPOCHS", configs.MLP_EPOCHS)
    patience = getattr(configs, "EARLY_STOP_PATIENCE", 50)

    best_val_loss = float("inf")
    best_state_dict = None
    no_improve_epochs = 0

    for epoch in tqdm(range(1, max_epochs + 1), desc=desc, ncols=100):
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

        if epoch == 1 or epoch % 50 == 0 or epoch == max_epochs:
            logger.info(
                f"[{desc}] Epoch {epoch}/{max_epochs} "
                f"TrainLoss={train_loss:.6f}  ValLoss={val_loss:.6f}"
            )

        # -----------------------------
        # Early stopping & best 모델 갱신
        # -----------------------------
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.info(
                    f"[{desc}] Early stopping at epoch {epoch} "
                    f"(no val improvement for {patience} epochs)."
                )
                break

    logger.info(f"[{desc}] Best ValLoss={best_val_loss:.6f}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model


# ============================================================
# Evaluation
#   - z_test_proc: (옵션) standardization된 latent
#   - y_test_raw: "원래 HOMO 값" (변환/스케일링 전)
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
        description="Sampling experiment on QM9 latents (Random vs k-center vs density-weighted)"
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

    logger.info("===== Sampling MLP Experiment (Random vs k-center vs density) =====")
    logger.info(f"NPZ path: {args.npz_path}")
    logger.info(
        f"Y_TRANSFORM={configs.Y_TRANSFORM}, "
        f"Y_SCALING={configs.Y_SCALING}, "
        f"Z_STANDARDIZE={configs.Z_STANDARDIZE}, "
        f"MAX_EPOCHS={getattr(configs, 'MAX_EPOCHS', configs.MLP_EPOCHS)}, "
        f"EARLY_STOP_PATIENCE={getattr(configs, 'EARLY_STOP_PATIENCE', 50)}"
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
    if getattr(configs, "Y_HIST_PLOT", False):
        hist_path = os.path.join(configs.RESULT_DIR, "homo_hist_raw.png")
    utils.describe_target(
        y_all,
        logger,
        name="HOMO",
        hist_out=hist_path,
        bins=getattr(configs, "Y_HIST_BINS", 50),
    )

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
    y_test_scaled = utils.scale_y_apply(y_test_trans, y_scaler_params)  # (직접 쓰진 않지만 일관성 유지)

    logger.info(
        f"Target scaling mode={y_scaler_params.get('mode', 'none')} "
        f"(Y_TRANSFORM={configs.Y_TRANSFORM})"
    )

    # 사용할 N 리스트 (train size 보다 큰 값은 제거)
    Ns = [n for n in configs.SAMPLING_NS if n <= len(z_train_proc)]
    logger.info(f"Sampling Ns: {Ns}")

    # --------------------------------------------------------
    # k-center index 미리 계산 (가장 큰 N에 대해 한 번만)
    # --------------------------------------------------------
    max_N = max(Ns)
    logger.info(f"Precomputing k-center indices for max N={max_N}...")
    kcenter_indices_full = kcenter_greedy(
        z_train_proc,
        n_samples=max_N,
        seed=args.seed + 1
    )

    # --------------------------------------------------------
    # density-weighted sampling용 확률 벡터 한 번 계산
    # --------------------------------------------------------
    logger.info("Computing density-based sampling probabilities on latent space...")
    density_probs = compute_inverse_density_weights(
        z_train_proc,
        k=getattr(configs, "DENSITY_KNN_K", 10),
        alpha=getattr(configs, "DENSITY_ALPHA", 1.0),
    )
    logger.info(
        f"Density probs: min={density_probs.min():.6e}, "
        f"max={density_probs.max():.6e}, "
        f"mean={density_probs.mean():.6e}"
    )

    # --------------------------------------------------------
    # Random / k-center / density 에 대해 실험
    # --------------------------------------------------------
    results_random = {
        "N": [],
        "MAE": [],
        "RMSE": [],
        "R2": [],
        "Y_TRANSFORM": [],
        "Y_SCALING": [],
        "Z_STANDARDIZE": [],
    }
    results_kcenter = {
        "N": [],
        "MAE": [],
        "RMSE": [],
        "R2": [],
        "Y_TRANSFORM": [],
        "Y_SCALING": [],
        "Z_STANDARDIZE": [],
    }
    results_density = {
        "N": [],
        "MAE": [],
        "RMSE": [],
        "R2": [],
        "Y_TRANSFORM": [],
        "Y_SCALING": [],
        "Z_STANDARDIZE": [],
    }

    rng = np.random.RandomState(args.seed + 123)

    # ---------------- Random (uniform) ----------------
    logger.info("### Random (uniform) sampling experiments ###")
    for N in Ns:
        logger.info(f"[Random] N = {N}")

        rand_idx = rng.choice(len(z_train_proc), size=N, replace=False)
        z_sub = z_train_proc[rand_idx]
        y_sub = y_train_scaled[rand_idx]

        model_r = train_mlp(
            z_sub, y_sub,
            z_val_proc, y_val_scaled,
            logger,
            desc=f"Random N={N}"
        )

        mae_r, rmse_r, r2_r = eval_mlp(
            model_r,
            z_test_proc,
            y_test_raw,
            y_scaler_params,
            configs.Y_TRANSFORM
        )
        logger.info(
            f"[Random] N={N}  "
            f"MAE={mae_r:.5f}, RMSE={rmse_r:.5f}, R2={r2_r:.5f}"
        )

        results_random["N"].append(N)
        results_random["MAE"].append(mae_r)
        results_random["RMSE"].append(rmse_r)
        results_random["R2"].append(r2_r)
        results_random["Y_TRANSFORM"].append(configs.Y_TRANSFORM)
        results_random["Y_SCALING"].append(configs.Y_SCALING)
        results_random["Z_STANDARDIZE"].append(configs.Z_STANDARDIZE)

    # ---------------- k-center ----------------
    logger.info("### k-center sampling experiments ###")
    for N in Ns:
        logger.info(f"[k-center] N = {N}")

        kc_idx = kcenter_indices_full[:N]
        z_sub = z_train_proc[kc_idx]
        y_sub = y_train_scaled[kc_idx]

        model_k = train_mlp(
            z_sub, y_sub,
            z_val_proc, y_val_scaled,
            logger,
            desc=f"k-center N={N}"
        )

        mae_k, rmse_k, r2_k = eval_mlp(
            model_k,
            z_test_proc,
            y_test_raw,
            y_scaler_params,
            configs.Y_TRANSFORM
        )
        logger.info(
            f"[k-center] N={N}  "
            f"MAE={mae_k:.5f}, RMSE={rmse_k:.5f}, R2={r2_k:.5f}"
        )

        results_kcenter["N"].append(N)
        results_kcenter["MAE"].append(mae_k)
        results_kcenter["RMSE"].append(rmse_k)
        results_kcenter["R2"].append(r2_k)
        results_kcenter["Y_TRANSFORM"].append(configs.Y_TRANSFORM)
        results_kcenter["Y_SCALING"].append(configs.Y_SCALING)
        results_kcenter["Z_STANDARDIZE"].append(configs.Z_STANDARDIZE)

    # ---------------- Density-weighted ----------------
    logger.info("### Density-weighted sampling experiments ###")
    for N in Ns:
        logger.info(f"[Density] N = {N}")

        dens_idx = rng.choice(
            len(z_train_proc),
            size=N,
            replace=False,
            p=density_probs
        )
        z_sub = z_train_proc[dens_idx]
        y_sub = y_train_scaled[dens_idx]

        model_d = train_mlp(
            z_sub, y_sub,
            z_val_proc, y_val_scaled,
            logger,
            desc=f"Density N={N}"
        )

        mae_d, rmse_d, r2_d = eval_mlp(
            model_d,
            z_test_proc,
            y_test_raw,
            y_scaler_params,
            configs.Y_TRANSFORM
        )
        logger.info(
            f"[Density] N={N}  "
            f"MAE={mae_d:.5f}, RMSE={rmse_d:.5f}, R2={r2_d:.5f}"
        )

        results_density["N"].append(N)
        results_density["MAE"].append(mae_d)
        results_density["RMSE"].append(rmse_d)
        results_density["R2"].append(r2_d)
        results_density["Y_TRANSFORM"].append(configs.Y_TRANSFORM)
        results_density["Y_SCALING"].append(configs.Y_SCALING)
        results_density["Z_STANDARDIZE"].append(configs.Z_STANDARDIZE)

    # --------------------------------------------------------
    # 결과 저장 (CSV + MAE 곡선)
    # --------------------------------------------------------
    os.makedirs(configs.RESULT_DIR, exist_ok=True)

    df_random = pd.DataFrame(results_random)
    df_kcenter = pd.DataFrame(results_kcenter)
    df_density = pd.DataFrame(results_density)

    random_csv = os.path.join(configs.RESULT_DIR, "results_random.csv")
    kcenter_csv = os.path.join(configs.RESULT_DIR, "results_kcenter.csv")
    density_csv = os.path.join(configs.RESULT_DIR, "results_density.csv")

    df_random.to_csv(random_csv, index=False)
    df_kcenter.to_csv(kcenter_csv, index=False)
    df_density.to_csv(density_csv, index=False)

    logger.info(f"Saved random results to: {random_csv}")
    logger.info(f"Saved k-center results to: {kcenter_csv}")
    logger.info(f"Saved density results to: {density_csv}")

    # MAE 곡선 플롯
    if not args.no_plot:
        mae_dict = {
            "random": results_random["MAE"],
            "k-center": results_kcenter["MAE"],
            "density": results_density["MAE"],
        }
        mae_png = os.path.join(configs.RESULT_DIR, "curve_mae.png")
        utils.save_learning_curve(
            x_values=Ns,
            y_dict=mae_dict,
            out_png=mae_png,
            xlabel="# train samples",
            ylabel="MAE (HOMO)",
            title="Random vs k-center vs density sampling (QM9 latents)",
        )
        logger.info(f"Saved MAE curve to: {mae_png}")

    logger.info("Sampling MLP experiment finished. ✅")


if __name__ == "__main__":
    main()
