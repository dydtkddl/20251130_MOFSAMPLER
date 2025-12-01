#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils.py
- seed 고정
- metric 계산 (numpy / torch)
- logging 세팅 (RotatingFileHandler 포함)
- device 헬퍼
- train/val/test split
- 학습 곡선 플롯 (샘플 수 vs MAE)
- target 변환/스케일링 유틸
- 구조 기반 descriptor 유틸:
    * per-atom CN / neighbor distance stats
    * node-level 구조 피쳐 패킹
    * RDF / ADF / global shape descriptor
"""

import os
import math
import random
import logging
from logging.handlers import RotatingFileHandler
from typing import Tuple, Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

import configs  # LOG_LEVEL, SEED, DESC_* 등 사용


# ==============================
# Seed 고정
# ==============================
def set_seed(seed: int) -> None:
    """random / numpy / torch / cuda 모두 seed 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 완전 재현성을 원하면 deterministic 설정
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==============================
# Device 헬퍼
# ==============================
def get_device() -> torch.device:
    """cuda 사용 가능하면 cuda, 아니면 cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================
# Metric 계산 (numpy 버전)
# ==============================
def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


# ---- alias (다른 스크립트에서 mae, rmse, r2_numpy 이름으로도 사용 가능하게) ----
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Alias for mae_np (편의용)."""
    return mae_np(y_true, y_pred)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Alias for rmse_np (편의용)."""
    return rmse_np(y_true, y_pred)


def r2_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Alias for r2_np (이름 취향용)."""
    return r2_np(y_true, y_pred)


# ==============================
# Metric 계산 (torch 버전)
# ==============================
def mae_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(y_true - y_pred)).item())


def rmse_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item())


def r2_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    if ss_tot.item() == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


# ==============================
# Logging 세팅
# ==============================
def get_logger(
    name: str,
    log_file: str = None,
    level: str = None
) -> logging.Logger:
    """
    RotatingFileHandler + 콘솔 핸들러를 사용하는 logger 생성.
    - name: logger 이름
    - log_file: 파일로도 남기고 싶으면 경로 지정 (None이면 파일 로깅 없음)
    - level: "DEBUG" / "INFO" 등 (None이면 configs.LOG_LEVEL 사용)
    """
    logger = logging.getLogger(name)

    # 이미 핸들러가 있으면 중복 추가 방지
    if logger.handlers:
        return logger

    if level is None:
        level = getattr(configs, "LOG_LEVEL", "INFO")

    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 콘솔 핸들러
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 파일 핸들러 (선택)
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = RotatingFileHandler(
            log_file,
            maxBytes=10_000_000,
            backupCount=5
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ==============================
# Train/Val/Test Split
# ==============================
def train_val_test_split_indices(
    N: int,
    ratios: Tuple[float, float, float],
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    전체 N개 인덱스를 ratios 비율로 나눈 train/val/test 인덱스 반환.
    """
    assert len(ratios) == 3, "ratios must be (train, val, test)"
    train_r, val_r, test_r = ratios
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "ratios must sum to 1"

    rng = np.random.RandomState(seed)
    indices = np.arange(N)
    rng.shuffle(indices)

    n_train = int(N * train_r)
    n_val = int(N * val_r)

    idx_train = indices[:n_train]
    idx_val = indices[n_train:n_train + n_val]
    idx_test = indices[n_train + n_val:]

    return idx_train, idx_val, idx_test


# ==============================
# 학습 곡선 플롯 (샘플 수 vs MAE)
# ==============================
def save_learning_curve(
    x_values: List[int],
    y_dict: Dict[str, List[float]],
    out_png: str,
    xlabel: str = "# train samples",
    ylabel: str = "MAE",
    title: str = "Sampling comparison"
) -> None:
    """
    x_values: N 리스트 (예: [10, 20, 50, ...])
    y_dict: {"random": [..], "latent": [..]} 형식
    """
    plt.figure(figsize=(8, 5))
    for label, ys in y_dict.items():
        plt.plot(x_values, ys, marker="o", label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()

    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ==============================
# Target(HOMO 등) 분포 요약 + 히스토그램
# ==============================
def describe_target(
    y: np.ndarray,
    logger: logging.Logger,
    name: str = "target",
    hist_out: str = None,
    bins: int = 50
) -> None:
    """
    y 분포에 대한 기본 통계 + (옵션) 히스토그램 PNG 저장.
    """
    y = np.asarray(y, dtype=np.float64)
    mean = np.mean(y)
    std = np.std(y)
    y_min = np.min(y)
    y_max = np.max(y)
    p1, p50, p99 = np.percentile(y, [1, 50, 99])

    logger.info(
        f"{name} stats (raw): "
        f"mean={mean:.4f}, std={std:.4f}, min={y_min:.4f}, max={y_max:.4f}, "
        f"p1={p1:.4f}, p50={p50:.4f}, p99={p99:.4f}"
    )

    if hist_out is not None:
        out_dir = os.path.dirname(hist_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        plt.figure(figsize=(6, 4))
        plt.hist(y, bins=bins, alpha=0.75)
        plt.xlabel(name)
        plt.ylabel("Count")
        plt.title(f"Histogram of {name}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(hist_out, dpi=300)
        plt.close()
        logger.info(f"{name} histogram saved to: {hist_out}")


# ==============================
# Target 변환 (예: signed_log1p)
# ==============================
def transform_y(y: np.ndarray, mode: str = "none") -> np.ndarray:
    """
    y → y_trans (training space 변환)
    mode:
        - "none"         : 그대로
        - "signed_log1p" : sign(y) * log(1 + |y|)
    """
    y = np.asarray(y, dtype=np.float32)

    if mode is None or mode.lower() == "none":
        return y.copy()

    mode = mode.lower()
    if mode == "signed_log1p":
        return np.sign(y) * np.log1p(np.abs(y)).astype(np.float32)

    # 알 수 없는 모드는 경고 후 그대로 반환
    print(f"[WARN] transform_y: unknown mode '{mode}', using 'none'")
    return y.copy()


def inverse_transform_y(y_trans: np.ndarray, mode: str = "none") -> np.ndarray:
    """
    y_trans → y_raw (역변환)
    mode:
        - "none"         : 그대로
        - "signed_log1p" : sign(y) * (exp(|y|) - 1)
    """
    y_trans = np.asarray(y_trans, dtype=np.float32)

    if mode is None or mode.lower() == "none":
        return y_trans.copy()

    mode = mode.lower()
    if mode == "signed_log1p":
        return np.sign(y_trans) * (np.expm1(np.abs(y_trans))).astype(np.float32)

    print(f"[WARN] inverse_transform_y: unknown mode '{mode}', using 'none'")
    return y_trans.copy()


# ==============================
# Target 스케일링 (standard / robust)
# ==============================
def scale_y_fit(
    y: np.ndarray,
    mode: str = "none"
) -> Tuple[np.ndarray, Dict]:
    """
    y (1D) 를 주면, mode에 따라 스케일링된 y와 파라미터 dict를 반환.
    mode:
        - "none"     : 그대로
        - "standard" : (y - mean) / std
        - "robust"   : (y - median) / IQR
    """
    y = np.asarray(y, dtype=np.float32)

    if mode is None or mode.lower() == "none":
        params = {"mode": "none"}
        return y.copy(), params

    mode = mode.lower()
    if mode == "standard":
        mean = float(np.mean(y))
        std = float(np.std(y))
        if std < 1e-8:
            std = 1.0
        y_scaled = (y - mean) / std
        params = {"mode": "standard", "mean": mean, "std": std}
        return y_scaled.astype(np.float32), params

    if mode == "robust":
        median = float(np.median(y))
        q1, q3 = np.percentile(y, [25, 75])
        iqr = float(q3 - q1)
        if iqr < 1e-8:
            iqr = 1.0
        y_scaled = (y - median) / iqr
        params = {"mode": "robust", "median": median, "iqr": iqr}
        return y_scaled.astype(np.float32), params

    print(f"[WARN] scale_y_fit: unknown mode '{mode}', using 'none'")
    params = {"mode": "none"}
    return y.copy(), params


def scale_y_apply(
    y: np.ndarray,
    params: Dict
) -> np.ndarray:
    """
    fit된 params를 사용하여 y에 동일 스케일링 적용.
    """
    y = np.asarray(y, dtype=np.float32)
    mode = params.get("mode", "none").lower()

    if mode == "none":
        return y.copy()

    if mode == "standard":
        mean = params["mean"]
        std = params["std"]
        return ((y - mean) / std).astype(np.float32)

    if mode == "robust":
        median = params["median"]
        iqr = params["iqr"]
        return ((y - median) / iqr).astype(np.float32)

    print(f"[WARN] scale_y_apply: unknown mode '{mode}', using 'none'")
    return y.copy()


def inverse_scale_y(
    y_scaled: np.ndarray,
    params: Dict
) -> np.ndarray:
    """
    스케일링된 y_scaled를 params 기준으로 원래 스케일로 복원.
    """
    y_scaled = np.asarray(y_scaled, dtype=np.float32)
    mode = params.get("mode", "none").lower()

    if mode == "none":
        return y_scaled.copy()

    if mode == "standard":
        mean = params["mean"]
        std = params["std"]
        return (y_scaled * std + mean).astype(np.float32)

    if mode == "robust":
        median = params["median"]
        iqr = params["iqr"]
        return (y_scaled * iqr + median).astype(np.float32)

    print(f"[WARN] inverse_scale_y: unknown mode '{mode}', using 'none'")
    return y_scaled.copy()


# ==============================
# Latent Z standardization (옵션)
# ==============================
def standardize_features_fit(
    X: np.ndarray,
    eps: float = 1e-8
) -> Tuple[np.ndarray, Dict]:
    """
    X: (N, d) 에 대해 feature-wise standardization.
    return: X_scaled, {"mean": mean(d,), "std": std(d,)}
    """
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    X_scaled = (X - mean) / std
    params = {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }
    return X_scaled.astype(np.float32), params


def standardize_features_apply(
    X: np.ndarray,
    params: Dict,
    eps: float = 1e-8
) -> np.ndarray:
    """
    fit된 params를 사용하여 X에 동일한 standardization 적용.
    """
    X = np.asarray(X, dtype=np.float32)
    mean = params["mean"]
    std = params["std"]
    std = np.where(std < eps, 1.0, std)
    X_scaled = (X - mean) / std
    return X_scaled.astype(np.float32)


# ============================================================
# 구조 기반 descriptor 유틸
#   - per-atom CN / distance stats
#   - node-level scalar feature 패킹
#   - RDF / ADF / global shape descriptor
# ============================================================

def _ensure_tensor_pos(pos: torch.Tensor) -> torch.Tensor:
    """
    pos를 float32 torch.Tensor로 보장.
    - 입력이 numpy array여도 허용.
    - device 정보는 유지(텐서면 그대로).
    """
    if isinstance(pos, torch.Tensor):
        return pos.to(dtype=torch.float32)
    # numpy or list → CPU tensor
    return torch.as_tensor(pos, dtype=torch.float32)


def compute_cn_and_dist_stats(
    pos: torch.Tensor,
    radius: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    각 atom에 대해 neighbor 기반 CN / min/mean/max distance 계산.

    입력:
        pos   : (N, 3) 좌표 (torch.Tensor or np.ndarray)
        radius: neighbor cutoff

    출력 (모두 (N,) on pos.device):
        cn     : coordination number (float)
        mean_d : neighbor 거리 평균 (이웃 없으면 0)
        min_d  : neighbor 거리 최소 (이웃 없으면 0)
        max_d  : neighbor 거리 최대 (이웃 없으면 0)
    """
    pos = _ensure_tensor_pos(pos)
    device = pos.device
    N = pos.size(0)

    if N == 0:
        zeros = torch.zeros(0, device=device)
        return zeros, zeros, zeros, zeros

    # pairwise 거리 행렬 (N, N)
    dists = torch.cdist(pos, pos, p=2)  # 자기 자신 포함, 0

    cn_list = []
    mean_list = []
    min_list = []
    max_list = []

    for i in range(N):
        # 자신 제외 & cutoff 이내 neighbor
        mask = (dists[i] > 0.0) & (dists[i] <= radius)
        neigh_d = dists[i][mask]

        if neigh_d.numel() == 0:
            cn_list.append(torch.tensor(0.0, device=device))
            mean_list.append(torch.tensor(0.0, device=device))
            min_list.append(torch.tensor(0.0, device=device))
            max_list.append(torch.tensor(0.0, device=device))
        else:
            cn_list.append(torch.tensor(float(neigh_d.numel()), device=device))
            mean_list.append(torch.mean(neigh_d))
            min_list.append(torch.min(neigh_d))
            max_list.append(torch.max(neigh_d))

    cn = torch.stack(cn_list)
    mean_d = torch.stack(mean_list)
    min_d = torch.stack(min_list)
    max_d = torch.stack(max_list)

    return cn, mean_d, min_d, max_d


def pack_node_struct_feats(
    cn: torch.Tensor,
    mean_d: torch.Tensor,
    min_d: torch.Tensor,
    max_d: torch.Tensor
) -> torch.Tensor:
    """
    per-atom scalar들을 NODE_STRUCT_DIM에 맞게 concat.

    - configs.NODE_USE_CN / NODE_USE_LOCAL_DENSITY / NODE_USE_DIST_STATS
      flag에 따라 포함할 피쳐가 결정됨.
    - 현재는 raw 값 그대로 사용 (encoder가 scaling 학습).
      필요하면 나중에 여기서 간단한 정규화 추가 가능.

    입력:
        cn, mean_d, min_d, max_d: (N,)

    출력:
        node_feats: (N, NODE_STRUCT_DIM)
    """
    # 모두 tensor로 통일
    cn = _ensure_tensor_pos(cn).view(-1)
    mean_d = _ensure_tensor_pos(mean_d).view(-1)
    min_d = _ensure_tensor_pos(min_d).view(-1)
    max_d = _ensure_tensor_pos(max_d).view(-1)

    device = cn.device
    N = cn.size(0)

    feats: List[torch.Tensor] = []

    if configs.NODE_USE_CN:
        feats.append(cn.view(N, 1))  # (N, 1)

    if configs.NODE_USE_LOCAL_DENSITY:
        # mean_d가 0인 경우 density=0
        eps = 1e-8
        local_density = 1.0 / (mean_d + eps)
        feats.append(local_density.view(N, 1))

    if configs.NODE_USE_DIST_STATS:
        feats.append(min_d.view(N, 1))
        feats.append(mean_d.view(N, 1))
        feats.append(max_d.view(N, 1))

    if len(feats) == 0:
        # 구조 피쳐 사용 안하면 0-dim 텐서 반환
        return torch.zeros(N, 0, device=device)

    node_feats = torch.cat(feats, dim=-1)  # (N, NODE_STRUCT_DIM 예상)
    assert node_feats.shape[1] == configs.NODE_STRUCT_DIM, (
        f"NODE_STRUCT_DIM mismatch: got {node_feats.shape[1]}, "
        f"expected {configs.NODE_STRUCT_DIM}"
    )
    return node_feats


def compute_rdf_descriptor(
    pos: torch.Tensor,
    num_bins: int = None,
    r_max: float = None
) -> torch.Tensor:
    """
    RDF (Radial Distribution Function) 기반 descriptor 생성.

    - 모든 atom pair i<j 에 대해 거리 d_ij 계산
    - [0, r_max] 구간을 num_bins로 등분한 histogram 생성
    - histogram을 정규화 후 hist^2와 concat → (2 * num_bins,)

    입력:
        pos     : (N, 3) (tensor or numpy)
        num_bins: 사용 bin 수 (None이면 configs.DESC_NUM_BINS_R)
        r_max   : 최대 거리 (None이면 configs.DESC_R_MAX)

    출력:
        desc_rdf: (2 * num_bins,)
    """
    pos = _ensure_tensor_pos(pos)
    device = pos.device

    if num_bins is None:
        num_bins = configs.DESC_NUM_BINS_R
    if r_max is None:
        r_max = configs.DESC_R_MAX

    N = pos.size(0)
    if N < 2:
        hist = torch.zeros(num_bins, device=device)
    else:
        dists = torch.pdist(pos, p=2)  # (N*(N-1)/2,)
        if dists.numel() == 0:
            hist = torch.zeros(num_bins, device=device)
        else:
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


def compute_adf_descriptor(
    pos: torch.Tensor,
    num_bins: int = None,
    radius: float = None
) -> torch.Tensor:
    """
    ADF (Angular Distribution Function) 기반 descriptor.

    - 각 center atom j에 대해, cutoff 이내 neighbor i, k 선택
    - angle(i-j-k) ∈ [0, π] 계산
    - angle들의 histogram + hist^2 → (2 * num_bins,)

    입력:
        pos     : (N, 3)
        num_bins: angle bin 수 (None이면 configs.DESC_NUM_BINS_ANGLE)
        radius  : neighbor cutoff (None이면 configs.NODE_RADIUS)

    출력:
        desc_adf: (2 * num_bins,)
    """
    pos = _ensure_tensor_pos(pos)
    device = pos.device
    N = pos.size(0)

    if num_bins is None:
        num_bins = configs.DESC_NUM_BINS_ANGLE
    if radius is None:
        radius = configs.NODE_RADIUS

    if N < 3:
        hist = torch.zeros(num_bins, device=device)
        hist_sq = hist ** 2
        return torch.cat([hist, hist_sq], dim=0)

    # pairwise 거리 행렬
    dists = torch.cdist(pos, pos, p=2)

    angle_list: List[torch.Tensor] = []

    for j in range(N):
        # center j 기준 neighbor index
        neigh_mask = (dists[j] > 0.0) & (dists[j] <= radius)
        neigh_idx = torch.nonzero(neigh_mask, as_tuple=False).view(-1)

        if neigh_idx.numel() < 2:
            continue

        # 이웃 쌍 (i, k) 순회 (작은 N이라 이중 루프 충분)
        for a in range(neigh_idx.numel()):
            i = neigh_idx[a]
            v_ij = pos[i] - pos[j]
            for b in range(a + 1, neigh_idx.numel()):
                k = neigh_idx[b]
                v_kj = pos[k] - pos[j]

                # angle between v_ij and v_kj
                dot = torch.dot(v_ij, v_kj)
                norm_ij = torch.norm(v_ij) + 1e-8
                norm_kj = torch.norm(v_kj) + 1e-8
                cos_theta = (dot / (norm_ij * norm_kj)).clamp(-1.0, 1.0)
                theta = torch.acos(cos_theta)  # [0, π]
                angle_list.append(theta)

    if len(angle_list) == 0:
        hist = torch.zeros(num_bins, device=device)
        hist_sq = hist ** 2
        return torch.cat([hist, hist_sq], dim=0)

    angles = torch.stack(angle_list)  # (M,)
    hist = torch.histc(
        angles,
        bins=num_bins,
        min=0.0,
        max=math.pi
    )

    if hist.sum() > 0:
        hist = hist / hist.sum()

    hist_sq = hist ** 2
    desc = torch.cat([hist, hist_sq], dim=0)  # (2 * num_bins,)
    return desc


def compute_shape_descriptor(pos: torch.Tensor) -> torch.Tensor:
    """
    Global shape descriptor (4D):
        - inertia tensor eigenvalues (3)
        - radius of gyration (1)

    입력:
        pos: (N, 3)

    출력:
        shape: (4,) tensor on pos.device
    """
    pos = _ensure_tensor_pos(pos)
    device = pos.device
    N = pos.size(0)

    if N == 0:
        return torch.zeros(4, device=device)

    # center-of-mass 기준으로 이동
    com = pos.mean(dim=0, keepdim=True)
    r = pos - com  # (N, 3)

    # radius of gyration
    r2 = (r ** 2).sum(dim=1)  # (N,)
    rg = torch.sqrt(torch.mean(r2) + 1e-12)

    # inertia tensor: I = sum_i [ (r_i·r_i) I - r_i r_i^T ]
    I = torch.zeros(3, 3, device=device)
    eye3 = torch.eye(3, device=device)
    for i in range(N):
        ri = r[i].view(3, 1)
        r2_i = (ri.view(-1) ** 2).sum()
        I = I + (r2_i * eye3 - ri @ ri.t())

    # eigenvalues (실수, 정렬)
    evals = torch.linalg.eigvalsh(I)  # (3,)
    evals, _ = torch.sort(evals.real)

    shape = torch.cat([evals, rg.view(1)], dim=0)  # (4,)
    return shape
