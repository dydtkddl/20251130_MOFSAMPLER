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
"""

import os
import random
import logging
from logging.handlers import RotatingFileHandler
from typing import Tuple, Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

import configs  # LOG_LEVEL, SEED 등 사용


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
