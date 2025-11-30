#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
configs.py
- 공통 하이퍼파라미터 및 경로 설정 (Single Source of Truth)
- 다른 스크립트에서 import 해서 사용
"""

import os

# ==============================
# 프로젝트/경로 설정
# ==============================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 원본/전처리 데이터 루트 (QM9 등)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_ROOT, exist_ok=True)

# QM9 구조 descriptor 캐시 파일 경로
DESC_CACHE_PATH = os.path.join(DATA_ROOT, "qm9_struct_desc.pt")

# 체크포인트(encoder/decoder/MLP 등) 저장 경로
MODEL_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(MODEL_DIR, exist_ok=True)

# 결과(npz, csv, plots) 저장 경로
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# QM9 latent 전용 출력 디렉토리 (encoder 학습 + npz 저장)
OUT_DIR = os.path.join(PROJECT_ROOT, "qm9_latent")
os.makedirs(OUT_DIR, exist_ok=True)

# latents_qm9.npz 공통 경로 (train_encoder / sampling 실험에서 같이 사용)
LATENT_NPZ_PATH = os.path.join(OUT_DIR, "latents_qm9.npz")

# ==============================
# Seed / Random 관련
# ==============================
SEED = 42
RANDOM_SEED = SEED  # alias

# ==============================
# Encoder (EquivGNN) 설정
# ==============================
LATENT_DIM = 128

# EquivGNN 내부 node feature irreps (encoder에서 기본값으로 사용)
ENC_HIDDEN_IRREPS = "32x0e + 16x1o"

ENC_LR = 1e-3
ENC_BATCH_SIZE = 64
ENC_EPOCHS = 100
ENC_WEIGHT_DECAY = 1e-5

# ==============================
# Decoder (invariant head) 설정
# ==============================
DEC_HIDDEN_DIM = 256
DEC_LR = ENC_LR         # encoder와 동일하게 두어도 OK
DEC_WEIGHT_DECAY = ENC_WEIGHT_DECAY

# ==============================
# 구조 descriptor 설정
#  - train_encoder_qm9.py 의 build_structural_descriptor 와 동일하게 사용
# ==============================
DESC_DIM = 128          # 최종 descriptor dimension
DESC_NUM_BINS = 64      # pairwise distance histogram bin 수
DESC_R_MAX = 5.0        # histogram 상한 거리

# ==============================
# QM9 target 설정 (HOMO index)
# ==============================
# PyG QM9 dataset 기준: y[:, 2]가 HOMO (기본)
QM9_HOMO_TARGET_INDEX = 2

# ==============================
# Residual MLP (HOMO 예측용) 설정
# ==============================
MLP_HIDDEN_DIM = 256
MLP_LAYERS = 3
MLP_LR = 1e-3
MLP_BATCH_SIZE = 256
MLP_EPOCHS = 200
MLP_WEIGHT_DECAY = 0.0

# ==============================
# 데이터 Split & 샘플링 설정
# ==============================
TRAIN_VAL_TEST_SPLIT = (0.8, 0.1, 0.1)  # 전체 QM9 기준 비율

# 샘플링 실험에서 사용할 train size 리스트
SAMPLING_NS = [10, 20, 50, 100, 200, 500, 1000, 2000]

# ==============================
# Target(HOMO) 변환 / 스케일링 설정
# ==============================
# Y_TRANSFORM:
#   - "none"          : 변환 없음 (기본, 이전 코드와 동일)
#   - "signed_log1p"  : sign(y) * log(1 + |y|)
Y_TRANSFORM = "none"
Y_TRANSFORM = "signed_log1p"
# Y_SCALING:
#   - "none"      : 스케일링 없음 (기본, 이전 코드와 동일)
#   - "standard"  : (y - mean) / std
#   - "robust"    : (y - median) / IQR
Y_SCALING = "none"
Y_SCALING = "standard"

# Target 분포 로그 기록 시 히스토그램도 PNG로 저장할지 여부
Y_HIST_PLOT = False
Y_HIST_BINS = 50

# ==============================
# Latent Z 스케일링 옵션
# ==============================
# Z_STANDARDIZE:
#   - False : z 그대로 사용 (기본, 이전 코드와 동일)
#   - True  : train 기준 feature-wise standardization
Z_STANDARDIZE = False
Z_STANDARDIZE = True
# ==============================
# DataLoader / Logging 옵션
# ==============================
NUM_WORKERS = 4          # DataLoader num_workers
PIN_MEMORY = True        # GPU 사용 시 DataLoader 옵션

# 기본 로그 레벨 ("DEBUG", "INFO", "WARNING" ...)
LOG_LEVEL = "INFO"
