#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
configs.py
- 공통 하이퍼파라미터 및 경로 설정 (Single Source of Truth)
- 다른 스크립트에서 import 해서 사용

구조 기반 확장 사항:
1) Node-level 구조 피쳐 설정 (NODE_* 관련)
2) Graph-level descriptor 설정 (RDF + ADF + Shape)
3) Latent density 기반 샘플링 설정 (DENSITY_*)
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
# Node-level 구조 피쳐 설정
#  - EquivGNNEncoder 입력으로 들어가는 scalar node feature
# ==============================
# 어떤 per-atom 구조 피쳐를 쓸지 flag로 관리
NODE_USE_CN = True              # coordination number
NODE_USE_LOCAL_DENSITY = True   # 평균 거리 역수 기반 local density
NODE_USE_DIST_STATS = True      # min/mean/max neighbor distance

# CN/neighbor 계산에 사용할 cutoff (encoder radius와 맞추는 걸 권장)
NODE_RADIUS = 5.0

# NODE_STRUCT_DIM 자동 계산 (flag 조합에 따라 결정)
_node_struct_dim = 0
if NODE_USE_CN:
    _node_struct_dim += 1               # cn
if NODE_USE_LOCAL_DENSITY:
    _node_struct_dim += 1               # local_density
if NODE_USE_DIST_STATS:
    _node_struct_dim += 3               # min_d, mean_d, max_d

NODE_STRUCT_DIM = _node_struct_dim      # EquivGNNEncoder에서 사용할 scalar 채널 수

# ==============================
# Graph-level 구조 descriptor 설정
#  - train_encoder_qm9.py 의 build_structural_descriptor 와 일관되게 사용
#  - RDF + ADF + Shape 구성
# ==============================
# RDF (Radial Distribution Function; pairwise distance histogram)
DESC_USE_RDF = True
DESC_NUM_BINS_R = 64        # [0, DESC_R_MAX] 구간을 등분
DESC_R_MAX = 5.0            # RDF 상한 거리 (QM9 분자 scale 기준)

# ADF (Angular Distribution Function; bond angle histogram)
DESC_USE_ADF = True
DESC_NUM_BINS_ANGLE = 32    # [0, π] 구간

# Global Shape descriptor (inertia eigenvalues + radius of gyration 등)
DESC_USE_SHAPE = True
DESC_SHAPE_DIM = 4          # 예: 3개 eigenvalues + 1개 Rg

# 각 파트별 descriptor dimension 계산
DESC_DIM_R = 2 * DESC_NUM_BINS_R if DESC_USE_RDF else 0          # hist + hist^2
DESC_DIM_A = 2 * DESC_NUM_BINS_ANGLE if DESC_USE_ADF else 0      # hist + hist^2
DESC_DIM_SHAPE = DESC_SHAPE_DIM if DESC_USE_SHAPE else 0

# 최종 graph-level descriptor dimension
DESC_DIM = DESC_DIM_R + DESC_DIM_A + DESC_DIM_SHAPE

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
#   - "none"          : 변환 없음
#   - "signed_log1p"  : sign(y) * log(1 + |y|)
Y_TRANSFORM = "signed_log1p"

# Y_SCALING:
#   - "none"      : 스케일링 없음
#   - "standard"  : (y - mean) / std
#   - "robust"    : (y - median) / IQR
Y_SCALING = "standard"

# Target 분포 로그 기록 시 히스토그램도 PNG로 저장할지 여부
Y_HIST_PLOT = False
Y_HIST_BINS = 50

# ==============================
# Latent Z 스케일링 옵션
# ==============================
# Z_STANDARDIZE:
#   - False : z 그대로 사용
#   - True  : train 기준 feature-wise standardization
Z_STANDARDIZE = True

# ==============================
# Latent density 기반 샘플링 설정
#  - sampling_mlp_experiment.py 에서 사용
# ==============================
DENSITY_KNN_K = 10      # k-NN 이웃 수
DENSITY_ALPHA = 1.0     # sparsity 강조 정도 (inverse-density^alpha)

# ==============================
# DataLoader / Logging 옵션
# ==============================
NUM_WORKERS = 4          # DataLoader num_workers
PIN_MEMORY = True        # GPU 사용 시 DataLoader 옵션

# 기본 로그 레벨 ("DEBUG", "INFO", "WARNING" ...)
LOG_LEVEL = "INFO"

# ==============================
# Edge feature 관련 (Phase 2에서 사용 예정)
# ==============================
EDGE_USE_RBF = True      # edge distance RBF embedding 사용할지 여부 (현재는 설계만)
EDGE_RBF_DIM = 32
EDGE_R_MAX = 5.0         # encoder radius와 동일하게 두는 것을 권장
