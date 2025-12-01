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

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader as PyGDataLoader

import configs
import utils
from models import EquivGNNEncoder, EquivDecoder


# ============================================================
# 구조 descriptor: RDF + ADF + Shape 조합
# ============================================================

def build_structural_descriptor(pos: torch.Tensor) -> torch.Tensor:
    """단일 분자에 대한 구조 descriptor 생성.

    구성:
        - RDF (Radial Distribution Function)
        - ADF (Angular Distribution Function) [옵션]
        - Global shape (inertia eigenvalues + Rg) [옵션]

    configs.DESC_USE_* 플래그와 DESC_DIM을 기준으로 길이 검증.
    반환: (DESC_DIM,) tensor (CPU)
    """
    # CPU에서 계산 (캐시용)
    if isinstance(pos, torch.Tensor):
        pos_cpu = pos.detach().cpu()
    else:
        pos_cpu = torch.as_tensor(pos, dtype=torch.float32)

    desc_parts = []

    if configs.DESC_USE_RDF:
        rdf = utils.compute_rdf_descriptor(pos_cpu)  # (2 * DESC_NUM_BINS_R,)
        desc_parts.append(rdf)

    if configs.DESC_USE_ADF:
        adf = utils.compute_adf_descriptor(pos_cpu)  # (2 * DESC_NUM_BINS_ANGLE,)
        desc_parts.append(adf)

    if configs.DESC_USE_SHAPE:
        shape = utils.compute_shape_descriptor(pos_cpu)  # (4,)
        desc_parts.append(shape)

    if len(desc_parts) == 0:
        # 이론상 발생하지 않게 configs에서 보장하는 게 좋지만, 방어코드
        desc = torch.zeros(configs.DESC_DIM, dtype=torch.float32)
    else:
        desc = torch.cat(desc_parts, dim=0).to(torch.float32)

    assert desc.shape[0] == configs.DESC_DIM, (
        f"Structural descriptor dim mismatch: got {desc.shape[0]}, "
        f"expected {configs.DESC_DIM}"
    )
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

        # 구조 latent (node-level 구조 피쳐 포함)
        z_graph = encoder(
            batch.z,
            batch.pos,
            batch.batch,
            node_struct_feats=batch.node_struct_feats,
        )
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

        z_graph = encoder(
            batch.z,
            batch.pos,
            batch.batch,
            node_struct_feats=batch.node_struct_feats,
        )
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
        z_graph = encoder(
            batch.z,
            batch.pos,
            batch.batch,
            node_struct_feats=batch.node_struct_feats,
        )  # (B, latent_dim)
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
    logger.info(
        "Descriptor config: "
        f"DESC_DIM={configs.DESC_DIM}, "
        f"USE_RDF={configs.DESC_USE_RDF}, "
        f"USE_ADF={configs.DESC_USE_ADF}, "
        f"USE_SHAPE={configs.DESC_USE_SHAPE}, "
        f"NODE_STRUCT_DIM={configs.NODE_STRUCT_DIM}, "
        f"NODE_RADIUS={configs.NODE_RADIUS}"
    )

    # ----------------------------------------------
    # QM9 Dataset 로드
    # ----------------------------------------------
    qm9_root = os.path.join(configs.DATA_ROOT, "qm9")
    logger.info(f"Loading QM9 dataset from: {qm9_root}")
    dataset = QM9(root=qm9_root)

    logger.info(f"QM9 total molecules (raw): {len(dataset)}")

    # ----------------------------------------------
    # 구조 descriptor 캐시 사용 (있으면 로드, 없으면 계산 후 저장)
    #  - 캐시에는 graph-level descriptor만 저장
    #  - node-level 구조 피쳐는 매 실행마다 다시 계산 (QM9 크기에서 비용 부담 적음)
    # ----------------------------------------------
    cache_path = getattr(configs, "DESC_CACHE_PATH", None)

    data_list = []
    if cache_path is not None and os.path.exists(cache_path):
        # ===== 1) 캐시 파일이 이미 있는 경우: 로드해서 붙이기 =====
        logger.info(f"Found descriptor cache at: {cache_path}")
        struct_desc_all = torch.load(cache_path, map_location="cpu")
        # struct_desc_all: (N_data, DESC_DIM) 가정

        assert struct_desc_all.shape[0] == len(dataset), \
            "Descriptor cache size does not match QM9 dataset size."

        for i, data in enumerate(dataset):
            desc = struct_desc_all[i]              # (DESC_DIM,)
            data.struct_desc = desc.unsqueeze(0)   # (1, DESC_DIM)

            # node-level 구조 피쳐는 매 실행 시 계산
            cn, mean_d, min_d, max_d = utils.compute_cn_and_dist_stats(
                data.pos, radius=configs.NODE_RADIUS
            )
            node_feats = utils.pack_node_struct_feats(cn, mean_d, min_d, max_d)
            data.node_struct_feats = node_feats  # (N_atoms, NODE_STRUCT_DIM)

            data_list.append(data)

        logger.info("Loaded structural descriptors from cache and recomputed node features.")

    else:
        # ===== 2) 캐시가 없으면: 새로 계산하고 저장 =====
        logger.info("Precomputing structural descriptors and node features for all molecules...")

        desc_list = []

        for data in tqdm(dataset, desc="Build descriptors & node feats", ncols=100):
            # graph-level descriptor
            desc = build_structural_descriptor(data.pos)  # (DESC_DIM,)
            data.struct_desc = desc.unsqueeze(0)          # (1, DESC_DIM)
            desc_list.append(desc.cpu())

            # node-level 구조 피쳐
            cn, mean_d, min_d, max_d = utils.compute_cn_and_dist_stats(
                data.pos, radius=configs.NODE_RADIUS
            )
            node_feats = utils.pack_node_struct_feats(cn, mean_d, min_d, max_d)
            data.node_struct_feats = node_feats  # (N_atoms, NODE_STRUCT_DIM)

            data_list.append(data)

        logger.info("Finished structural descriptor & node feature precomputation.")

        # desc_list를 하나의 텐서로 쌓아서 캐시 파일로 저장
        if cache_path is not None:
            struct_desc_all = torch.stack(desc_list, dim=0)  # (N_data, DESC_DIM)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(struct_desc_all, cache_path)
            logger.info(f"Saved structural descriptor cache to: {cache_path}")

    # descriptor + node feature까지 포함된 유효 데이터 개수 기준
    N_data = len(data_list)
    logger.info(f"Effective dataset size (with descriptors & node features): {N_data}")

    # ----------------------------------------------
    # Train / Val / Test split (data_list 기준)
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

    # 인덱스로 리스트 슬라이싱해서 서브셋 구성
    train_subset = [data_list[i] for i in idx_train]
    val_subset = [data_list[i] for i in idx_val]

    # full_loader는 latent 추출용 (train/val/test 전체)
    full_dataset = data_list
    full_loader = PyGDataLoader(
        full_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=configs.NUM_WORKERS,
        pin_memory=configs.PIN_MEMORY
    )

    train_loader = PyGDataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=configs.NUM_WORKERS,
        pin_memory=configs.PIN_MEMORY
    )

    val_loader = None
    if len(idx_val) > 0:
        val_loader = PyGDataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=configs.NUM_WORKERS,
            pin_memory=configs.PIN_MEMORY
        )

    # ----------------------------------------------
    # 모델/옵티마 초기화
    # ----------------------------------------------
    encoder = EquivGNNEncoder(latent_dim=args.latent_dim).to(device)
    decoder = EquivDecoder(
        latent_dim=args.latent_dim,
        desc_dim=configs.DESC_DIM
    ).to(device)

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
    for data in data_list:
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
