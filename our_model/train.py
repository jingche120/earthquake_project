"""
動態現地型地震預警系統 - 訓練腳本
train.py

專案結構:
  C:\\Users\\user\\Desktop\\L_Earthquake_Model\\earthquake_project\\
  ├── our_model\\
  │   ├── eew_anp_model.py       ← model 定義
  │   ├── train.py               ← 本檔案
  │   └── split_indices\\
  │       ├── train_indices.npy
  │       ├── val_indices.npy
  │       └── test_indices.npy
  ├── source\\
  │   ├── all_stastion_info.json  ← 166 站座標 {"EGFH": [lat, lon], ...}
  │   └── hualien_stations_info.json ← 花蓮 21 站名 ["EGFH", ...]
  └── ...

  D:\\filtered_by_Hualien_events_extracted_waveforms\\
  ├── 1201040334\\               ← 事件資料夾
  │   ├── 1201040334_EGFH_HL_SMT_10.npy  ← (3, 3000)
  │   └── ...
  ├── pga_labels.npy             ← (1683, 166)
  └── ...

用法:
  cd our_model
  python train.py                        ← 全部用預設值
  python train.py --batch_size 4 --lr 5e-4  ← 覆寫部分參數
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

# ---- 路徑設定 ----
# 本檔案位於: earthquake_project/our_model/train.py
THIS_DIR = Path(__file__).resolve().parent             # our_model/
PROJECT_DIR = THIS_DIR.parent                          # earthquake_project/

# 把 our_model/ 加入 sys.path，這樣可以 import eew_anp_model
sys.path.insert(0, str(THIS_DIR))
from eew_anp_model import EEWANP, ModelConfig, AsymmetricMSELoss


# =============================================================================
# 1. 預設路徑 (根據實際專案結構)
# =============================================================================
DEFAULTS = {
    "data_dir":             r"D:\filtered_by_Hualien_events_extracted_waveforms",
    "all_station_json":     str(PROJECT_DIR / "source" / "all_stastion_info.json"),
    "hualien_station_json": str(PROJECT_DIR / "source" / "hualien_stations_info.json"),
    "split_dir":            str(THIS_DIR / "split_indices"),
    "output_dir":           str(THIS_DIR / "checkpoints"),
}


# =============================================================================
# 2. Training Config
# =============================================================================
class TrainConfig:
    """訓練超參數"""
    epochs: int = 200
    batch_size: int = 8         # RTX 5090 32GB
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20          # Early stopping
    min_delta: float = 1e-4

    # AMP (Mixed Precision)
    use_amp: bool = True

    # 儲存
    save_every: int = 10

    # Scheduler
    warmup_epochs: int = 5


# =============================================================================
# 3. Station Manager
# =============================================================================
class StationManager:
    """
    管理測站資訊:
      - 全部 166 站: 名稱、座標、index
      - 花蓮 21 站: 名稱及其在 166 站中的 index
    """
    def __init__(self, all_station_json: str, hualien_station_json: str):
        # 全部 166 站 {"EGFH": [lat, lon], ...}
        with open(all_station_json, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        self.all_names = list(all_data.keys())
        self.n_all = len(self.all_names)
        self.name_to_idx = {name: i for i, name in enumerate(self.all_names)}

        # 座標 (166, 2) — JSON 存 [lat, lon]，轉成 [lon, lat]
        self.all_coords = np.array(
            [[v[1], v[0]] for v in all_data.values()],
            dtype=np.float32
        )

        # 花蓮 21 站 ["EGFH", "ESL", ...]
        with open(hualien_station_json, "r", encoding="utf-8") as f:
            hualien_names = json.load(f)

        self.src_names = hualien_names
        self.n_src = len(hualien_names)

        # 花蓮站在 166 站中的 index
        self.src_indices = np.array(
            [self.name_to_idx[name] for name in hualien_names],
            dtype=np.int64
        )
        self.src_coords = self.all_coords[self.src_indices]  # (21, 2)

        print(f"  全部測站: {self.n_all} 站")
        print(f"  花蓮來源站: {self.n_src} 站")
        print(f"  來源站 indices: {self.src_indices.tolist()}")

    def parse_station_from_filename(self, filename: str) -> str:
        """1201040334_EGFH_HL_SMT_10.npy → EGFH"""
        parts = filename.replace(".npy", "").split("_")
        return parts[1]


# =============================================================================
# 4. Data Augmentation Config
# =============================================================================
class AugConfig:
    """資料擴增設定"""
    enable: bool = True           # 是否啟用擴增（僅 train 用）
    top_k: int = 100              # 挑選 PGA 最大的前 K 個事件做 oversampling
    oversample_times: int = 1     # 高 PGA 事件額外重複幾次（1 = 加倍）
    flip_prob: float = 0.5        # 波形反轉極性的機率
    scale_prob: float = 0.5       # 隨機縮放的機率
    scale_range: tuple = (0.8, 1.2)  # 縮放因子範圍


# =============================================================================
# 5. Dataset
# =============================================================================
class EEWDataset(Dataset):
    """
    每個 __getitem__ 回傳一個地震事件:
      - src_waveforms: (N_SRC, 3, 3000)  花蓮 21 站波形 (缺失站為 0)
      - src_coords:    (N_SRC, 2)        花蓮 21 站座標
      - tgt_coords:    (166, 2)          全部 166 站座標
      - pga:           (166,)            全部 166 站 PGA (無資料為 0)
      - src_mask:      (N_SRC,)          bool, 哪些 Source 站本事件有波形

    擴增策略 (僅 train, 參考 Saad et al. 2024):
      1. Oversampling: PGA 最大的 top_k 事件 index 重複加入
      2. 反轉極性: 波形 × -1 (PGA 不變，因為 PGA = max|amplitude|)
      3. 隨機縮放: 波形 × scale_factor, PGA × scale_factor
    """
    def __init__(self, data_dir, event_folders, indices, pga_labels, station_mgr,
                 aug_cfg: AugConfig = None):
        self.data_dir = Path(data_dir)
        self.event_folders = event_folders
        self.pga_labels = pga_labels
        self.mgr = station_mgr
        self.aug_cfg = aug_cfg  # None = 不擴增 (val/test)

        self.src_coords = torch.tensor(station_mgr.src_coords, dtype=torch.float32)
        self.tgt_coords = torch.tensor(station_mgr.all_coords, dtype=torch.float32)

        # --- 建立 index 列表（含 oversampling）---
        self.indices = list(indices)
        self.augmented_flags = [False] * len(indices)  # 標記是否為擴增樣本

        if aug_cfg is not None and aug_cfg.enable:
            # 計算每個事件的最大 PGA
            event_max_pga = np.array([pga_labels[i].max() for i in indices])
            # 找出 top_k 的 index（在 indices 陣列內的位置）
            top_k = min(aug_cfg.top_k, len(indices))
            top_positions = np.argsort(event_max_pga)[-top_k:]

            # 將高 PGA 事件 index 重複加入
            n_added = 0
            for _ in range(aug_cfg.oversample_times):
                for pos in top_positions:
                    self.indices.append(indices[pos])
                    self.augmented_flags.append(True)
                    n_added += 1

            pga_threshold = event_max_pga[top_positions].min()
            print(f"  擴增: top {top_k} 事件 × {aug_cfg.oversample_times} 次 "
                  f"(PGA ≥ {pga_threshold:.2f}), 新增 {n_added} 筆, "
                  f"總計 {len(self.indices)} 筆")

    def __len__(self):
        return len(self.indices)

    def _apply_augmentation(self, src_waveforms, pga):
        """
        對波形和 PGA 施加隨機擴增:
          1. 反轉極性: waveform × -1 (PGA 不變)
          2. 隨機縮放: waveform × s, pga × s
        """
        cfg = self.aug_cfg

        # 反轉極性 (50% 機率)
        if torch.rand(1).item() < cfg.flip_prob:
            src_waveforms = -src_waveforms

        # 隨機縮放
        if torch.rand(1).item() < cfg.scale_prob:
            scale = torch.empty(1).uniform_(cfg.scale_range[0], cfg.scale_range[1]).item()
            src_waveforms = src_waveforms * scale
            # PGA 正比於振幅，縮放後 PGA 也要跟著縮放
            pga = pga * scale

        return src_waveforms, pga

    def __getitem__(self, idx):
        event_idx = self.indices[idx]
        is_augmented = self.augmented_flags[idx]
        event_name = self.event_folders[event_idx]
        event_dir = self.data_dir / event_name

        # --- 讀取花蓮 21 站波形 ---
        n_src = self.mgr.n_src
        src_waveforms = torch.zeros(n_src, 3, 3000, dtype=torch.float32)
        src_mask = torch.zeros(n_src, dtype=torch.bool)

        # 建立 {station_name: path} 對照
        file_map = {}
        for f in event_dir.glob("*.npy"):
            station_code = self.mgr.parse_station_from_filename(f.name)
            file_map[station_code] = f

        # 填入花蓮站波形
        for src_i, station_name in enumerate(self.mgr.src_names):
            if station_name in file_map:
                waveform = np.load(file_map[station_name])  # (3, 3000)
                src_waveforms[src_i] = torch.from_numpy(waveform.astype(np.float32))
                src_mask[src_i] = True

        # --- PGA 標籤 ---
        pga = torch.tensor(self.pga_labels[event_idx], dtype=torch.float32)

        # --- 擴增（僅對標記為擴增的樣本 or 所有 train 樣本都有機率觸發）---
        if self.aug_cfg is not None and self.aug_cfg.enable:
            src_waveforms, pga = self._apply_augmentation(src_waveforms, pga)

        return src_waveforms, self.src_coords, self.tgt_coords, pga, src_mask


def collate_fn(batch):
    """
    Stack batch:
      y_src:    (B, 21, 3, 3000)
      x_src:    (B, 21, 2)
      x_tgt:    (B, 166, 2)
      pga_real: (B, 166, 1)
      src_mask: (B, 21)
    """
    src_wav, src_crd, tgt_crd, pga, mask = zip(*batch)
    return (
        torch.stack(src_wav),
        torch.stack(src_crd),
        torch.stack(tgt_crd),
        torch.stack(pga).unsqueeze(-1),
        torch.stack(mask),
    )


# =============================================================================
# 5. LR Scheduler
# =============================================================================
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            factor = (self.last_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + np.cos(np.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]


# =============================================================================
# 6. Train / Validate
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for y_src, x_src, x_tgt, pga_real, src_mask in loader:
        y_src = y_src.to(device)
        x_src = x_src.to(device)
        x_tgt = x_tgt.to(device)
        pga_real = pga_real.to(device)

        optimizer.zero_grad()

        with autocast("cuda", enabled=use_amp):
            pga_pred = model(y_src, x_src, x_tgt)  # (B, 166, 1)

            # 只對有 PGA 標籤的位置計算 loss (pga > 0 = 有資料)
            valid = (pga_real > 0)  # (B, 166, 1) bool
            if valid.any():
                loss = criterion(pga_pred[valid], pga_real[valid])
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    all_preds, all_reals = [], []
    n_batches = 0

    for y_src, x_src, x_tgt, pga_real, src_mask in loader:
        y_src = y_src.to(device)
        x_src = x_src.to(device)
        x_tgt = x_tgt.to(device)
        pga_real = pga_real.to(device)

        with autocast("cuda", enabled=use_amp):
            pga_pred = model(y_src, x_src, x_tgt)
            valid = (pga_real > 0)
            if valid.any():
                loss = criterion(pga_pred[valid], pga_real[valid])
            else:
                loss = torch.tensor(0.0, device=device)

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(pga_pred.cpu())
        all_reals.append(pga_real.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_reals = torch.cat(all_reals, dim=0)
    valid = (all_reals > 0)
    vp = all_preds[valid]
    vr = all_reals[valid]

    mae = (vp - vr).abs().mean().item() if vp.numel() > 0 else 0.0
    mse = ((vp - vr) ** 2).mean().item() if vp.numel() > 0 else 0.0

    return total_loss / max(n_batches, 1), mae, mse


# =============================================================================
# 7. Checkpoint
# =============================================================================
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "val_loss": val_loss,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"], ckpt["val_loss"]


# =============================================================================
# 8. Main
# =============================================================================
def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Config ----
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.lr is not None:
        train_cfg.lr = args.lr
    if args.epochs is not None:
        train_cfg.epochs = args.epochs
    if args.lambd is not None:
        model_cfg.lambd = args.lambd

    # ---- 測站資訊 ----
    print("\n--- Station Info ---")
    station_mgr = StationManager(args.all_station_json, args.hualien_station_json)
    model_cfg.N_SRC = station_mgr.n_src   # 21
    model_cfg.M_TGT = station_mgr.n_all   # 166

    # ---- 事件列表 ----
    print("\n--- Event Data ---")
    data_dir = Path(args.data_dir)
    event_folders = sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and d.name not in ("merged_tensors", "split_indices")
    ])
    n_events = len(event_folders)
    print(f"  事件總數: {n_events}")

    # ---- PGA 標籤 ----
    pga_labels = np.load(data_dir / "pga_labels.npy")  # (1683, 166)
    print(f"  PGA labels: {pga_labels.shape}")
    print(f"  PGA 非零率: {(pga_labels > 0).sum() / pga_labels.size * 100:.1f}%")

    # ---- Split ----
    split_dir = Path(args.split_dir)
    train_idx = np.load(split_dir / "train_indices.npy")
    val_idx   = np.load(split_dir / "val_indices.npy")
    test_idx  = np.load(split_dir / "test_indices.npy")
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # ---- Datasets ----
    print("\n--- Building Datasets ---")
    aug_cfg = AugConfig()
    if args.no_aug:
        aug_cfg.enable = False
    if args.top_k is not None:
        aug_cfg.top_k = args.top_k
    if args.oversample_times is not None:
        aug_cfg.oversample_times = args.oversample_times

    train_ds = EEWDataset(data_dir, event_folders, train_idx, pga_labels, station_mgr, aug_cfg=aug_cfg)
    val_ds   = EEWDataset(data_dir, event_folders, val_idx,   pga_labels, station_mgr, aug_cfg=None)  # val 不擴增

    # ---- 儲存 config ----
    config_record = {
        "model": {k: getattr(model_cfg, k) for k in dir(model_cfg)
                  if not k.startswith("_") and not callable(getattr(model_cfg, k))},
        "train": {k: getattr(train_cfg, k) for k in dir(train_cfg)
                  if not k.startswith("_") and not callable(getattr(train_cfg, k))},
        "data": {
            "n_events": n_events,
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "n_train_after_aug": len(train_ds),
            "src_stations": station_mgr.src_names,
            "augmentation": {
                "enable": aug_cfg.enable,
                "top_k": aug_cfg.top_k,
                "oversample_times": aug_cfg.oversample_times,
                "flip_prob": aug_cfg.flip_prob,
                "scale_range": aug_cfg.scale_range,
            },
        }
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2, ensure_ascii=False, default=str)

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True
    )

    # ---- Model ----
    print("\n--- Model ---")
    model = EEWANP(model_cfg).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = AsymmetricMSELoss(lambd=model_cfg.lambd)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, train_cfg.warmup_epochs, train_cfg.epochs)
    scaler = GradScaler("cuda", enabled=train_cfg.use_amp)

    # ---- Resume ----
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        print(f"\n  Resuming from {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        start_epoch += 1
        print(f"  Epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    # ---- Training ----
    print(f"\n{'='*65}")
    print(f"  Epochs: {train_cfg.epochs} | BS: {train_cfg.batch_size} | LR: {train_cfg.lr}")
    print(f"  AMP: {train_cfg.use_amp} | λ: {model_cfg.lambd}")
    print(f"  Source: {station_mgr.n_src} 花蓮站 → Target: {station_mgr.n_all} 全站")
    print(f"  Train samples: {len(train_ds)} (原 {len(train_idx)} + 擴增 {len(train_ds)-len(train_idx)})")
    print(f"{'='*65}\n")

    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "lr": []}

    for epoch in range(start_epoch, train_cfg.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, train_cfg.use_amp)
        val_loss, val_mae, val_mse = validate(model, val_loader, criterion, device, train_cfg.use_amp)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["lr"].append(lr)

        print(
            f"[{epoch+1:3d}/{train_cfg.epochs}] "
            f"train={train_loss:.4f} val={val_loss:.4f} "
            f"MAE={val_mae:.4f} MSE={val_mse:.4f} "
            f"lr={lr:.2e} {elapsed:.1f}s"
        )

        if val_loss < best_val_loss - train_cfg.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss, output_dir / "best_model.pt")
            print(f"  ✓ best saved (val={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % train_cfg.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss, output_dir / f"ckpt_{epoch+1:03d}.pt")

    np.savez(output_dir / "history.npz", **{k: np.array(v) for k, v in history.items()})
    print(f"\nDone. Best val_loss: {best_val_loss:.6f}")

    # ---- Test ----
    print("\n--- Test ---")
    test_ds = EEWDataset(data_dir, event_folders, test_idx, pga_labels, station_mgr, aug_cfg=None)  # test 不擴增
    test_loader = DataLoader(
        test_ds, batch_size=train_cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True
    )
    load_checkpoint(output_dir / "best_model.pt", model)
    model.to(device)
    test_loss, test_mae, test_mse = validate(model, test_loader, criterion, device, train_cfg.use_amp)
    print(f"  Loss={test_loss:.4f} MAE={test_mae:.4f} MSE={test_mse:.4f}")

    with open(output_dir / "test_results.json", "w") as f:
        json.dump({"test_loss": test_loss, "test_mae": test_mae, "test_mse": test_mse}, f, indent=2)


# =============================================================================
# 9. CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEW-ANP Training")

    # 路徑參數 (都有預設值，直接 python train.py 就能跑)
    parser.add_argument("--data_dir",             type=str, default=DEFAULTS["data_dir"])
    parser.add_argument("--all_station_json",     type=str, default=DEFAULTS["all_station_json"])
    parser.add_argument("--hualien_station_json", type=str, default=DEFAULTS["hualien_station_json"])
    parser.add_argument("--split_dir",            type=str, default=DEFAULTS["split_dir"])
    parser.add_argument("--output_dir",           type=str, default=DEFAULTS["output_dir"])

    # 訓練參數 (可選覆寫)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--lambd",      type=float, default=None)
    parser.add_argument("--resume",     type=str,   default=None)

    # 擴增參數
    parser.add_argument("--no_aug",          action="store_true", help="關閉資料擴增")
    parser.add_argument("--top_k",           type=int,   default=None, help="Oversample 前 K 大 PGA 事件 (預設 100)")
    parser.add_argument("--oversample_times", type=int,  default=None, help="高 PGA 事件額外重複次數 (預設 1)")

    args = parser.parse_args()
    main(args)
