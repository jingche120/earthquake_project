"""
test.py — 載入最佳權重，對測試事件做推論，輸出比較結果與視覺化圖表。
功能：
  1. 從 HDF5 中隨機抽取 N 筆事件做推論
  2. 輸出每筆事件的「預測 PGA」vs「實際 PGA」比較
  3. 散點圖（全部測站）
  4. 單一事件的地圖視覺化（在測站座標上畫 PGA 分布）
  5. 計算整體指標（MAE, RMSE, R²）
"""
"""
test.py — 載入 train.py 切出的測試集 index，做推論並輸出比較結果。
"""

import os
import json
import glob
import csv
import yaml
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from model import GATModel

# ============ 讀取設定檔 ============
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

h5_path         = cfg["data"]["h5_path"]
station_json    = cfg["data"]["station_json"]
adj_path        = cfg["data"]["adj_path"]
split_index_dir = cfg["data"]["split_index_dir"]
num_stations    = cfg["model"]["num_stations"]
model_dir       = cfg["model"]["model_dir"]
output_dir      = cfg["test"]["output_dir"]
# ====================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  使用裝置：{device}")


def find_best_checkpoint(model_dir):
    """找到資料夾中 epoch 數字最大的 .pth 檔（即最佳權重）。"""
    pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"在 {model_dir} 中找不到任何 .pth 檔案")
    pth_files.sort(key=lambda x: int(os.path.basename(x).replace(".pth", "")))
    return pth_files[-1]


def compute_metrics(pred, true, mask):
    pred_valid = pred[mask]
    true_valid = true[mask]
    mae = np.mean(np.abs(pred_valid - true_valid))
    rmse = np.sqrt(np.mean((pred_valid - true_valid) ** 2))
    ss_res = np.sum((true_valid - pred_valid) ** 2)
    ss_tot = np.sum((true_valid - np.mean(true_valid)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return mae, rmse, r2


def main():
    os.makedirs(output_dir, exist_ok=True)

    # 1. 讀取測站資訊
    with open(station_json, 'r', encoding='utf-8') as f:
        station_data = json.load(f)
    station_names = list(station_data.keys())
    coords = np.array(list(station_data.values()))
    loc = torch.tensor(coords, dtype=torch.float32).to(device)
    adj = torch.tensor(np.load(adj_path), dtype=torch.float32).to(device)

    # 2. 讀取 train.py 儲存的測試集 index
    test_index_path = os.path.join(split_index_dir, "test_indices.npy")
    if not os.path.exists(test_index_path):
        raise FileNotFoundError(
            f"找不到 {test_index_path}\n"
            f"請先執行 train.py 來切割資料集並儲存 index。"
        )
    test_indices = np.load(test_index_path)
    test_indices.sort()
    num_test_events = len(test_indices)
    print(f"📋 載入測試集 index：{num_test_events} 筆（來源：{test_index_path}）")

    # 3. 載入模型
    best_ckpt = find_best_checkpoint(model_dir)
    print(f"📦 載入權重：{best_ckpt}")

    model = GATModel(num_stations=num_stations).to(device)
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    model.eval()

    # 4. 推論
    with h5py.File(h5_path, 'r') as hf:
        total_events = len(hf['pga'])
        print(f"📊 資料集共 {total_events} 筆事件，測試 {num_test_events} 筆")

        all_preds = []
        all_trues = []
        all_masks = []

        print("\n" + "=" * 70)
        print(f"{'事件':>6} | {'有效測站':>8} | {'MAE':>10} | {'RMSE':>10} | {'R²':>10}")
        print("=" * 70)

        for i, idx in enumerate(test_indices):
            wave = hf['wave'][idx]
            pga_true = hf['pga'][idx].astype(np.float32)

            wave_t = torch.tensor(wave, dtype=torch.float32).unsqueeze(0).to(device)
            batch_loc = loc.unsqueeze(0)
            batch_adj = adj.unsqueeze(0)

            with torch.no_grad():
                pga_pred = model(wave_t, batch_loc, batch_adj)

            pga_pred = pga_pred.squeeze(0).cpu().numpy()
            mask = pga_true != 0
            valid_count = mask.sum()

            if valid_count > 0:
                mae, rmse, r2 = compute_metrics(pga_pred, pga_true, mask)
                print(f"  {idx:>5} | {valid_count:>8} | {mae:>10.4f} | {rmse:>10.4f} | {r2:>10.4f}")
            else:
                print(f"  {idx:>5} | {valid_count:>8} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10}")

            all_preds.append(pga_pred)
            all_trues.append(pga_true)
            all_masks.append(mask)

    # 5. 整體指標
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    all_masks = np.array(all_masks)
    overall_mask = all_masks.flatten()

    overall_mae, overall_rmse, overall_r2 = compute_metrics(
        all_preds.flatten(), all_trues.flatten(), overall_mask
    )

    print("=" * 70)
    print(f"📊 整體指標（{overall_mask.sum()} 個有效測站點）")
    print(f"   MAE  = {overall_mae:.4f}")
    print(f"   RMSE = {overall_rmse:.4f}")
    print(f"   R²   = {overall_r2:.4f}")

    # ============ 圖表 1：散點圖 ============
    pred_flat = all_preds.flatten()[overall_mask]
    true_flat = all_trues.flatten()[overall_mask]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(true_flat, pred_flat, alpha=0.3, s=10, c='steelblue')
    max_val = max(true_flat.max(), pred_flat.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlabel('True PGA (gal)', fontsize=13)
    ax.set_ylabel('Predicted PGA (gal)', fontsize=13)
    ax.set_title(f'PGA Prediction vs Ground Truth\nMAE={overall_mae:.4f}  RMSE={overall_rmse:.4f}  R²={overall_r2:.4f}', fontsize=14)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect('equal')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_all.png"), dpi=150)
    plt.close()
    print(f"\n📈 散點圖已儲存：{output_dir}/scatter_all.png")

    # ============ 圖表 2：Log-scale 散點圖 ============
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    log_mask = (pred_flat > 0) & (true_flat > 0)
    pred_log = pred_flat[log_mask]
    true_log = true_flat[log_mask]
    ax.scatter(true_log, pred_log, alpha=0.3, s=10, c='steelblue')
    min_val = min(true_log.min(), pred_log.min()) * 0.5
    max_val = max(true_log.max(), pred_log.max()) * 2
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Perfect prediction')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('True PGA (gal) - log scale', fontsize=13)
    ax.set_ylabel('Predicted PGA (gal) - log scale', fontsize=13)
    ax.set_title('PGA Prediction vs Ground Truth (Log Scale)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_log.png"), dpi=150)
    plt.close()
    print(f"📈 Log 散點圖已儲存：{output_dir}/scatter_log.png")

    # ============ 圖表 3：單一事件地圖比較 ============
    event_idx = 0
    pred_event = all_preds[event_idx]
    true_event = all_trues[event_idx]
    mask_event = all_masks[event_idx]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    valid_true = true_event[mask_event]
    valid_pred = pred_event[mask_event]
    vmin = min(valid_true.min(), valid_pred.min()) if len(valid_true) > 0 else 0
    vmax = max(valid_true.max(), valid_pred.max()) if len(valid_true) > 0 else 1
    lat, lon = coords[:, 0], coords[:, 1]

    for ax_idx, (data, title, cmap) in enumerate([
        (true_event[mask_event], f'True PGA (Event {test_indices[event_idx]})', 'hot_r'),
        (pred_event[mask_event], f'Predicted PGA (Event {test_indices[event_idx]})', 'hot_r'),
    ]):
        ax = axes[ax_idx]
        sc = ax.scatter(lon[mask_event], lat[mask_event], c=data,
                        cmap=cmap, s=60, vmin=vmin, vmax=vmax, edgecolors='black', linewidth=0.5)
        ax.scatter(lon[~mask_event], lat[~mask_event], c='lightgray', s=20, alpha=0.5)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.colorbar(sc, ax=ax, label='PGA (gal)')

    # 誤差圖
    ax = axes[2]
    error = pred_event[mask_event] - true_event[mask_event]
    err_max = max(abs(error.min()), abs(error.max())) if len(error) > 0 else 1
    sc = ax.scatter(lon[mask_event], lat[mask_event], c=error,
                    cmap='RdBu_r', s=60, vmin=-err_max, vmax=err_max, edgecolors='black', linewidth=0.5)
    ax.scatter(lon[~mask_event], lat[~mask_event], c='lightgray', s=20, alpha=0.5)
    ax.set_title(f'Error: Pred - True (Event {test_indices[event_idx]})', fontsize=13)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(sc, ax=ax, label='Error (gal)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"map_event_{test_indices[event_idx]}.png"), dpi=150)
    plt.close()
    print(f"🗺️  地圖比較圖已儲存：{output_dir}/map_event_{test_indices[event_idx]}.png")

    # ============ 圖表 4：各事件 MAE 長條圖 ============
    event_maes = []
    for i in range(num_test_events):
        m = all_masks[i]
        event_maes.append(np.mean(np.abs(all_preds[i][m] - all_trues[i][m])) if m.sum() > 0 else 0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(num_test_events), event_maes, color='steelblue', edgecolor='black')
    ax.set_xticks(range(num_test_events))
    ax.set_xticklabels([str(idx) for idx in test_indices], rotation=45)
    ax.set_xlabel('Event Index', fontsize=12)
    ax.set_ylabel('MAE (gal)', fontsize=12)
    ax.set_title('Per-Event MAE', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_event_mae.png"), dpi=150)
    plt.close()
    print(f"📊 各事件 MAE 長條圖已儲存：{output_dir}/per_event_mae.png")

    # ============ CSV 輸出 ============
    csv_path = os.path.join(output_dir, "predictions.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['event_index', 'station', 'latitude', 'longitude', 'true_pga', 'pred_pga', 'error'])
        for i, idx in enumerate(test_indices):
            for j in range(num_stations):
                if all_masks[i][j]:
                    writer.writerow([
                        idx, station_names[j], coords[j, 0], coords[j, 1],
                        f"{all_trues[i][j]:.6f}", f"{all_preds[i][j]:.6f}",
                        f"{all_preds[i][j] - all_trues[i][j]:.6f}"
                    ])
    print(f"📄 詳細預測結果已儲存：{csv_path}")
    print("\n🎉 測試完成！")


if __name__ == "__main__":
    main()