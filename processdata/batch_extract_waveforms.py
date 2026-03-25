"""
批次波形裁切程式
=================
功能：
  1. 讀取 filtered_hualien_taipei_events.csv 中的 source_event_id 與 station_code
  2. 建立 event_earliest_p_arrival_sample 字典（每個事件中最早的 P 波到達時間）
  3. 用該事件最早P波到達時間計算裁切範圍: [p_arrival - 500, p_arrival + 1500]（共 2000 點）
  4. 從 HDF5 檔案中讀取對應波形並裁切，存成 事件id_測站名稱.npy

依賴：tool/waveform_tool.py 中的 WaveformExtractor
"""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import sys
import time
import pandas as pd
import numpy as np

# 將上一層目錄加入搜尋路徑，以便 import tool.waveform_tool
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tool.waveform_tool import WaveformExtractor


def main():
    # ==========================================
    # 路徑設定
    # ==========================================
    csv_path = os.path.join(parent_dir, 'source', 'test.csv')
    hdf5_path = '/Volumes/mcnlab2/CWA_processed_data/all.hdf5'
    output_dir = os.path.join(os.path.dirname(__file__), 'extracted_waveforms')

    # ==========================================
    # 步驟 1：讀取 CSV
    # ==========================================
    print(f"[*] 讀取 CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df['source_event_id'] = df['source_event_id'].astype(str).str.strip()
    df['station_code'] = df['station_code'].astype(str).str.strip()
    print(f"[+] CSV 共 {len(df)} 筆紀錄")

    # ==========================================
    # 步驟 2：建立 event_earliest_p_arrival_sample 字典
    #   key: source_event_id (str)
    #   value: 該事件所有測站中最早的 trace_p_arrival_sample (int)
    # ==========================================
    event_earliest_p_arrival_sample = (
        df.groupby('source_event_id')['trace_p_arrival_sample']
        .min()
        .to_dict()
    )
    print(f"[+] 共建立 {len(event_earliest_p_arrival_sample)} 個事件的最早 P 波到達時間字典")

    # ==========================================
    # 步驟 3：去除重複的 (event_id, station_code) 組合
    #   避免同一事件同一測站有多筆紀錄時重複處理
    # ==========================================
    unique_pairs = df[['source_event_id', 'station_code']].drop_duplicates()
    print(f"[+] 去重後共有 {len(unique_pairs)} 組 (事件, 測站) 需要處理")

    # ==========================================
    # 步驟 4：初始化 WaveformExtractor（建立 HDF5 快速索引）
    # ==========================================
    print(f"\n[*] 初始化 WaveformExtractor...")
    start_time = time.time()
    extractor = WaveformExtractor(hdf5_path)
    init_time = time.time() - start_time
    print(f"[+] 初始化完成，耗時 {init_time:.2f} 秒\n")

    # ==========================================
    # 步驟 5：逐筆提取、裁切、存檔
    # ==========================================
    os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    skip_no_p = 0
    skip_not_found = 0
    total = len(unique_pairs)

    print(f"[*] 開始批次提取波形，輸出目錄: {output_dir}")
    print("=" * 60)

    for i, (_, row) in enumerate(unique_pairs.iterrows()):
        event_id = row['source_event_id']
        station_code = row['station_code']

        # 查詢該事件最早的 P 波到達時間
        p_arrival = event_earliest_p_arrival_sample.get(event_id)
        if p_arrival is None or pd.isna(p_arrival):
            skip_no_p += 1
            continue

        p_arrival = int(p_arrival)

        # 計算裁切範圍: P波前5秒(-500) 到 P波後15秒(+1500)，共2000點
        start_idx = p_arrival - 500
        end_idx = p_arrival + 1500
        slice_range = (start_idx, end_idx)

        # 使用 WaveformExtractor 提取並存檔
        result = extractor.extract(
            event_id=event_id,
            station_code=station_code,
            slice_range=slice_range,
            output_dir=output_dir
        )

        if result is not None:
            success_count += 1
            if success_count % 500 == 0 or success_count <= 5:
                print(f"  [{success_count}/{total}] 已儲存: {os.path.basename(result)} "
                      f"(P波={p_arrival}, 範圍=[{start_idx}, {end_idx}))")
        else:
            skip_not_found += 1

    # ==========================================
    # 步驟 6：結果統計
    # ==========================================
    print("=" * 60)
    print(f"[完成] 批次波形裁切結束！")
    print(f"  - 成功儲存: {success_count} 筆")
    print(f"  - 跳過 (無P波資料): {skip_no_p} 筆")
    print(f"  - 跳過 (HDF5找不到): {skip_not_found} 筆")
    print(f"  - 輸出目錄: {output_dir}")


if __name__ == "__main__":
    main()
