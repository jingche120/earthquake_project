import h5py
import pandas as pd
import numpy as np

# ==========================================
# 模組 1: P波到達時間查詢器
# ==========================================
class PWaveArrivalLocator:
    def __init__(self, csv_path):
        """初始化：載入 CSV 並確認欄位格式"""
        print(f"[*] 正在初始化並載入 CSV 檔案: {csv_path} ...")
        self.df = pd.read_csv(csv_path, low_memory=False)
        
        required_cols = ['source_event_id', 'station_code', 'trace_p_arrival_sample']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"[!] CSV 檔案中缺少必要的欄位: {col}")
                
        # 強制轉為字串以利後續使用 startswith 比對
        self.df['source_event_id'] = self.df['source_event_id'].astype(str)
        print("[+] CSV 載入完成，查詢器準備就緒！\n")

    def get_arrival_time(self, hdf5_key):
        """根據 HDF5 檔名，回傳對應的 P波採樣點"""
        try:
            # HDF5 key 格式: 2012_EVENTID_STATION_...
            # 先去掉 2012_ 前綴，再解析
            if hdf5_key.startswith('2012_'):
                stripped_key = hdf5_key[5:]  # 去掉 '2012_' 前綴
            else:
                stripped_key = hdf5_key
            
            parts = stripped_key.split('_')
            if len(parts) < 2:
                return None
            event_id = parts[0]      
            station_code = parts[1]  
        except Exception:
            return None

        # 雙重條件比對
        condition = (self.df['source_event_id'].str.startswith(event_id)) & \
                    (self.df['station_code'] == station_code)
                    
        matched_row = self.df[condition]
        
        if not matched_row.empty:
            arrival_sample = matched_row['trace_p_arrival_sample'].iloc[0]
            if pd.isna(arrival_sample):
                return None
            return int(arrival_sample)
        else:
            return None

# ==========================================
# 模組 2: HDF5 波形擷取與裁切主程式
# ==========================================
def create_p_wave_dataset(hdf5_path, locator, limit=10):
    """
    結合查詢器，從 HDF5 萃取 P 波前後 500 筆資料並打包成字典
    """
    final_dataset = {}
    success_count = 0
    skip_count = 0
    
    print(f"[*] 準備開啟 HDF5 檔案: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        hdf5_keys = list(f.keys())
        print(f"[+] HDF5 內共有 {len(hdf5_keys)} 筆資料，開始進行裁切 (目標: {limit} 筆)...")
        print("-" * 40)
        
        for key in hdf5_keys:
            # 達到測試上限即停止
            if success_count >= limit:
                print(f"[*] 已達到測試上限 ({limit} 筆)，停止擷取。")
                break

            # 1. 呼叫查詢器取得 P波位置
            p_arrival = locator.get_arrival_time(key)
            
            # 如果成功找到 P波位置
            if p_arrival is not None:
                try:
                    # 2. 抓取原始波形資料
                    waveform = f[key][:] 
                except Exception as e:
                    print(f"  [!] 無法讀取 {key} 的波形: {e}")
                    skip_count += 1
                    continue
                
                # 3. 計算裁切範圍
                start_idx = p_arrival - 500
                end_idx = p_arrival + 500
                
                # 4. 邊界保護與裁切
                if start_idx >= 0 and end_idx <= len(waveform):
                    sliced_wave = waveform[start_idx:end_idx]
                    
                    # 5. 存入字典
                    final_dataset[key] = {
                        'trace_p_arrival_sample': p_arrival,
                        '波': sliced_wave.tolist()
                    }
                    print(f"✅ 成功擷取: {key} (P波位置: {p_arrival})")
                    success_count += 1
                else:
                    skip_count += 1
            else:
                skip_count += 1
                
    print("-" * 40)
    print(f"📊 總結：成功擷取 {success_count} 筆，略過 {skip_count} 筆。")
    print("-" * 40)
    
    return final_dataset

# ==========================================
# 執行區塊
# ==========================================
if __name__ == "__main__":
    # --- 路徑設定 ---
    hdf5_file = r'Y:\CWA_processed_data\all.hdf5'
    csv_file = r'Y:\CWA_processed_data\all_metadata.csv'
    
    try:
        # 步驟 A: 初始化查詢器
        my_locator = PWaveArrivalLocator(csv_file)
        
        # 步驟 B: 執行資料擷取 (預設測試 10 筆)
        my_new_data = create_p_wave_dataset(hdf5_file, my_locator, limit=10)
        
        # 步驟 C: 最終資料結構檢查
        if my_new_data:
            print("\n[檢查] 最終字典的第一筆資料結構：")
            first_key = list(my_new_data.keys())[0]
            print(f"Key: {first_key}")
            print(f" - trace_p_arrival_sample: {my_new_data[first_key]['trace_p_arrival_sample']}")
            print(f" - 波形陣列長度: {len(my_new_data[first_key]['波'])} 點")
            
    except Exception as e:
        print(f"\n[!] 程式執行發生錯誤: {e}")