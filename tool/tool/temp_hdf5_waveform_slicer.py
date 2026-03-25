import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import numpy as np

class WaveformExtractor:
    """
    用於自 HDF5 地震波形資料庫中，高效率提取並裁切特定波形序列的工具類別。

    此類別為解決透過 SMB/NFS 網路掛載硬碟讀取 HDF5 檔案時，因頻繁開啟檔案
    及搜尋索引而導致的極度效能低落問題。

    運作機制：
    在實例化 (Initialization) 階段，會開啟目標 HDF5 檔案一次，遍歷其內部
    所有 Key，並在記憶體中建立 O(1) 的 Hash Map 快速索引。
    它會自動過濾 HDF5 Key 的前綴（例如 '2012_'、'2021_' 等年份標籤），
    僅保留乾淨的 `trace_name` 作為比對鍵值，以完美對接 Metadata CSV 的資料。

    Attributes:
        hdf5_path (str): 來源 HDF5 檔案的絕對或相對路徑。
        fast_lookup (dict): 儲存 `trace_name` 對應至真實 `HDF5 Key` 的快速索引字典。
    """

    def __init__(self, hdf5_path):
        """
        初始化 WaveformExtractor 實例，並建立 HDF5 的快速搜尋索引。

        Args:
            hdf5_path (str): HDF5 檔案的路徑 (例如 '/Volumes/mcnlab2/.../all.hdf5')

        Raises:
            FileNotFoundError: 如果指定的 HDF5 檔案路徑不存在。
        """
        self.hdf5_path = hdf5_path
        self.fast_lookup = {}
        
        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(f"[!] 找不到 HDF5 檔案，請檢查掛載路徑: {self.hdf5_path}")
            
        print(f"[*] 正在初始化波形萃取器，建立快速索引 (只需幾秒鐘)...")
        with h5py.File(self.hdf5_path, 'r') as f:
            for key in f.keys():
                # 去除 HDF5 檔案中特有的年份前綴，提取乾淨的 trace_name
                # 例如: '2012_1201010230_ALS_HL_SMT_10' -> '1201010230_ALS_HL_SMT_10'
                trace_name = key.split('_', 1)[-1]
                self.fast_lookup[trace_name] = key
                    
        print(f"[+] 萃取器準備就緒！共快取了 {len(self.fast_lookup)} 筆波形位置。")

    def extract(self, trace_name, slice_range, output_dir="extracted_waveforms"):
        """
        根據給定的波形名稱與時間範圍，自 HDF5 中提取資料並裁切成固定長度的 numpy 陣列。

        本方法具備「邊界保護」與「Zero-padding」機制。若要求的裁切範圍超出原始波形
        的真實長度（例如 P波來得太早或檔案結束得太快），會自動於空缺處補零，
        確保所有輸出的陣列形狀皆完美一致。

        Args:
            trace_name (str): 目標波形的乾淨名稱 (不含年份前綴)，對應 CSV 的 trace_name。
                              (例如: '1201010230_ALS_HL_SMT_10')
            slice_range (tuple of int): 預期裁切的範圍，格式為 (start_index, end_index)。
                                        (例如: 要求截取 2000 個採樣點可輸入 (1000, 3000))
            output_dir (str, optional): 儲存輸出 `.npy` 檔案的資料夾路徑。
                                        預設為當前目錄下的 "extracted_waveforms"。

        Returns:
            str or None: 如果成功裁切並儲存，回傳該 `.npy` 檔案的完整路徑。
                         如果找不到該波形或讀取失敗，則印出錯誤訊息並回傳 None。
        """
        trace_name = str(trace_name).strip()
        start_idx, end_idx = slice_range
        expected_length = end_idx - start_idx
        
        target_key = self.fast_lookup.get(trace_name)
        
        if target_key is None:
            print(f"[!] 找不到資料: {trace_name}")
            return None

        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                waveform = f[target_key][:] 
        except Exception as e:
            print(f"[!] 讀取波形 {trace_name} 發生錯誤: {e}")
            return None

        # 針對 (channels, time_steps) 維度的裁切與補零邏輯
        channels = waveform.shape[0]   
        time_steps = waveform.shape[1] 
        
        sliced_wave = np.zeros((channels, expected_length))
        
        actual_start = max(0, start_idx)
        actual_end = min(time_steps, end_idx)
        
        insert_start = 0 if start_idx >= 0 else abs(start_idx)
        insert_end = insert_start + (actual_end - actual_start)
        
        if insert_start < insert_end:
            sliced_wave[:, insert_start:insert_end] = waveform[:, actual_start:actual_end]
            
        os.makedirs(output_dir, exist_ok=True)
        out_filename = os.path.join(output_dir, f"{trace_name}.npy") 
        np.save(out_filename, sliced_wave)
        
        return out_filename