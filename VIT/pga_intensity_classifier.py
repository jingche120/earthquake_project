import numpy as np
import pandas as pd

def analyze_pga_distribution(npz_file_path):
    print(f"[*] 正在載入資料集: {npz_file_path} ...")
    
    try:
        # 1. 讀取 npz 檔案中的 PGA 標籤
        with np.load(npz_file_path, allow_pickle=True) as data:
            pgas = data['pgas']
            total_samples = len(pgas)
            
        print(f"[+] 成功讀取 {total_samples} 筆 PGA 資料！\n")
        
        # 2. 定義震度分級標準 (依據地動加速度 cm/s^2 或 gal)
        # 這裡的 bins 代表每個級別的切分邊界
        bins = [0, 0.8, 2.5, 8.0, 25.0, 80.0, 250.0, 400.0, np.inf]
        
        # 定義對應的級別名稱
        labels = [
            '0 級 (無感, <0.8)', 
            '1 級 (微震, 0.8~2.5)', 
            '2 級 (輕震, 2.5~8.0)', 
            '3 級 (弱震, 8.0~25)', 
            '4 級 (中震, 25~80)', 
            '5 級 (強震, 80~250)', 
            '6 級 (烈震, 250~400)', 
            '7 級 (劇震, >=400)'
        ]
        
        # 3. 使用 pandas 的 cut 函數進行自動分類
        # right=False 代表區間包含左邊界但不包含右邊界 (例如 [0.8, 2.5))
        category_series = pd.cut(pgas, bins=bins, labels=labels, right=False)
        
        # 4. 統計各級別的數量
        summary_df = pd.value_counts(category_series).sort_index().reset_index()
        summary_df.columns = ['震度分級', '資料筆數']
        
        # 新增佔比欄位，方便觀察資料不平衡狀況
        summary_df['佔比 (%)'] = (summary_df['資料筆數'] / total_samples * 100).round(2)
        
        # 5. 印出漂亮的分佈彙總表
        print("=" * 45)
        print(" 資料集 PGA 震度分佈彙總表")
        print("=" * 45)
        # 使用 to_string() 隱藏 index 以保持畫面乾淨
        print(summary_df.to_string(index=False))
        print("=" * 45)
        
        return summary_df
        
    except FileNotFoundError:
        print(f"[!] 錯誤：找不到檔案 '{npz_file_path}'。")
    except Exception as e:
        print(f"[!] 分析時發生錯誤: {e}")

if __name__ == "__main__":
    # 替換成你實際的檔案名稱
    dataset_file = '3000_p_wave_dataset.npz'
    analyze_pga_distribution(dataset_file)