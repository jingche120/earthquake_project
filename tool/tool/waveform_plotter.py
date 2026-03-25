import os
import numpy as np
import matplotlib.pyplot as plt

class WaveformPlotter:
    """
    用於讀取地震波形 NumPy 陣列 (.npy) 並繪製三軸 (Z, N, E) 波形圖的視覺化工具。

    此類別負責處理波形資料的視覺化，包含時間軸換算、多通道繪圖、
    以及關鍵相位 (如 P 波到達時間) 的標示，並將結果輸出為圖片檔。

    Attributes:
        output_dir (str): 儲存波形圖片的目標資料夾路徑。
        fs (int or float): 儀器的取樣率 (Sampling frequency, Hz)，預設為 100 Hz。
        p_arrival_sec (float): P 波在圖形時間軸上的標示位置 (秒)，預設為 5.0 秒。
    """

    def __init__(self, output_dir="waveform_plots", fs=100, p_arrival_sec=5.0):
        """
        初始化波形繪圖器。

        Args:
            output_dir (str, optional): 圖片輸出的資料夾。預設為 "waveform_plots"。
            fs (int or float, optional): 地震儀取樣率。預設為 100。
            p_arrival_sec (float, optional): P 波到達的相對時間 (秒)。
                                             因截取 P 波前 500 點，100Hz 下即為第 5 秒。預設為 5.0。
        """
        self.output_dir = output_dir
        self.fs = fs
        self.p_arrival_sec = p_arrival_sec
        
        # 確保輸出圖片的資料夾存在
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[*] 波形繪圖器初始化完成，圖片將輸出至: {self.output_dir}")

    def plot_from_npy(self, npy_path, trace_name=None):
        """
        讀取指定的 .npy 檔案，繪製波形圖並儲存為 .png 圖片。

        本方法預期讀取的 NumPy 陣列形狀為 (channels, time_steps)，
        例如 (3, 2000)，分別代表 Z, N, E 三個分量。

        Args:
            npy_path (str): 切割好的 .npy 波形檔案路徑。
            trace_name (str, optional): 顯示在圖表標題與檔名上的波形名稱。
                                        若未提供，將自動從 npy_path 的檔名中提取。

        Returns:
            str or None: 如果成功繪製並儲存，回傳該 `.png` 檔案的完整路徑。
                         如果讀取失敗或維度錯誤，則印出錯誤訊息並回傳 None。
        """
        if not os.path.exists(npy_path):
            print(f"[!] 找不到檔案: {npy_path}")
            return None

        # 如果沒有給定 trace_name，就自動用檔名當作 trace_name (去掉 .npy 後綴)
        if trace_name is None:
            trace_name = os.path.splitext(os.path.basename(npy_path))[0]

        try:
            # 讀取 numpy 陣列
            waveform = np.load(npy_path)
        except Exception as e:
            print(f"[!] 無法讀取 {npy_path}: {e}")
            return None

        # 檢查維度是否符合預期 (至少要有 3 個通道)
        if waveform.ndim != 2 or waveform.shape[0] < 3:
            print(f"[!] 波形維度錯誤: {waveform.shape}。預期應類似 (3, 2000)。")
            return None

        # 計算時間軸
        samples = waveform.shape[1]
        t = np.arange(samples) / self.fs
        
        # 建立 3x1 的畫布
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        channel_labels = ["Z", "N", "E"]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

        # 逐一繪製三個通道
        for i in range(3):
            axs[i].plot(t, waveform[i], color=colors[i], linewidth=1)
            axs[i].set_title(f"Channel: {channel_labels[i]}")
            axs[i].set_ylabel("Amplitude")
            axs[i].grid(True, linestyle='--', alpha=0.7)
            
            # 標示 P 波位置
            axs[i].axvline(x=self.p_arrival_sec, color='r', linestyle='--', label='P-arrival')
            
            # 只在第一張圖顯示圖例，避免畫面雜亂
            if i == 0:
                axs[i].legend(loc='upper right')

        # 設定底部 X 軸標籤與總標題
        total_time_sec = samples / self.fs
        axs[-1].set_xlabel(f"Time (s) [0-{total_time_sec:.1f}s]")
        plt.suptitle(f"Trace: {trace_name} ({total_time_sec:.1f}s window)", fontsize=14)
        plt.tight_layout()
        
        # 儲存圖片並關閉畫布 (非常重要，否則迴圈畫圖會塞爆記憶體)
        png_path = os.path.join(self.output_dir, f"{trace_name}_waveform.png")
        plt.savefig(png_path, dpi=150) # 加入 dpi=150 讓圖片更清晰
        plt.close(fig) 
        
        # print(f"🖼️ 波形圖片已儲存: {png_path}")
        return png_path