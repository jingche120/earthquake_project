import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PGADataset(Dataset):
    def __init__(self, npz_file_path):
        """
        讀取 .npz 檔案並將其轉換為 PyTorch Tensors
        """
        # 載入資料
        data = np.load(npz_file_path)
        
        # 取出波形資料 X，並確保型態為 float32
        # 形狀: (100, 3, 400)
        self.waveforms = torch.tensor(data['waveforms'], dtype=torch.float32)
        
        # 取出 PGA 標籤 y，並確保型態為 float32
        # 【陷阱修正】原始形狀 (100,) -> 必須轉換為 (100, 1) 以對齊模型輸出
        self.pgas = torch.tensor(data['pgas'], dtype=torch.float32).unsqueeze(1)
        
        # (選擇性) 如果你需要 key 或 p_arrival 可以在這裡存起來
        self.keys = data['keys']

    def __len__(self):
        # 回傳資料總筆數
        return len(self.waveforms)

    def __getitem__(self, idx):
        # 根據 index 回傳一組 (X, y)
        return self.waveforms[idx], self.pgas[idx]