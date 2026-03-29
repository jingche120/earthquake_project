import os
import json
import random
import yaml
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import GATModel

# ============ 讀取設定檔 ============
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
 
# 資料路徑
h5_path = cfg["data"]["h5_path"]
station_json = cfg["data"]["station_json"]
adj_path = cfg["data"]["adj_path"]
split_index_dir = cfg["data"]["split_index_dir"]
num_test_events = cfg["split"]["num_test_events"]
train_ratio     = cfg["split"]["train_ratio"]
val_ratio       = cfg["split"]["val_ratio"]

# 模型設定
num_stations = cfg["model"]["num_stations"]
model_dir = cfg["model"]["model_dir"]
 
# 訓練參數
batch_size = cfg["train"]["batch_size"]
num_epochs = cfg["train"]["num_epochs"]
train_split = cfg["train"]["train_split"]
lr = cfg["train"]["learning_rate"]
weight_decay = cfg["train"]["weight_decay"]
patience = cfg["train"]["patience"]
seed = cfg["train"]["seed"]
date = cfg["train"]["date"]
loss_alpha = cfg["train"]["loss"]["alpha"]
loss_beta = cfg["train"]["loss"]["beta"]
sched_factor = cfg["train"]["scheduler"]["factor"]
sched_patience = cfg["train"]["scheduler"]["patience"]

# ============ 固定隨機種子（從 config.yaml 讀取） ============
random.seed(seed)                          # Python random
np.random.seed(seed)                       # NumPy
torch.manual_seed(seed)                    # PyTorch CPU
torch.cuda.manual_seed(seed)               # PyTorch 單 GPU
torch.cuda.manual_seed_all(seed)           # PyTorch 多 GPU
torch.backends.cudnn.deterministic = True  # cuDNN 確定性模式（結果可重現）
torch.backends.cudnn.benchmark = False     # 禁止自動選演算法（避免不確定性）
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  使用裝置：{device}")
 

# ============ Dataset ============
class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path):
        self.h5_file_path = h5_file_path
        with h5py.File(h5_file_path, 'r') as hf:
            self.length = len(hf['pga'])
            self.labels = hf['pga'][:]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, 'r') as hf:
            wave = hf['wave'][idx]
            label = hf['pga'][idx]
        return wave, label


# ============ 讀取測站座標 ============
with open(station_json, 'r', encoding='utf-8') as f:
    station_data = json.load(f)

loc_list = list(station_data.values())  # [[lat, lon], ...]
loc = torch.tensor(loc_list, dtype=torch.float32).to(device)  # [166, 2]
print(f"📋 測站座標：{loc.shape}")

# ============ 讀取鄰接矩陣 ============
adj = np.load(adj_path)
adj = torch.tensor(adj, dtype=torch.float32).to(device)  # [166, 166]
print(f"📐 鄰接矩陣：{adj.shape}")

# ============ 資料切割：Test → Train / Val ============
dataset = HDF5Dataset(h5_path)
dataset_size = len(dataset)
print(f"📊 資料集大小：{dataset_size}")
 
os.makedirs(split_index_dir, exist_ok=True)
 
# 全部 index 打亂
all_indices = np.arange(dataset_size)
np.random.shuffle(all_indices)
 
# Step 1：先抽出測試集
test_indices = all_indices[:num_test_events]
remaining_indices = all_indices[num_test_events:]
 
# Step 2：剩餘切成 train / val（70 / 30）
split_point = int(len(remaining_indices) * train_ratio)
train_indices = remaining_indices[:split_point]
val_indices = remaining_indices[split_point:]
 
# 儲存 index（確保 test.py 用同一組測試集）
np.save(os.path.join(split_index_dir, "test_indices.npy"), test_indices)
np.save(os.path.join(split_index_dir, "train_indices.npy"), train_indices)
np.save(os.path.join(split_index_dir, "val_indices.npy"), val_indices)
 
print(f"\n📂 資料切割結果（已儲存至 {split_index_dir}/）：")
print(f"   測試集：{len(test_indices)} 筆")
print(f"   訓練集：{len(train_indices)} 筆")
print(f"   驗證集：{len(val_indices)} 筆")
 
# DataLoader
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True)





# ============ 模型 ============
model = GATModel(num_stations=num_stations).to(device)
model_path = f"./parameter/gat_{date}"
os.makedirs(model_path, exist_ok=True)
os.makedirs('./train_fig/loss', exist_ok=True)

# 印出模型參數量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"🧠 模型參數量：{total_params:,}（可訓練：{trainable_params:,}）")


# ============ 損失函數 ============
class CustomLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, outputs, targets):
        loss = self.loss(outputs, targets)
        weight = torch.where((outputs < targets) & (targets >= 0.8), self.alpha, self.beta)
        weighted_loss = loss * weight
        # 遮罩：只計算有資料的測站（target != 0）
        mask = (targets != 0).float()
        masked_loss = weighted_loss * mask
        weighted_loss = masked_loss.sum() / mask.sum().clamp(min=1)
        return weighted_loss


criterion = CustomLoss(alpha=1.2, beta=1).to(device)
param_groups = [
    {'params': [p for n, p in model.named_parameters() if 'bias' not in n], 'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if 'bias' in n], 'weight_decay': 0}
]
optimizer = AdamW(param_groups, lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=6)


# ============ Early Stopping ============
class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0, path='gat.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_model_path = os.path.join(self.path, f"{epoch}.pth")
        torch.save(model.state_dict(), save_model_path)
        self.val_loss_min = val_loss


# ============ 訓練與驗證 ============
def train_epoch(model, train_loader, optimizer, criterion, device, loc, adj):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device).float()    # [B, 166]
        batch_size = inputs.shape[0]

        batch_loc = loc.unsqueeze(0).expand(batch_size, -1, -1)   # [B, 166, 2]
        batch_adj = adj.unsqueeze(0).expand(batch_size, -1, -1)   # [B, 166, 166]

        optimizer.zero_grad()
        outputs = model(inputs, batch_loc, batch_adj)              # [B, 166]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion, device, loc, adj):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device).float()
            batch_size = inputs.shape[0]

            batch_loc = loc.unsqueeze(0).expand(batch_size, -1, -1)
            batch_adj = adj.unsqueeze(0).expand(batch_size, -1, -1)

            outputs = model(inputs, batch_loc, batch_adj)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

    return running_loss / len(val_loader)


# ============ 開始訓練 ============
train_losses = []
val_losses = []
early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_path)

print("\n" + "=" * 50)
print("🚀 開始訓練")
print("=" * 50)

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device, loc, adj)
    train_losses.append(train_loss)

    val_loss = validate_epoch(model, val_loader, criterion, device, loc, adj)
    val_losses.append(val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    scheduler.step(val_loss)
    early_stopping(val_loss, model, epoch)

    if early_stopping.early_stop:
        print("Early stopping")
        break

# ============ 繪製損失曲線 ============
plt.figure(figsize=(12, 8))
plt.plot(train_losses, label='Train Loss', color='g')
plt.plot(val_losses, label='Validation Loss', color='b')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'./train_fig/loss/{date}_loss.png')
plt.close()
print(f"\n✅ 損失曲線已儲存：./train_fig/loss/{date}_loss.png")