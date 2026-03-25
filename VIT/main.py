import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd

# 引入自定義的模組
from dataprecess import PGADataset
from model import DLPGA_Model

def train_standard_model():
    # 參數設定
    batch_size = 16  # 如果資料變成 1000 筆，這裡可以嘗試調成 32 或 64
    epochs = 50
    learning_rate = 1e-3
    
    # 紀錄每個 Epoch 的 Loss
    train_history = []
    test_history = []
    
    print("📦 正在載入資料集...")
    # 這裡可以換成你的 3000_p_wave_dataset.npz
    dataset = PGADataset('3000_p_wave_dataset.npz') 
    
    # ==========================================
    #  1. 依照論文習慣：單次隨機切分 (例如 80% 訓練, 20% 測試)
    # ==========================================
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    
    # 固定 random_seed 確保每次切分結果一致，方便重現實驗
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ 資料切分完成！訓練集: {len(train_dataset)} 筆, 測試集: {len(test_dataset)} 筆。")
    
    # ==========================================
    #  2. 啟用 MPS (Mac GPU) 或 CUDA
    # ==========================================
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"\n🚀 開始標準訓練管線 (Device: {device})...")
    
    model = DLPGA_Model().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 最佳模型紀錄器
    best_test_loss = float('inf')
    best_model_path = 'best_dlpga_model_standard.pth'
    
    # ==========================================
    # 3. 進入單次 50 Epochs 迴圈
    # ==========================================
    for epoch in range(epochs):
        # 訓練階段
        model.train()
        train_epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            
        avg_train_loss = train_epoch_loss / len(train_loader)
        train_history.append(avg_train_loss)
        
        #  測試階段
        model.eval()
        test_epoch_loss = 0.0
        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                test_preds = model(X_test)
                test_loss = criterion(test_preds, y_test)
                test_epoch_loss += test_loss.item()
                
        avg_test_loss = test_epoch_loss / len(test_loader)
        test_history.append(avg_test_loss)
        
        # 檢查是否破紀錄，儲存權重
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), best_model_path)
            
        # 每 10 個 Epoch 印出一次進度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train MAE: {avg_train_loss:.4f} | Test MAE: {avg_test_loss:.4f}")

    print("\n 訓練結束！")
    print(f" 最佳模型權重已儲存至: {best_model_path} (Test MAE: {best_test_loss:.4f})")
    
    # ==========================================
    #  4. 繪製特定 Epoch 的成績表格
    # ==========================================
    print("\n === DLPGA 模型訓練階段成績單 ===")
    target_epochs = [1, 25, 50]
    indices = [e - 1 for e in target_epochs]
    
    report_data = {
        "Epoch": target_epochs,
        "訓練集平均 MAE": [round(train_history[i], 4) for i in indices],
        "測試集平均 MAE": [round(test_history[i], 4) for i in indices]
    }
    df_report = pd.DataFrame(report_data)
    print(df_report.to_string(index=False))
    print("======================================\n")
    
    # ==========================================
    #  5. 繪製 Train vs Test Loss 雙曲線
    # ==========================================
    print(" 正在繪製 Loss 曲線...")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_history, linestyle='-', color='b', label='Train Loss')
    plt.plot(range(1, epochs + 1), test_history, linestyle='-', color='r', label='Test Loss')

    plt.title('DLPGA Model - Train vs Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.grid(True)
    plt.legend()
    
    plt.savefig('standard_train_test_loss_curve.png')
    print(" 圖片已儲存為 standard_train_test_loss_curve.png")
    plt.show()

if __name__ == "__main__":
    train_standard_model()