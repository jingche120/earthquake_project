import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # [新增] 引入 pandas 來畫漂亮的表格

from dataprecess import PGADataset
from model import DLPGA_Model

def train_kfold_model():
    batch_size = 16
    epochs = 50
    learning_rate = 1e-3
    k_folds = 5
    
    print(" 正在載入資料集...")
    dataset = PGADataset('3000_p_wave_dataset.npz')
    
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    all_folds_train_loss = np.zeros((k_folds, epochs))
    all_folds_test_loss = np.zeros((k_folds, epochs))
    
    # [新增] 全局最佳模型紀錄器
    best_global_test_loss = float('inf') 
    best_model_path = 'best_dlpga_model.pth'
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\n 開始 {k_folds}-Fold 交叉驗證 (Device: {device})...")
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f"\n{'='*15} 正在執行 Fold {fold+1}/{k_folds} {'='*15}")
        
        train_sub = Subset(dataset, train_ids)
        test_sub = Subset(dataset, test_ids)
        
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_sub, batch_size=batch_size, shuffle=False)
        
        model = DLPGA_Model().to(device)
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            # 🏋️訓練階段
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
            all_folds_train_loss[fold, epoch] = avg_train_loss
            
            # 測試階段
            model.eval()
            test_epoch_loss = 0.0
            with torch.no_grad():
                for X_test, y_test in test_loader:
                    X_test, y_test = X_test.to(device), y_test.to(device)
                    test_preds = model(X_test)
                    test_loss = criterion(test_preds, y_test)
                    test_epoch_loss += test_loss.item()
                    
            avg_test_loss = test_epoch_loss / len(test_loader)
            all_folds_test_loss[fold, epoch] = avg_test_loss
            
            # [新增] 檢查是否破紀錄，若是則儲存權重！
            if avg_test_loss < best_global_test_loss:
                best_global_test_loss = avg_test_loss
                # 儲存模型權重 (state_dict)
                torch.save(model.state_dict(), best_model_path)
            
            if epoch == epochs - 1:
                print(f"Fold {fold+1} 最終成績 -> Train MAE: {avg_train_loss:.4f} | Test MAE: {avg_test_loss:.4f}")

    print("\n 5-Fold 交叉驗證訓練結束！")
    print(f"最佳模型權重已儲存至: {best_model_path} (Test MAE: {best_global_test_loss:.4f})")
    
    # 結算平均成績
    avg_train_curve = all_folds_train_loss.mean(axis=0)
    avg_test_curve = all_folds_test_loss.mean(axis=0)
    
    # ==========================================
    # [新增] 繪製特定 Epoch 的成績表格
    # ==========================================
    print("\n ===  模型訓練階段成績單 ===")
    # 陣列 index 從 0 開始，所以 Epoch 1, 25, 50 分別對應 index 0, 24, 49
    target_epochs = [1, 25, 50]
    indices = [e - 1 for e in target_epochs]
    
    # 使用 DataFrame 讓終端機印出來整齊漂亮
    report_data = {
        "Epoch": target_epochs,
        "訓練集平均 MAE": avg_train_curve[indices].round(4),
        "測試集平均 MAE": avg_test_curve[indices].round(4)
    }
    df_report = pd.DataFrame(report_data)
    # 隱藏 pandas 的 index 以求美觀
    print(df_report.to_string(index=False))
    print("======================================\n")
    
    # 繪製曲線 (與原本相同)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), avg_train_curve, linestyle='-', color='b', label='Average Train Loss')
    plt.plot(range(1, epochs + 1), avg_test_curve, linestyle='-', color='r', label='Average Test Loss')
    
    std_test_curve = all_folds_test_loss.std(axis=0)


    plt.title(f'DLPGA Model - {k_folds}-Fold Cross Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.grid(True)
    plt.legend()
    plt.savefig('kfold_train_test_loss_curve.png')
    plt.show()

if __name__ == "__main__":
    train_kfold_model()