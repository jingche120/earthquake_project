import torch
import torch.nn as nn
import numpy as np
import sys
import os

# ================= 設定區 =================
PYTORCH_MODEL_PATH = "./epoch11.pth"   # PyTorch 模型路徑
DATA_BIN_PATH = "input_data.bin"       # 輸入資料路徑
ONNX_EXPORT_PATH = "tc_unet.onnx"      # ONNX 輸出路徑
BATCH_SIZE = 128
CHANNELS = 3
LENGTH = 1000
# ==========================================

# --- 1. 載入模型定義 (TC_UNet.py) ---
try:
    import TC_UNet
except ImportError:
    print("錯誤：找不到 TC_UNet.py，請確認它在同一個資料夾。")
    sys.exit(1)

# 修正 torch.load 的路徑問題
main_mod = sys.modules['__main__']
main_mod.CBAM = TC_UNet.CBAM
main_mod.ChannelAttention = TC_UNet.ChannelAttention
main_mod.SpatialAttention = TC_UNet.SpatialAttention
main_mod.PositionalEncoding = TC_UNet.PositionalEncoding
main_mod.UNet = TC_UNet.UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 準備輸入資料 ---
if os.path.exists(DATA_BIN_PATH):
    # print(f"[Python] 讀取現有檔案: {DATA_BIN_PATH}")
    try:
        numpy_input = np.fromfile(DATA_BIN_PATH, dtype=np.float32)
        numpy_input = numpy_input.reshape(BATCH_SIZE, CHANNELS, LENGTH)
        input_tensor = torch.from_numpy(numpy_input).to(device)
    except Exception as e:
        print(f"讀取資料失敗: {e}")
        sys.exit(1)
else:
    print(f"[Python] 警告：找不到 {DATA_BIN_PATH}，正在生成隨機資料...")
    input_tensor = torch.rand(BATCH_SIZE, CHANNELS, LENGTH).to(device)
    input_tensor = (input_tensor * 20) - 10
    # 存檔
    numpy_input = input_tensor.cpu().numpy().astype(np.float32)
    numpy_input.tofile(DATA_BIN_PATH)

# --- 3. 執行預測 ---
try:
    model = torch.load(PYTORCH_MODEL_PATH, map_location=device, weights_only=False)
    model = model.float().eval()
    
    with torch.no_grad():
        # output_mask 是 [128, 1, 1000], event_prob 是 [128, 1]
        output_mask, event_prob = model(input_tensor)
        
    # 轉成 numpy 方便列印
    prob_np = event_prob.cpu().numpy()
    
except Exception as e:
    print(f"模型執行失敗: {e}")
    sys.exit(1)

# --- 4. 格式化輸出 (為了跟 C++ 一模一樣) ---
print("\n" + "="*30)
print(" [Python] 預測結果 (Event Prob List):")
print("="*30)




















# 將 numpy array 展平並轉成 list
flat_list = prob_np.flatten().tolist()

# 使用字串格式化，確保每個數字都是小數點後 8 位，並用逗號隔開
formatted_output = "[" + ",\n".join(f"{x:.8f}" for x in flat_list) + "]"

print(formatted_output)
print("="*30 + "\n")

# --- 5. 匯出 ONNX (如果還沒匯出的話) ---
if not os.path.exists(ONNX_EXPORT_PATH):
    # print(f"[Python] 正在匯出 ONNX: {ONNX_EXPORT_PATH}")
    torch.onnx.export(
        model, input_tensor, ONNX_EXPORT_PATH,
        input_names=["input"], output_names=["output_mask", "event_prob"],
        opset_version=14,
        dynamic_axes={'input': {0: 'batch_size'}, 'output_mask': {0: 'batch_size'}, 'event_prob': {0: 'batch_size'}}
    )