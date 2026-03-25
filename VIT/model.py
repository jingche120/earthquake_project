import torch
import torch.nn as nn
from module_unit import ConvBlock128FM, PatchEncoderEmbedding, TransformerBlock
# 主模型: DLPGA_Model
# ==========================================
class DLPGA_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN 特徵提取
        self.conv_block1 = ConvBlock128FM(in_channels=3)
        self.conv_block2 = ConvBlock128FM(in_channels=128)
        
        # Patch 切割與 Embedding
        self.patch_encoder = PatchEncoderEmbedding()
        
        # 兩層 Transformer
        self.transformer1 = TransformerBlock()
        self.transformer2 = TransformerBlock()
        
        # 後處理與預測
        self.global_dropout = nn.Dropout(0.5)
        # Flatten 後的維度: 80 tokens * 100 dim = 8000. 加上物理特徵 1 = 8001
        self.final_fc = nn.Linear(8001, 1)

    def forward(self, x):
        print("\n🚀 === 模型 Forward Pass 開始 ===")
        print(f"[DLPGA_Model] 原始輸入形狀: {x.shape}")
        
        # 保存原始輸入以提取物理特徵
        raw_x = x 
        
        # 1. CNN Blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        
        # 2. Patch & Embedding
        x = self.patch_encoder(x)
        
        # 3. Transformer Blocks
        x = self.transformer1(x)
        x = self.transformer2(x)
        
        # 4. Feature Fusion
        x = self.global_dropout(x)
        
        # 攤平 (B, 80, 100) -> (B, 8000)
        flattened_x = x.view(x.size(0), -1) 
        
        # 提取物理特徵：最大絕對振幅 (B, 3, 400) -> (B, 1)
        # 先取絕對值，拉平成 (B, 1200)，找最大值，然後保持 2D 形狀
        max_amp = raw_x.abs().view(raw_x.size(0), -1).max(dim=1)[0].unsqueeze(1)
        
        # Concat 拼接 (B, 8000) 和 (B, 1) -> (B, 8001)
        fused_features = torch.cat([flattened_x, max_amp], dim=1)
        print(f"[Feature Fusion] Concat 後形狀: {fused_features.shape}")
        
        # 5. 最終預測
        pred_pga = self.final_fc(fused_features)
        print(f"🎯 [DLPGA_Model] 最終預測 PGA 形狀: {pred_pga.shape}")
        print("🏁 === 模型 Forward Pass 結束 ===\n")
        
        return pred_pga

# ==========================================
# 測試驅動程式 (Dummy Data Test)
# ==========================================
if __name__ == "__main__":
    # 建立 Dummy Data: Batch Size = 32, Channels = 3, Seq Length = 400
    dummy_input = torch.randn(32, 3, 400)
    
    # 實例化模型
    model = DLPGA_Model()
    
    # 執行 Forward Pass
    output = model(dummy_input)