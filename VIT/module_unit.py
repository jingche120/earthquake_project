import torch
import torch.nn as nn

# ==========================================
# 模組 1: Convolution Block 128 FM
# ==========================================
class ConvBlock128FM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block_name = f"ConvBlock128FM (in={in_channels})"
        #* conv1d
        # [備註] PyTorch 1D 卷積預設輸入為 (Batch, Channels, Length)。卷積核只會在最後一個維度（時間軸 L=400）上滑動，不會跨 Channel 滑動。
        # 三個小的 Conv Block (Conv1d -> BN -> ReLU)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.1)
        
        # 處理 Residual Add 的維度匹配 (針對第一層 3->128)
        # [備註] 執行 Element-wise Add (+) 時形狀必須完全相同。第一層 CNN 原始輸入為 3 通道，輸出為 128 通道，必須安插一個 kernel_size=1 的 Conv1d 進行升維投影後才能相加。
        self.shortcut = nn.Conv1d(in_channels, 128, kernel_size=1) if in_channels != 128 else nn.Identity()

    def forward(self, x):
        print(f"[{self.block_name}] 📥 進入形狀: {x.shape}")
        
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        
        # Residual Add (Bug 1 修正處：正確的捷徑相加)
        res = x2 + self.shortcut(x)
        
        x3 = self.conv3(res)
        out = self.dropout(x3)
        
        print(f"[{self.block_name}] 📤 出去形狀: {out.shape}")
        return out
    
# ==========================================
# 模組 2: Patch Encoder and Embedding
# ==========================================
class PatchEncoderEmbedding(nn.Module):
    def __init__(self, in_channels=128, seq_len=400, patch_size=5, embed_dim=100):
        super().__init__()
        self.num_patches = seq_len // patch_size
        self.patch_size = patch_size
        flattened_dim = in_channels * patch_size # 128 * 5 = 640
        
        self.linear_proj = nn.Linear(flattened_dim, embed_dim)
        # Positional Embedding (Bug 2 修正處：形狀為 1 x P x R)
        # [備註] 位置編碼宣告為 (1, P, R)。Batch 維度設為 1 是為了利用 PyTorch Broadcasting，節省記憶體並適應不同的 Batch Size。
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    def forward(self, x):
        print(f"[PatchEncoder] 📥 進入形狀: {x.shape}")
        B, C, L = x.shape
        
        # 切塊與 Reshape (B, 128, 400) -> (B, 128, 80, 5) -> (B, 80, 128, 5)
        x = x.view(B, C, self.num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).contiguous()
        
        # 攤平 (B, 80, 128, 5) -> (B, 80, 640)
        x = x.view(B, self.num_patches, -1)
        
        # Linear 投影 (B, 80, 640) -> (B, 80, 100)
        tokens = self.linear_proj(x)
        
        # 加上位置編碼
        out = tokens + self.pos_embed
        
        print(f"[PatchEncoder] 📤 出去形狀: {out.shape}")
        return out


# ==========================================
# 模組 3: Transformer Block
# ==========================================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=100, num_heads=2):
        super().__init__()
        # PyTorch 的 LayerNorm 預設作用在最後一個維度 (embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        # batch_first=True 讓輸入形狀維持 (B, Seq, Feature)
        # [備註] MHA 切分（Split）的是「特徵維度 (embed_dim)」，絕對不是「序列長度 (seq_len)」。80 個 Token 的全局視野不能被切斷。
        #* 是切最後一個維度
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) 
        
        self.norm2 = nn.LayerNorm(embed_dim)
        # MLP 區塊 (200 神經元 -> ReLU -> 100 神經元 -> Dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 200),
            nn.ReLU(),
            nn.Linear(200, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        print(f"[TransformerBlock] 📥 進入形狀: {x.shape}")
        
        # Norm -> MHA -> Add
        norm_x = self.norm1(x)
        #* 在呼叫 PyTorch 的 nn.MultiheadAttention 時，括號內的前三個參數嚴格對應注意力機制的核心運算矩陣：Query (Q)、Key (K)、Value (V)。
        #* 如果是encoder跨decoder 這種cross attention 就要注意 Query (Q) 是來自 Decoder，而 Key (K) 和 Value (V) 才是來自 Encoder 。
        #* Decoder 拿著自己的問題 (Q)，去翻閱 Encoder 整理好的目錄 (K)，最後把 Encoder 的精華內容 (V) 抄過來
        attn_out, _ = self.mha(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # Norm -> MLP -> Add
        norm_x2 = self.norm2(x)
        mlp_out = self.mlp(norm_x2)
        out = x + mlp_out
        
        print(f"[TransformerBlock] 📤 出去形狀: {out.shape}")
        return out