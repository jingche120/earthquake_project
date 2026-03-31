"""
動態現地型地震預警系統 - 模型實作
Architecture: Attentive Neural Processes (Deterministic-only Path)
Waveform Extractor: ViT-based (Saad et al., 2024 DLPGA)

規格書對應:
  - Module A: Context Encoder (Source → K, V)
  - Module B: Target Query   (Target → Q)
  - Module C: Spatial Cross-Attention (Q, K, V → C_tgt)
  - Module D: Regression Decoder (C_tgt → PGA prediction)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 0. Hyperparameter Config
# =============================================================================
class ModelConfig:
    """合理預設值，可自行調整"""
    # 資料維度
    N_SRC = 21          # 來源測站數
    M_TGT = 166         # 目標測站數
    IN_CHANNELS = 3     # 波形通道 (Z, N, E)
    SEQ_LEN = 3000      # 採樣點數 (100 Hz × 30 s)

    # 空間編碼器
    d_loc = 64          # 空間語意向量維度
    pe_freqs = 32       # 正弦/餘弦位置編碼頻率數 → 編碼後維度 = 2 * 2 * pe_freqs = 128

    # 波形萃取器 (DLPGA ViT)
    cnn_feat_maps = 128
    cnn_kernel = 3
    cnn_dropout = 0.1
    patch_size = 5
    proj_dim = 100      # Patch embedding projection dimension
    n_transformer = 2   # Transformer 層數
    n_heads = 2         # MHA head 數
    tfm_mlp_dim = 200   # Transformer 內 MLP 第一層
    tfm_dropout = 0.1
    pool_dropout = 0.5  # Transformer 輸出後的 dropout

    d_wave = 128        # 波形特徵最終維度 (pool 後再投影)

    # Cross-Attention
    d = 128             # Q, K, V 維度

    # Decoder
    decoder_hidden = 128

    # Loss
    lambd = 5.0         # Asymmetric MSE 低估懲罰係數


# =============================================================================
# 1. 共用子網路: 空間編碼器 (Spatial Encoder)
# =============================================================================
class SinCosPositionalEncoding(nn.Module):
    """
    將 2D 座標 (lon, lat) 透過多頻率正弦/餘弦展開，
    增強空間解析度。
    輸入: (*, 2)
    輸出: (*, 2 * 2 * n_freqs)  即每個座標軸 × sin/cos × n_freqs
    """
    def __init__(self, n_freqs: int = 32):
        super().__init__()
        # 頻率從 2^0 到 2^(n_freqs-1)，取 log-linear spacing
        freqs = 2.0 ** torch.linspace(0, n_freqs - 1, n_freqs)  # (n_freqs,)
        self.register_buffer("freqs", freqs)
        self.out_dim = 2 * 2 * n_freqs  # lon×(sin,cos)×n_freqs + lat×(sin,cos)×n_freqs

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (*, 2)"""
        # coords_expanded: (*, 2, 1) * (n_freqs,) → (*, 2, n_freqs)
        scaled = coords.unsqueeze(-1) * self.freqs  # (*, 2, n_freqs)
        # sin + cos → (*, 2, 2*n_freqs) → flatten → (*, 4*n_freqs)
        encoded = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)  # (*, 2, 2*n_freqs)
        return encoded.flatten(-2)  # (*, 4*n_freqs)


class SpatialEncoder(nn.Module):
    """
    規格書 §3.1: 座標 → 正弦/餘弦位置編碼 → MLP → d_loc
    共享於 Source 與 Target。
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.pe = SinCosPositionalEncoding(n_freqs=cfg.pe_freqs)
        pe_dim = self.pe.out_dim  # 4 * pe_freqs = 128
        self.mlp = nn.Sequential(
            nn.Linear(pe_dim, 128),
            nn.ReLU(),
            nn.Linear(128, cfg.d_loc),
            nn.ReLU(),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (B, S, 2)  — S 可以是 N 或 M
        return: (B, S, d_loc)
        """
        h = self.pe(coords)    # (B, S, pe_dim)
        return self.mlp(h)     # (B, S, d_loc)


# =============================================================================
# 2. 共用子網路: 波形特徵萃取器 (Waveform Extractor — DLPGA ViT)
# =============================================================================

class ConvBlock(nn.Module):
    """單一 Conv Block: Conv1D → BN → ReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvolutionBlock(nn.Module):
    """
    一個 Convolution Block = 3 個 ConvBlock + Residual + Dropout
    (論文 Fig. 2: input → CB1 → CB2 → (+input) → CB3 → Dropout)
    """
    def __init__(self, in_ch, feat_maps=128, kernel_size=3, dropout=0.1):
        super().__init__()
        self.cb1 = ConvBlock(in_ch, feat_maps, kernel_size)
        self.cb2 = ConvBlock(feat_maps, feat_maps, kernel_size)
        self.cb3 = ConvBlock(feat_maps, feat_maps, kernel_size)
        self.dropout = nn.Dropout(dropout)
        # 如果 in_ch != feat_maps，需要 1x1 conv 做 residual 匹配
        self.residual_proj = nn.Conv1d(in_ch, feat_maps, 1) if in_ch != feat_maps else nn.Identity()

    def forward(self, x):
        """x: (B*N, C, L)"""
        h = self.cb1(x)
        h = self.cb2(h)
        h = h + self.residual_proj(x)  # Residual connection
        h = self.cb3(h)
        h = self.dropout(h)
        return h


class PatchEmbedding(nn.Module):
    """
    將 1D feature map 切成 non-overlapping patches 並投影到 proj_dim。
    輸入: (B*N, feat_maps, L)
    輸出: (B*N, n_patches, proj_dim)
    """
    def __init__(self, feat_maps, seq_len, patch_size=5, proj_dim=100):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        # 每個 patch: feat_maps × patch_size → proj_dim
        self.proj = nn.Linear(feat_maps * patch_size, proj_dim)
        # 可學習的位置嵌入
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, proj_dim) * 0.02)

    def forward(self, x):
        """x: (B*N, feat_maps, L)"""
        B, C, L = x.shape
        # Reshape into patches: (B*N, n_patches, C * patch_size)
        x = x[:, :, :self.n_patches * self.patch_size]  # 截斷到可整除
        x = x.reshape(B, C, self.n_patches, self.patch_size)  # (B*N, C, P, ps)
        x = x.permute(0, 2, 1, 3).reshape(B, self.n_patches, C * self.patch_size)  # (B*N, P, C*ps)
        x = self.proj(x) + self.pos_embed  # (B*N, P, proj_dim)
        return x


class TransformerBlock(nn.Module):
    """
    標準 Pre-Norm Transformer Block:
    Norm → MHA → Residual → Norm → MLP → Residual
    """
    def __init__(self, dim, n_heads=2, mlp_dim=200, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: (B*N, P, dim)"""
        h = self.norm1(x)
        h, _ = self.mha(h, h, h)
        x = x + h
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x


class WaveformExtractor(nn.Module):
    """
    規格書 §3.2 + 論文 Saad et al. 2024 DLPGA ViT
    架構: 2 × ConvolutionBlock → PatchEmbedding → N × Transformer → Pool → Linear → d_wave

    額外: 將 max absolute amplitude 拼接到最終特徵（論文設計）。

    輸入: (B, N, 3, 3000)
    輸出: (B, N, d_wave)
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # --- CNN Backbone ---
        self.conv_block1 = ConvolutionBlock(
            in_ch=cfg.IN_CHANNELS, feat_maps=cfg.cnn_feat_maps,
            kernel_size=cfg.cnn_kernel, dropout=cfg.cnn_dropout
        )
        self.conv_block2 = ConvolutionBlock(
            in_ch=cfg.cnn_feat_maps, feat_maps=cfg.cnn_feat_maps,
            kernel_size=cfg.cnn_kernel, dropout=cfg.cnn_dropout
        )

        # --- Patch Encoder ---
        self.patch_embed = PatchEmbedding(
            feat_maps=cfg.cnn_feat_maps, seq_len=cfg.SEQ_LEN,
            patch_size=cfg.patch_size, proj_dim=cfg.proj_dim
        )

        # --- Transformer Layers ---
        self.transformers = nn.ModuleList([
            TransformerBlock(
                dim=cfg.proj_dim, n_heads=cfg.n_heads,
                mlp_dim=cfg.tfm_mlp_dim, dropout=cfg.tfm_dropout
            )
            for _ in range(cfg.n_transformer)
        ])
        self.pool_dropout = nn.Dropout(cfg.pool_dropout)

        # --- 最終投影: (proj_dim + 1) → d_wave ---
        # +1 是 max absolute amplitude
        self.out_proj = nn.Linear(cfg.proj_dim + 1, cfg.d_wave)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        waveforms: (B, N, 3, 3000)
        return:    (B, N, d_wave)
        """
        B, N, C, L = waveforms.shape

        # 計算 max absolute amplitude (論文設計)
        max_amp = waveforms.abs().flatten(2).max(dim=-1).values  # (B, N)
        max_amp = max_amp.unsqueeze(-1)  # (B, N, 1)

        # Reshape for CNN: (B*N, 3, 3000)
        x = waveforms.reshape(B * N, C, L)

        # CNN
        x = self.conv_block1(x)  # (B*N, 128, 3000)
        x = self.conv_block2(x)  # (B*N, 128, 3000)

        # Patch Embedding
        x = self.patch_embed(x)  # (B*N, n_patches, proj_dim)

        # Transformer
        for tfm in self.transformers:
            x = tfm(x)  # (B*N, n_patches, proj_dim)

        x = self.pool_dropout(x)

        # Global Average Pooling over patches → (B*N, proj_dim)
        x = x.mean(dim=1)

        # Reshape back: (B, N, proj_dim)
        x = x.reshape(B, N, -1)

        # 拼接 max amplitude → (B, N, proj_dim + 1)
        x = torch.cat([x, max_amp], dim=-1)

        # 投影到 d_wave
        x = self.out_proj(x)  # (B, N, d_wave)
        return x


# =============================================================================
# 3. Module A: 動態上下文編碼器 (Context Encoder)
# =============================================================================
class ContextEncoder(nn.Module):
    """
    規格書 §4 Module A:
    Source 波形 → WaveformExtractor → h_wave
    Source 座標 → SpatialEncoder   → h_loc_src
    K = Linear(h_loc_src)
    V = Linear(concat(h_wave, h_loc_src))
    """
    def __init__(self, spatial_encoder: SpatialEncoder, waveform_extractor: WaveformExtractor, cfg: ModelConfig):
        super().__init__()
        self.spatial_enc = spatial_encoder
        self.wave_ext = waveform_extractor
        self.key_proj = nn.Linear(cfg.d_loc, cfg.d)
        self.value_proj = nn.Linear(cfg.d_wave + cfg.d_loc, cfg.d)

    def forward(self, y_src: torch.Tensor, x_src: torch.Tensor):
        """
        y_src: (B, N, 3, 3000)  — 來源波形
        x_src: (B, N, 2)        — 來源座標
        return: K (B, N, d), V (B, N, d)
        """
        h_wave = self.wave_ext(y_src)          # (B, N, d_wave)
        h_loc_src = self.spatial_enc(x_src)    # (B, N, d_loc)

        K = self.key_proj(h_loc_src)                              # (B, N, d)
        V = self.value_proj(torch.cat([h_wave, h_loc_src], dim=-1))  # (B, N, d)
        return K, V


# =============================================================================
# 4. Module B: 空間條件查詢 (Target Query)
# =============================================================================
class TargetQuery(nn.Module):
    """
    規格書 §4 Module B:
    Target 座標 → 共享 SpatialEncoder → h_loc_tgt
    Q = Linear(h_loc_tgt)
    """
    def __init__(self, spatial_encoder: SpatialEncoder, cfg: ModelConfig):
        super().__init__()
        self.spatial_enc = spatial_encoder
        self.query_proj = nn.Linear(cfg.d_loc, cfg.d)

    def forward(self, x_tgt: torch.Tensor):
        """
        x_tgt: (B, M, 2)
        return: Q (B, M, d), h_loc_tgt (B, M, d_loc) — 保留供 decoder 使用
        """
        h_loc_tgt = self.spatial_enc(x_tgt)   # (B, M, d_loc)
        Q = self.query_proj(h_loc_tgt)        # (B, M, d)
        return Q, h_loc_tgt


# =============================================================================
# 5. Module C: 空間交叉注意力 (Spatial Cross-Attention)
# =============================================================================
class SpatialCrossAttention(nn.Module):
    """
    規格書 §4 Module C:
    C_tgt = Softmax(Q @ K^T / sqrt(d)) @ V
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.scale = math.sqrt(cfg.d)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        """
        Q: (B, M, d)
        K: (B, N, d)
        V: (B, N, d)
        return: C_tgt (B, M, d)
        """
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, M, N)
        attn_weights = F.softmax(attn_scores, dim=-1)                 # (B, M, N)
        C_tgt = torch.bmm(attn_weights, V)                            # (B, M, d)
        return C_tgt, attn_weights


# =============================================================================
# 6. Module D: 回歸解碼器 (Regression Decoder)
# =============================================================================
class RegressionDecoder(nn.Module):
    """
    規格書 §4 Module D:
    Residual: output = C_tgt + Q   → (B, M, d)
    MLP → PGA prediction (B, M, 1)
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(cfg.d, cfg.decoder_hidden),
            nn.ReLU(),
            nn.Linear(cfg.decoder_hidden, cfg.decoder_hidden),
            nn.ReLU(),
            nn.Linear(cfg.decoder_hidden, 1),
        )

    def forward(self, C_tgt: torch.Tensor, Q: torch.Tensor):
        """
        C_tgt: (B, M, d)
        Q:     (B, M, d)
        return: PGA_pred (B, M, 1)
        """
        h = C_tgt + Q  # Residual connection, (B, M, d)
        return self.decoder(h)  # (B, M, 1)


# =============================================================================
# 7. 損失函數: 不對稱 MSE (Asymmetric MSE)
# =============================================================================
class AsymmetricMSELoss(nn.Module):
    """
    規格書 §5:
    e = PGA_real - PGA_pred
    e > 0 (低估/漏報) → λ * e²
    e ≤ 0 (高估)      → e²
    """
    def __init__(self, lambd: float = 5.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, pga_pred: torch.Tensor, pga_real: torch.Tensor):
        """
        pga_pred: (B, M, 1)
        pga_real: (B, M, 1)
        return: scalar loss
        """
        e = pga_real - pga_pred                          # 正值 = 低估
        e2 = e ** 2
        underest_mask = (e > 0).float()
        weighted = underest_mask * self.lambd * e2 + (1 - underest_mask) * e2
        return weighted.mean()


# =============================================================================
# 8. 完整模型: EEW-ANP
# =============================================================================
class EEWANP(nn.Module):
    """
    動態現地型地震預警系統 — 完整前向傳播

    Forward:
      Source (y_src, x_src) → ContextEncoder → K, V
      Target (x_tgt)        → TargetQuery    → Q
      Cross-Attention(Q, K, V)               → C_tgt
      Decoder(C_tgt + Q)                     → PGA_pred (B, M, 1)
    """
    def __init__(self, cfg: ModelConfig = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg

        # 共享空間編碼器
        self.spatial_encoder = SpatialEncoder(cfg)

        # 波形萃取器
        self.waveform_extractor = WaveformExtractor(cfg)

        # 四大模組
        self.context_encoder = ContextEncoder(self.spatial_encoder, self.waveform_extractor, cfg)
        self.target_query = TargetQuery(self.spatial_encoder, cfg)
        self.cross_attention = SpatialCrossAttention(cfg)
        self.decoder = RegressionDecoder(cfg)

    def forward(self, y_src, x_src, x_tgt):
        """
        y_src: (B, N, 3, 3000)  — 來源測站波形
        x_src: (B, N, 2)        — 來源測站座標
        x_tgt: (B, M, 2)        — 目標測站座標
        return: PGA_pred (B, M, 1)
        """
        K, V = self.context_encoder(y_src, x_src)   # (B, N, d), (B, N, d)
        Q, _ = self.target_query(x_tgt)             # (B, M, d)
        C_tgt, attn_w = self.cross_attention(Q, K, V)  # (B, M, d)
        pga_pred = self.decoder(C_tgt, Q)            # (B, M, 1)
        return pga_pred

    def predict_with_attention(self, y_src, x_src, x_tgt):
        """同 forward，但額外回傳 attention weights 供分析"""
        K, V = self.context_encoder(y_src, x_src)
        Q, _ = self.target_query(x_tgt)
        C_tgt, attn_w = self.cross_attention(Q, K, V)
        pga_pred = self.decoder(C_tgt, Q)
        return pga_pred, attn_w


# =============================================================================
# 9. 測試: Shape Verification
# =============================================================================
if __name__ == "__main__":
    cfg = ModelConfig()
    model = EEWANP(cfg)

    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 假資料測試
    B = 2
    y_src = torch.randn(B, cfg.N_SRC, cfg.IN_CHANNELS, cfg.SEQ_LEN)
    x_src = torch.randn(B, cfg.N_SRC, 2)
    x_tgt = torch.randn(B, cfg.M_TGT, 2)

    print(f"\n--- Input Shapes ---")
    print(f"y_src: {y_src.shape}")
    print(f"x_src: {x_src.shape}")
    print(f"x_tgt: {x_tgt.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        pga_pred = model(y_src, x_src, x_tgt)
    print(f"\n--- Output ---")
    print(f"PGA_pred: {pga_pred.shape}")  # 預期: (2, 166, 1)

    # Loss 測試
    pga_real = torch.randn(B, cfg.M_TGT, 1).abs() * 10
    criterion = AsymmetricMSELoss(lambd=cfg.lambd)
    loss = criterion(pga_pred, pga_real)
    print(f"Asymmetric MSE Loss: {loss.item():.4f}")

    # Attention weights
    with torch.no_grad():
        _, attn_w = model.predict_with_attention(y_src, x_src, x_tgt)
    print(f"Attention weights: {attn_w.shape}")  # 預期: (2, 166, 21)
    print(f"Attention sum per target (should be 1.0): {attn_w[0, 0].sum().item():.6f}")
