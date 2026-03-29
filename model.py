import os
import glob
import h5py
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, Conv1d, Linear, Dropout, MaxPool1d
from torch.nn import BatchNorm1d
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import math

seed = 42
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
np.random.permutation(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 測站數設定 ============
NUM_STATIONS = 166
# ====================================

class PositionEmbedding(nn.Module):
    def __init__(self, emb_dim, rotation=None, rotation_anchor=None):
        super(PositionEmbedding, self).__init__()
        wavelengths = ((0.01, 10), (0.01, 10)) 
        self.emb_dim = emb_dim
        self.wavelengths = wavelengths
        self.rotation = rotation
        self.rotation_anchor = rotation_anchor

        if rotation is not None and rotation_anchor is None:
            raise ValueError('Rotations in the positional embedding require a rotation anchor')

        if rotation is not None:
            c, s = torch.cos(rotation), torch.sin(rotation)
            self.rotation_matrix = torch.tensor([[c, -s], [s, c]])
        else:
            self.rotation_matrix = None

        min_lat, max_lat = wavelengths[0]
        min_lon, max_lon = wavelengths[1]

        lat_dim = emb_dim // 4
        lon_dim = emb_dim // 4

        self.lat_coeff = 2 * torch.tensor(np.pi) * 1. / min_lat * ((min_lat / max_lat) ** (torch.arange(lat_dim) / lat_dim)).to(device)
        self.lon_coeff = 2 * torch.tensor(np.pi) * 1. / min_lon * ((min_lon / max_lon) ** (torch.arange(lon_dim) / lon_dim)).to(device)

        lat_sin_mask = torch.arange(emb_dim) % 4 == 0
        lat_cos_mask = torch.arange(emb_dim) % 4 == 1
        lon_sin_mask = torch.arange(emb_dim) % 4 == 2
        lon_cos_mask = torch.arange(emb_dim) % 4 == 3
       
        self.mask = torch.zeros(emb_dim, dtype=torch.int64).to(device)
        self.mask[lat_sin_mask] = torch.arange(lat_dim).to(device)
        self.mask[lat_cos_mask] = lat_dim + torch.arange(lat_dim).to(device)
        self.mask[lon_sin_mask] = 2 * lat_dim + torch.arange(lon_dim).to(device)
        self.mask[lon_cos_mask] = 2 * lat_dim + lon_dim + torch.arange(lon_dim).to(device)

    def forward(self, x, mask=None):
        if self.rotation is not None:
            lat_base = x[:, :, 0]
            lon_base = x[:, :, 1]
            lon_base *= torch.cos(lat_base * np.pi / 180)

            lat_base -= self.rotation_anchor[0]
            lon_base -= self.rotation_anchor[1] * torch.cos(self.rotation_anchor[0] * math.pi / 180)

            latlon = torch.stack([lat_base, lon_base], dim=-1)
            rotated = torch.matmul(latlon, self.rotation_matrix)

            lat_base = rotated[:, :, 0:1] * self.lat_coeff
            lon_base = rotated[:, :, 1:2] * self.lon_coeff
        else:
            lat_base = x[:, :, 0:1] * self.lat_coeff.to(x.device)
            lon_base = x[:, :, 1:2] * self.lon_coeff.to(x.device)
    
        output = torch.cat([torch.sin(lat_base), torch.cos(lat_base),
                            torch.sin(lon_base), torch.cos(lon_base)], dim=-1).to(device)

        output = torch.index_select(output, 2, self.mask)

        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            output *= mask

        return output
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'emb_dim=' + str(self.emb_dim) + ')'

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.2,
                 leaky_relu_negative_slope: float = 0.2
                 ):
        super().__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features
        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        batch_size = h.shape[0]
        num_nodes = h.shape[1]
        h=h.float()
        g = self.linear(h).view(batch_size, num_nodes, self.n_heads, self.n_hidden)
        g_repeat = g.repeat(1,num_nodes,1,1)
        g_repeat_interleave = g.repeat_interleave(num_nodes, dim=1)
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        g_concat = g_concat.view(batch_size, num_nodes, num_nodes, self.n_heads, 2 * self.n_hidden)
        e = self.activation(self.attn(g_concat))
        e = e.squeeze(-1)
        e = e.masked_fill(adj_mat.unsqueeze(3) == 0, float('-inf'))
        a = self.softmax(e)
        a = a * adj_mat.unsqueeze(3)
        a = self.dropout(a)
        attn_res = torch.einsum('bijh,bjhf->bihf', a, g)

        if self.is_concat:
            attn_res = self.activation(attn_res.reshape(batch_size,num_nodes,self.n_heads * self.n_hidden))
            return attn_res
        else:
            return attn_res.mean(dim=2)

class GraphAttentionV2Layer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.2,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False
                 ):
        super().__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        batch_size, num_nodes, in_features = h.size()
        h=h.float()
        g_l = self.linear_l(h).view(batch_size, num_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(batch_size, num_nodes, self.n_heads, self.n_hidden)
        g_l_repeat = g_l.repeat(1,num_nodes,1,1)
        g_r_repeat_interleave = g_r.repeat_interleave(num_nodes, dim=1)
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(batch_size, num_nodes, num_nodes, self.n_heads, self.n_hidden)
        e = self.attn(self.activation(g_sum))
        e = e.squeeze(-1)
        e = e * adj_mat.unsqueeze(3)
        e = e.masked_fill(adj_mat.unsqueeze(3) == 0, float('-inf'))
        a = self.softmax(e)
        
        a = self.dropout(a)
        attn_res = torch.einsum('bijh,bjhf->bihf', a, g_r)

        if self.is_concat:
            attn_res = self.activation(attn_res.reshape(batch_size,num_nodes,self.n_heads * self.n_hidden))
            return attn_res
        else:
            return attn_res.mean(dim=2)
        
class ExpSparseGraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 top_m: int, dropout: float = 0.2, leaky_relu_negative_slope: float = 0.2):
        super().__init__()
        self.n_heads = n_heads
        self.top_m = top_m
        self.n_hidden = out_features // n_heads
        self.linear_l = nn.Linear(in_features, out_features, bias=False)
        self.linear_r = nn.Linear(in_features, out_features, bias=False)
        self.attn = nn.Linear(self.n_hidden , 1)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        batch_size, num_nodes, in_features = h.size()
        h=h.float()
        g_l = self.linear_l(h).view(batch_size, num_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(batch_size, num_nodes, self.n_heads, self.n_hidden)
        g_l_repeat = g_l.repeat(1,num_nodes,1,1)
        g_r_repeat_interleave = g_r.repeat_interleave(num_nodes, dim=1)
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(batch_size, num_nodes, num_nodes, self.n_heads, self.n_hidden)
        attn_scores = self.attn(self.activation(g_sum))
        attn_scores = attn_scores.squeeze(-1)
        
        attn_scores = attn_scores * adj.unsqueeze(3)
        attn_scores = attn_scores.masked_fill(adj.unsqueeze(3) == 0, float('-inf'))
        
        # Top-K 篩選
        topk_scores, topk_indices = torch.topk(attn_scores, self.top_m, dim=2)
        topk_mask = torch.zeros_like(attn_scores).scatter_(2, topk_indices, 1)

        attn_scores = attn_scores.masked_fill(topk_mask == 0, float('-inf'))
        attn_weights = self.softmax(attn_scores)
        
        attn_weights = self.dropout(attn_weights)
        output = torch.einsum('bijh,bjhf->bihf', attn_weights, g_r)
        output = self.activation(output.reshape(batch_size, num_nodes, -1))
        return output

class MultiPerspectiveGraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int, top_m: int,
                 dropout: float = 0.2, leaky_relu_negative_slope: float = 0.2):
        super().__init__()

        hidden = out_features
        self.global_attention = GraphAttentionV2Layer(in_features, hidden, n_heads,
                                                      dropout=dropout, leaky_relu_negative_slope=leaky_relu_negative_slope)
        self.local_attention = ExpSparseGraphAttentionLayer(in_features, hidden, n_heads, top_m=top_m,
                                                              dropout=dropout, leaky_relu_negative_slope=leaky_relu_negative_slope)

    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        global_output = self.global_attention(h, adj)
        local_output = self.local_attention(h, adj)
        combined_output = global_output + local_output
        return combined_output


class GATModel(nn.Module):
    def __init__(self, num_stations=NUM_STATIONS):
        super(GATModel, self).__init__()
        self.num_stations = num_stations

        # ---- 波形特徵提取（per-station，與測站數無關） ----
        self.conv1 = Conv2d(3, 8, (1, 1), stride=(1, 1))
        self.conv2 = Conv2d(8, 32, (1, 3), stride=(1, 3))
        self.conv3 = Conv1d(32, 64, 5, padding="same")
        self.fc1 = nn.Linear(64000, 1024)       # 64 * 1000 = 64000（不變）
        self.fc2 = nn.Linear(1026, 400)          # 1024 + 1(index) + 1(max_val) = 1026

        # ---- 位置編碼 ----
        self.sinembed = PositionEmbedding(400)

        # ---- GAT 層（與測站數無關，動態適應） ----
        self.gat1 = MultiPerspectiveGraphAttentionLayer(in_features=400, out_features=400, n_heads=1, top_m=5)
        self.gat2 = MultiPerspectiveGraphAttentionLayer(in_features=400, out_features=400, n_heads=1, top_m=5)

        # ---- 輸出層（改為 per-station 預測） ----
        # 原本：flatten 所有站 → fc → 16 站輸出
        # 現在：每站獨立 fc → 1 個 PGA 值
        self.fc_out1 = nn.Linear(400, 128)
        self.fc_out2 = nn.Linear(128, 1)

        self.dropout = Dropout(0.1)
        self.gelu = nn.GELU()

    def forward(self, batch_data, loc, adj):
        batch_size, sta_num, channal, dim = batch_data.shape
        x = batch_data.float()
        loc = loc.float()
        adj = adj.float()

        # ---- 提取每站的 max 特徵 ----
        abs_x = torch.abs(x)
        max_indices = torch.argmax(abs_x.view(batch_size, sta_num, -1), dim=-1, keepdim=True)
        max_abs_values = torch.gather(abs_x.view(batch_size, sta_num, -1), dim=-1, index=max_indices)
        max_indices = (max_indices % 3000 / 3000).float()

        # ---- 波形 CNN 特徵提取 ----
        x = x.permute(0, 2, 1, 3)                                    # [B, 3, S, 3000]
        x = self.gelu(self.conv1(x))                                  # [B, 8, S, 3000]
        x = self.gelu(self.conv2(x))                                  # [B, 32, S, 1000]
        x = x.permute(0, 2, 1, 3)                                    # [B, S, 32, 1000]
        x = torch.reshape(x, (-1, x.shape[2], x.shape[3]))           # [B*S, 32, 1000]
        x = self.gelu(self.conv3(x))                                  # [B*S, 64, 1000]
        x = self.dropout(x)
        x = torch.reshape(x, (-1, sta_num, x.shape[1] * x.shape[2])) # [B, S, 64000]
        x = self.fc1(x)                                               # [B, S, 1024]
        x = self.dropout(x)
        x = torch.cat((max_indices, max_abs_values, x), dim=-1)       # [B, S, 1026]
        x = self.fc2(x)                                               # [B, S, 400]

        # ---- 加上位置編碼 ----
        loc_encode = self.sinembed(loc)                                # [B, S, 400]
        x = torch.add(x, loc_encode)

        # ---- GAT 層 ----
        residual = x
        x = self.gat1(x, adj)
        x = self.gelu(x)
        x = torch.add(x, residual)

        residual = x
        x = self.gat2(x, adj)
        x = self.gelu(x)
        x = torch.add(x, residual)                                    # [B, S, 400]

        # ---- Per-station 輸出（不再 flatten 所有站） ----
        x = self.fc_out1(x)                                           # [B, S, 128]
        x = self.gelu(x)
        x = self.fc_out2(x)                                           # [B, S, 1]
        x = x.squeeze(-1)                                             # [B, S]
        x = F.relu(x)
        return x