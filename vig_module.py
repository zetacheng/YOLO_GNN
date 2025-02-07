import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from grapher import GrapherModule, FFNModule
from patch_embedding import PatchEmbedding
from dim_manager import Dim_Manager # ✅ 引入維度管理

class ViG(nn.Module):
    def __init__(self):
        super(ViG, self).__init__()
        self.dim_manager = Dim_Manager()
        feature_dim = self.dim_manager.enquireDimValue("ViG_feature_dim")

        self.patch_embed = PatchEmbedding(in_channels=3)
        self.grapher = GrapherModule(feature_dim, feature_dim)  
        self.ffn = FFNModule(feature_dim, feature_dim)

    def forward(self, x):
        x = self.patch_embed(x).contiguous()   # ✅ Patch 嵌入 (batch_size, num_patches, embed_dim)
        x, edge_index = self.grapher(x)  # ✅ 透過 GNN 生成局部圖結構 (x, edge_index)
        x = self.ffn(x, edge_index)  # ✅ 透過 GNN 變換特徵

        # 🔍 Debugging
        #print(f"🔍 [DEBUG] ViG Output: Features Shape: {x.shape}, Edge Index Shape: {edge_index.shape}")

        return x, edge_index  # ✅ 返回特徵和 Edge Index，讓 ViG-GNN 進一步處理
