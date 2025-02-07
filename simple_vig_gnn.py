import torch
import torch.nn as nn
import torch.nn.functional as F
from vig_module import ViG
from torch_geometric.nn import SAGEConv, knn_graph, global_mean_pool
from dim_manager import Dim_Manager  # ✅ 引入維度管理

class SimpleViG_GNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleViG_GNN, self).__init__()
        self.dim_manager = Dim_Manager()
        feature_dim = self.dim_manager.enquireDimValue("ViG_feature_dim")
        hidden_dim = self.dim_manager.enquireDimValue("GNN1_Output")
        self.knn_neighbors = self.dim_manager.enquireDimValue("knn_neighbors")
        self.batch_size = self.dim_manager.meta.batch_size  # ✅ **正確獲取 batch_size**
        self.num_gnn_layers = self.dim_manager.meta.num_gnn_layers  # ✅ 動態獲取 GNN 層數

        self.vig = ViG()

        self.gnn_layers = nn.ModuleList()
        for i in range(self.num_gnn_layers):
            in_dim = feature_dim if i == 0 else hidden_dim
            out_dim = num_classes if i == self.num_gnn_layers - 1 else hidden_dim
            self.gnn_layers.append(SAGEConv(in_dim, out_dim))

    def forward(self, x):
        # 1️⃣ ViG 轉換影像為圖節點
        features, edge_index = self.vig(x)  
        #print(f"🔍 [DEBUG] ViG Output: Features Shape: {features.shape}, Edge Index Shape: {edge_index.shape}")

        # 2️⃣ 確保 edge_index 是 kNN 計算的
        edge_index = knn_graph(features, k=self.knn_neighbors, loop=True)

        # 3️⃣ **正確的 batch 分配**
        batch_size = x.shape[0]  # ✅ 獲取當前批次大小
        num_nodes_per_image = features.shape[0] // batch_size  # ✅ 計算每張圖的節點數
        batch = torch.arange(batch_size, device=features.device).repeat_interleave(num_nodes_per_image)

        # 4️⃣ **多層 GNN 運算**
        for gnn_layer in self.gnn_layers:
            features = F.relu(gnn_layer(features, edge_index))

        # 5️⃣ **Global Pooling，確保 batch 不崩壞**
        features = global_mean_pool(features, batch)  

        #print(f"🔍 [DEBUG] Final Output Shape: {features.shape}")
        return features
