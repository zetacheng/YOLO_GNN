import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, SAGEConv, TopKPooling
from vig_module import ViG
from dim_manager import Dim_Manager # ✅ 引入維度管理

class ViG_GNN(nn.Module):
    def __init__(self, num_classes):
        super(ViG_GNN, self).__init__()
        self.dim_manager = Dim_Manager()  # ✅ 使用統一維度管理
        ViG_feature_dim = self.dim_manager.enquireDimValue("ViG_feature_dim")
        gnn1_input_dim = self.dim_manager.enquireDimValue("GNN1_Input")
        gnn1_output_dim = self.dim_manager.enquireDimValue("GNN1_Output")
#        gnn2_input_dim = self.dim_manager.enquireDimValue("GNN2_Input")
#        gnn2_output_dim = self.dim_manager.enquireDimValue("GNN2_Output")
        self.knn_neighbors = self.dim_manager.enquireDimValue("knn_neighbors")
        self.batch_size = self.dim_manager.meta.batch_size  # ✅ **正確獲取 batch_size**
        
        # **設定 Pooling Ratio**
        self.pool_ratio = 0.75  # ✅ 調整 Pooling 比例，確保節點數減少時仍有足夠訊息

         # ✅ 確保 ViG 輸出維度與 GNN1 的輸入維度一致
        assert ViG_feature_dim == gnn1_input_dim, "❌ ViG 輸出維度與 GNN1 不匹配！"       

        self.vig = ViG() # ✅ ViG 局部學習
        self.gnn1 = SAGEConv(gnn1_input_dim, gnn1_output_dim)  # ✅ Pooling 1 前的 GNN
        self.pool1 = TopKPooling(gnn1_output_dim, ratio=self.pool_ratio)  # ✅ 第一次 Pooling

#        self.gnn2 = SAGEConv(gnn2_input_dim, gnn2_output_dim)  # ✅ Pooling 2 前的 GNN
#        self.pool2 = TopKPooling(gnn2_output_dim, ratio=self.pool_ratio)  # ✅ 第二次 Pooling
        
        # ✅ 初始化 `final_gnn`，稍後可能會動態調整
#        self.final_gnn = SAGEConv(gnn2_output_dim, num_classes)
        self.final_gnn = SAGEConv(gnn1_output_dim, num_classes)

    def forward(self, x):
        # 1️⃣ ViG 輸出局部圖結構
        features, edge_index = self.vig(x)
        #print(f"🔍 [DEBUG] ViG Output: Features Shape: {features.shape}, Edge Index Shape: {edge_index.shape}")

        # 2️⃣ GNN1 提取特徵
        features = self.gnn1(features, edge_index)  # ✅ 先用 GNN 提取特徵
        # 3️⃣ 重新計算 edge_index（確保與 GNN1 特徵對應）
        edge_index = knn_graph(features, k=min(self.knn_neighbors, features.shape[0] - 1), loop=True)

        # 4️⃣ Top-K Pooling（第一層）
        features, edge_index, _, batch, _, _ = self.pool1(features, edge_index)
        batch_size_1 = batch.unique().size(0)  # ✅ **正確計算 batch_size**
        #print(f"🔍 [DEBUG] After Pooling 1: Features Shape: {features.shape}, Edge Index Shape: {edge_index.shape}")
        #print(f"🔍 [DEBUG] Batch Size 1: {batch_size_1}")
       # **當 batch_size_1 太小時，直接轉為全連接**
#        if batch_size_1 < self.knn_neighbors:
#            print(f"⚠️ Warning: batch_size_1 ({batch_size_1}) 太小，改為全連接")
#            edge_index = torch.combinations(torch.arange(features.shape[0]), r=2).T.to(features.device)
#        else:
        edge_index = knn_graph(features, k=min(self.knn_neighbors, features.shape[0] - 1), loop=True)

#        # 4️⃣ Top-K Pooling（第二層）
#        features = self.gnn2(features, edge_index)  # ✅ 再用 GNN 提取特徵
#        features, edge_index, _, batch, _, _ = self.pool2(features, edge_index)
#        batch_size_2 = batch.unique().size(0)  # ✅ **正確計算 batch_size**
#        print(f"🔍 [DEBUG] After Pooling 2: Features Shape: {features.shape}, Edge Index Shape: {edge_index.shape}")

#       # **當 batch_size_2 太小時，直接轉為全連接**
#        if batch_size_2 < self.knn_neighbors:
#            print(f"⚠️ Warning: batch_size_2 ({batch_size_2}) 太小，改為全連接")
#            edge_index = torch.combinations(torch.arange(features.shape[0]), r=2).T.to(features.device)
#        else:
#        edge_index = knn_graph(features, k=min(self.knn_neighbors, features.shape[0] - 1), loop=True)

        # 5️⃣ 最終 GraphSAGE 進行全連接分類
        # ✅ **動態調整 final GNN**
        final_gnn_input_dim = features.shape[1]  # **取最後 GNN 的輸出維度**
        if self.final_gnn.in_channels != final_gnn_input_dim:
            print(f"⚠️ Warning: Adjusting final GNN input_dim to {final_gnn_input_dim}")
            self.final_gnn = SAGEConv(final_gnn_input_dim, self.dim_manager.meta.vig_num_classes).to(features.device)

        # ✅ **確保 batch_size 與 target batch_size 一致**
        if features.shape[0] != self.batch_size:
            print(f"⚠️ Warning: Adjusting final batch_size from {features.shape[0]} to {self.batch_size}")
            features = features[:self.batch_size]

        # 5️⃣ 最終 GraphSAGE 進行全連接分類
        final_output = self.final_gnn(features, edge_index)
        print(f"🔍 [DEBUG] Final Output Shape: {final_output.shape}")

        return final_output
