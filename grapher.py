import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, SAGEConv
from torch_geometric.utils import to_undirected

class GrapherModule(nn.Module):
    """Graph convolution module using GraphSAGE"""
    def __init__(self, in_dim, out_dim):
        super(GrapherModule, self).__init__()
        self.sage1 = SAGEConv(in_dim, out_dim)  # ✅ Use GraphSAGE
        self.sage2 = SAGEConv(out_dim, out_dim)  # ✅ Use GraphSAGE

    def forward(self, x, edge_index=None):
        # ✅ 確保 `edge_index` 正確生成
        if edge_index is None:
            batch_size, num_patches, feature_dim = x.shape
            x = x.reshape(batch_size * num_patches, feature_dim)  # ✅ 使用 reshape 避免 view 的錯誤  # Flatten patches
            batch = torch.arange(batch_size, device=x.device).repeat_interleave(num_patches)

            edge_index = knn_graph(x, k=5, loop=True, batch=batch)

        edge_index = edge_index.long()  # ✅ 確保格式正確
        edge_index = to_undirected(edge_index)  # ✅ 讓 Edge 變成無向圖

        x = self.sage1(x, edge_index)  # ✅ Apply GraphSAGE
        x = F.relu(x)
        x = self.sage2(x, edge_index)

        #print(f"🔍 [DEBUG] GrapherModule Output: Features Shape: {x.shape}, Edge Index Shape: {edge_index.shape}")

        return x, edge_index  # ✅ 只返回兩個值

class FFNModule(nn.Module):
    """Graph-based Feature Transformation using GraphSAGE"""
    def __init__(self, in_dim, hidden_dim):
        super(FFNModule, self).__init__()
        self.sage1 = SAGEConv(in_dim, hidden_dim)  # ✅ GraphSAGE 替代 FC
        self.sage2 = SAGEConv(hidden_dim, in_dim)  # ✅ GraphSAGE 替代 FC

    def forward(self, x, edge_index):
        x = F.relu(self.sage1(x, edge_index))  # ✅ Graph-based feature transformation
        x = self.sage2(x, edge_index)
        return x
