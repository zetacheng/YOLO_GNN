import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from meta_manager import Meta

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)  # ✅ Use SAGE instead of GCN
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, output_dim)
        dropout_rate=Meta().enquireMetaValue("dropout_rate")
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)  # Add dropout after activation
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)  # ✅ Ensure pooling does not collapse batch
        return x
