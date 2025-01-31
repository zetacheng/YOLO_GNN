import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class EnhancedHierarchicalGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes):
        super(EnhancedHierarchicalGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.skip_linear = torch.nn.Linear(input_dim, output_dim)
        self.fc = torch.nn.Linear(output_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = self.conv3(x2, edge_index) + self.skip_linear(x)
        pooled = global_mean_pool(x3, batch)
        out = self.fc(pooled)
        return out
