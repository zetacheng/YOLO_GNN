import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_attention=False):
        super(GNN, self).__init__()
        self.use_attention = use_attention

        if use_attention:
            # Graph Attention Layers
            self.conv1 = GATConv(input_dim, hidden_dim, heads=2, concat=True)
            self.conv2 = GATConv(hidden_dim * 2, hidden_dim, heads=2, concat=True)
            self.conv3 = GATConv(hidden_dim * 2, output_dim, heads=1, concat=False)
        else:
            # Graph Convolutional Layers
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch, edge_weight=None):
        """
        Forward pass for the GNN.
        :param x: Node features
        :param edge_index: Graph edge indices
        :param batch: Batch indices for pooling
        :param edge_weight: Optional edge weights for graph connections
        :return: Pooled graph embeddings
        """
        if self.use_attention:
            # Attention-based convolution
            x = F.elu(self.conv1(x, edge_index))
            x = F.elu(self.conv2(x, edge_index))
            x = self.conv3(x, edge_index)
        else:
            # Standard GCN convolution
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = F.relu(self.conv2(x, edge_index, edge_weight))
            x = self.conv3(x, edge_index, edge_weight)

        # Global mean pooling
        x = global_mean_pool(x, batch)
        return x
