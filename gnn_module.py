import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGPooling, global_mean_pool


class HierarchicalGAT(nn.Module):
    """
    Hierarchical Graph Attention Network with SAGPooling.

    Builds a three-level understanding of each image:
        Level 1 (16 nodes): raw spatial patches - texture, colour, local shape
        Level 2 (~8 nodes): emerging part groupings after SAGPool
        Level 3 (~4 nodes): high-level parts (e.g. head, body, limbs)
        Global pool -> single graph embedding

    GAT edges learn WHICH regions attend to each other (analogous to
    Transformer self-attention), making the model interpretable: the
    attention weights show which spatial relationships matter for a class.

    Each class has its own HierarchicalGAT instance so it can specialise
    in the part-relationships that define that class.
    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_heads=4, pool_ratio=0.5, dropout_rate=0.3):
        super().__init__()

        # --- Level 1: input_dim -> hidden_dim * num_heads ---
        self.gat1 = GATConv(
            input_dim, hidden_dim,
            heads=num_heads, concat=True, dropout=dropout_rate
        )
        self.pool1 = SAGPooling(hidden_dim * num_heads, ratio=pool_ratio)

        # --- Level 2: hidden_dim*num_heads -> hidden_dim * num_heads ---
        self.gat2 = GATConv(
            hidden_dim * num_heads, hidden_dim,
            heads=num_heads, concat=True, dropout=dropout_rate
        )
        self.pool2 = SAGPooling(hidden_dim * num_heads, ratio=pool_ratio)

        # --- Level 3: -> output_dim (single head, no concat) ---
        self.gat3 = GATConv(
            hidden_dim * num_heads, output_dim,
            heads=1, concat=False, dropout=dropout_rate
        )

    def forward(self, x, edge_index, batch):
        # Level 1
        x = F.elu(self.gat1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)

        # Level 2
        x = F.elu(self.gat2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)

        # Level 3
        x = self.gat3(x, edge_index)

        # Graph-level embedding
        return global_mean_pool(x, batch)   # [n_graphs, output_dim]
