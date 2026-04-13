import torch
import torch.nn as nn
import torch.nn.functional as F
from yolo_module import YOLO
from gnn_module import HierarchicalGAT


class YOLO_GNN(nn.Module):
    """
    YOLO_GNN: interpretable image classifier.

    Pipeline
    --------
    1. YOLO backbone extracts a 4x4 spatial feature map (16 nodes per image).
       Each node corresponds to a real region of the input image, carrying
       texture / colour / shape information for that patch.

    2. A precomputed 8-connected spatial adjacency graph links neighbouring
       patches.  This mirrors the spatial structure of the image.

    3. YOLO's global-average-pool head produces rough object_logits used
       ONLY to select which class-GNNs to run (and as an auxiliary loss to
       ensure this selection is meaningful).

    4. For each image, the top-k class-GNNs are run.  Samples that share
       the same selected class are batched together for GPU efficiency.

    5. Each class-specific HierarchicalGAT learns:
         - which spatial regions matter for that class (GAT attention weights)
         - how to group regions into parts (SAGPooling)
         - 16 nodes -> ~8 -> ~4 -> 1 graph embedding
       The attention weights make predictions interpretable.

    6. Top-k GNN outputs are averaged -> FC -> final class logits.

    Returns
    -------
    final_output  : [B, num_classes]  - primary classification logits
    object_logits : [B, num_classes]  - YOLO auxiliary logits
    """

    def __init__(self, input_size, num_classes, feature_dim,
                 gnn_hidden_dim, gnn_output_dim, gnn_num_heads,
                 pool_ratio, dropout_rate, top_k):
        super(YOLO_GNN, self).__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.gnn_output_dim = gnn_output_dim
        self.top_k = top_k

        self.yolo = YOLO(input_size, num_classes, feature_dim)

        # One specialised GNN per class
        self.gnns = nn.ModuleList([
            HierarchicalGAT(feature_dim, gnn_hidden_dim, gnn_output_dim,
                            gnn_num_heads, pool_ratio, dropout_rate)
            for _ in range(num_classes)
        ])

        self.fc = nn.Linear(gnn_output_dim, num_classes)

        # Precompute 8-connected adjacency for 4x4 spatial grid (fixed for CIFAR-10)
        self.register_buffer('base_edge_index', self._build_spatial_graph(grid_size=4))

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_spatial_graph(grid_size=4):
        """8-connected adjacency graph for a (grid_size x grid_size) patch grid."""
        edges = []
        n = grid_size
        for i in range(n):
            for j in range(n):
                node = i * n + j
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < n:
                            edges.append([node, ni * n + nj])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _batch_edge_index(self, n_graphs, device):
        """Tile base_edge_index for n_graphs graphs (each with 16 nodes)."""
        E = self.base_edge_index.size(1)
        # offsets: [n_graphs, 1, 1]  *  node count per graph
        offsets = torch.arange(n_graphs, device=device).view(n_graphs, 1, 1) * 16
        # expanded: [n_graphs, 2, E]
        expanded = self.base_edge_index.unsqueeze(0).expand(n_graphs, -1, -1)
        # result:   [2, n_graphs * E]
        return (expanded + offsets).permute(1, 0, 2).reshape(2, -1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        batch_size = x.size(0)

        # Step 1: YOLO feature extraction
        object_logits, node_features = self.yolo(x)
        # object_logits : [B, num_classes]
        # node_features : [B, 16, feature_dim]

        # Step 2: Select top-k class GNNs per image
        _, top_k_indices = torch.topk(object_logits, self.top_k, dim=1)
        # top_k_indices : [B, top_k]

        # Step 3: Run GNNs class-by-class, batching all images that need the same GNN
        gnn_output = torch.zeros(batch_size, self.gnn_output_dim, device=x.device)
        counts     = torch.zeros(batch_size, device=x.device)

        for class_id in range(self.num_classes):
            # Which images selected this class in any top-k slot?
            mask = (top_k_indices == class_id).any(dim=1)  # [B] bool
            n_sel = int(mask.sum())
            if n_sel == 0:
                continue

            # Flatten selected images' nodes: [n_sel * 16, feature_dim]
            x_flat    = node_features[mask].reshape(n_sel * 16, self.feature_dim)
            batch_vec = torch.arange(n_sel, device=x.device).repeat_interleave(16)
            edge_idx  = self._batch_edge_index(n_sel, x.device)

            # Run class-specific GNN -> [n_sel, gnn_output_dim]
            out = self.gnns[class_id](x_flat, edge_idx, batch_vec)

            gnn_output[mask] += out
            counts[mask]     += 1

        # Average across top-k contributions
        combined     = gnn_output / counts.clamp(min=1).unsqueeze(1)  # [B, gnn_output_dim]
        final_output = self.fc(combined)                               # [B, num_classes]

        return final_output, object_logits
