import torch
import torch.nn as nn
import torch.nn.functional as F
from yolo_module import YOLO
from gnn_module import GNN
from torch_geometric.nn import knn_graph, GCNConv

class YOLO_GNN(nn.Module):
    def __init__(self, input_size, num_classes, feature_dim, gnn_hidden_dim, gnn_output_dim, top_k, knn_neighbors):
        super(YOLO_GNN, self).__init__()
        self.yolo = YOLO(input_size, num_classes, feature_dim)
        self.gnns = nn.ModuleList([GNN(feature_dim, gnn_hidden_dim, gnn_output_dim) for _ in range(num_classes)])
        self.final_gnn = GCNConv(gnn_output_dim, num_classes)  # GNN replaces FC
        self.top_k = top_k
        self.knn_neighbors = knn_neighbors
        self.feature_dim = feature_dim

    def break_down_features(self, features):
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        full = features
        split_size = self.feature_dim // 4
        object_level = features[:, :split_size]
        component_level = features[:, split_size:2*split_size]
        detail_level = features[:, 2*split_size:3*split_size]
        remaining = features[:, 3*split_size:]
        
        return full, object_level, component_level, detail_level, remaining

    def build_hierarchical_graph(self, features):
        full, obj, comp, detail, remaining = self.break_down_features(features)
        
        max_size = max(full.size(1), obj.size(1), comp.size(1), detail.size(1), remaining.size(1))
        full = F.pad(full, (0, max_size - full.size(1)))
        obj = F.pad(obj, (0, max_size - obj.size(1)))
        comp = F.pad(comp, (0, max_size - comp.size(1)))
        detail = F.pad(detail, (0, max_size - detail.size(1)))
        remaining = F.pad(remaining, (0, max_size - remaining.size(1)))
        
        x = torch.cat([full, obj, comp, detail, remaining], dim=0)
        
        edge_index = knn_graph(x, k=self.knn_neighbors, loop=True)
        
        return x, edge_index

    def forward(self, x):
        object_logits, features = self.yolo(x)
        
        _, top_k_indices = torch.topk(object_logits, self.top_k, dim=1)
        
        gnn_outputs = []
        for i in range(x.size(0)):
            batch_features = features[i]
            x_graph, edge_index = self.build_hierarchical_graph(batch_features)
            batch = torch.zeros(x_graph.size(0), dtype=torch.long, device=x.device)
            
            for idx in top_k_indices[i]:
                gnn = self.gnns[idx]
                gnn_output = gnn(x_graph, edge_index, batch)
                gnn_outputs.append(gnn_output)

        combined_output = torch.stack(gnn_outputs, dim=0)
        
        combined_output = combined_output.view(-1, combined_output.size(-1))
        
        # Apply final GNN classifier instead of FC layer
        final_output = self.final_gnn(combined_output, edge_index)
        final_output = final_output.view(x.size(0), self.top_k, -1)
        final_output = final_output.mean(dim=1)
        
        return final_output
