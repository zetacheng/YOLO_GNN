import torch
import torch.nn as nn
from yolo_module import YOLO
from gnn_module import GNN
from graph_builder import GraphBuilder

class YOLO_GNN(nn.Module):
    def __init__(self, input_size, num_classes, feature_dim, gnn_hidden_dim, gnn_output_dim, top_k, knn_neighbors, use_attention=False):
        super(YOLO_GNN, self).__init__()
        self.yolo = YOLO(input_size, num_classes, feature_dim)
        self.gnns = nn.ModuleList([GNN(feature_dim, gnn_hidden_dim, gnn_output_dim, use_attention=use_attention) for _ in range(num_classes)])
        self.fc = nn.Linear(gnn_output_dim, num_classes)
        self.top_k = top_k
        self.knn_neighbors = knn_neighbors
        self.graph_builder = GraphBuilder()

    def forward(self, x):
        object_logits, features = self.yolo(x)
        _, top_k_indices = torch.topk(object_logits, self.top_k, dim=1)
        
        gnn_outputs = []
        for i in range(x.size(0)):
            batch_features = features[i]
            graph = self.graph_builder.build_graph(batch_features, feature_maps=batch_features)
            
            for idx in top_k_indices[i]:
                gnn = self.gnns[idx]
                gnn_output = gnn(graph.x, graph.edge_index, batch=None, edge_weight=graph.edge_attr)
                gnn_outputs.append(gnn_output)

        combined_output = torch.stack(gnn_outputs, dim=0)
        combined_output = combined_output.view(-1, combined_output.size(-1))
        final_output = self.fc(combined_output)
        final_output = final_output.view(x.size(0), self.top_k, -1)
        final_output = final_output.mean(dim=1)
        
        return final_output
