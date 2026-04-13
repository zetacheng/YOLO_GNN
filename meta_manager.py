class Meta:
    def __init__(self):
        self.yolo_input_size = (32, 32)
        self.yolo_num_classes = 10
        self.feature_dim = 64         # node feature dim (per spatial location)
        self.gnn_hidden_dim = 32      # hidden dim per GAT attention head
        self.gnn_num_heads = 4        # number of GAT attention heads
        self.gnn_output_dim = 64      # final GNN output dim
        self.pool_ratio = 0.5         # SAGPool keep ratio: 16 nodes -> 8 -> 4
        self.learning_rate = 0.002
        self.batch_size = 256
        self.num_epochs = 50
        self.top_k = 3                # top-k class GNNs to run per image
        self.aux_loss_weight = 0.3    # weight for YOLO auxiliary classification loss
        self.early_stopping_patience = 10
        self.dropout_rate = 0.3

    def enquireMetaValue(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Parameter '{key}' not found in Params.")
