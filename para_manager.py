class Params:
    def __init__(self):
        self.yolo_input_size = (32, 32)
        self.yolo_num_classes = 10
        self.feature_dim = 64
        self.gnn_hidden_dim = 32
        self.gnn_output_dim = 64
        self.learning_rate = 0.001  # This is now the maximum learning rate
        self.batch_size = 64
        self.num_epochs = 100  # Increased from 50
        self.top_k = 3
        self.knn_neighbors = 5

    def enquireMetaValue(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Parameter '{key}' not found in Params.")
