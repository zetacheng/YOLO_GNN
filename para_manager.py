class Params:
    def __init__(self):
        self.yolo_input_size = (32, 32)
        self.yolo_num_classes = 10
        self.gnn_input_dim = 64
        self.gnn_hidden_dim = 64
        self.gnn_output_dim = 64
        self.gnn_num_layers = 2
        self.learning_rate = 0.0001
        self.batch_size = 32
        self.num_epochs = 50
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.mixup_alpha = 0.2
        self.cutmix_alpha = 1.0

    def enquireMetaValue(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Parameter '{key}' not found in Params.")

