class Meta:
    def __init__(self):
        self.vig_input_size = (32, 32)
        self.vig_num_classes = 10
        self.patch_size = 4  # ✅ 確保 patch_size 與 PatchEmbedding 一致
        self.feature_dim = 64  # ✅ 確保 feature_dim 與 patch_size 相容
        self.gnn_hidden_dim = 32
        self.gnn_output_dim = 64
        self.learning_rate = 0.001  # ✅ 減小學習率，避免 batch_size 影響
        self.batch_size = 64
        self.num_epochs = 50
        self.knn_neighbors = 5  # ✅ 增加 KNN 連接數，避免圖過度稀疏
        self.num_gnn_layers = 10  # ✅ Specify how many GNN layers you want
        self.early_stopping_patience = 10
        self.dropout_rate = 0.3

    def enquireMetaValue(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Parameter '{key}' not found in Params.")
