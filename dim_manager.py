from meta_manager import Meta

class Dim_Manager:
    """統一管理 ViG-GNN 所有維度計算"""

    def __init__(self):
        self.meta = Meta()  # ✅ 獲取 meta_manager 設定

        # ✅ 直接從 meta 獲取
        self.base_feature_dim = self.meta.feature_dim  # ViG 輸出
        self.base_gnn_hidden_dim = self.meta.gnn_hidden_dim
        self.base_gnn_output_dim = self.meta.gnn_output_dim
        self.base_patch_size = self.meta.vig_input_size[0] // 4  # ✅ 動態計算合理的 patch_size
        self.base_knn_neighbors = self.meta.knn_neighbors

        # ✅ 記錄所有模組的角色
        self.role_dims = {
            # ViG
            "ViG_feature_dim": self.base_feature_dim,
            
            "ViG_Output": self.base_feature_dim,  # ViG 的輸出
            "GNN1_Input": self.base_feature_dim,  # GNN1 應該與 ViG 輸出對應
            
#            "GNN1_Output": self.base_gnn_hidden_dim,  # GNN1 隱藏層
#            "GNN2_Input": self.base_gnn_hidden_dim,  # GNN2 的輸入應該等於 GNN1 的輸出

            "GNN1_Output": self.base_gnn_output_dim,  # GNN1 最終輸出
            
            "Final_GNN_Output": None,  # 🚀 **初始化為 None，之後動態設置**

            "knn_neighbors": self.base_knn_neighbors,
            # PatchEmbedding
            "patch_feature_dim": self.base_feature_dim,
            "Patch_Embedding": self.base_patch_size,
            "TopKPooling": self.base_gnn_output_dim,
        }

    def enquireDimValue(self, role):
        """根據角色查詢對應的維度"""
        if role in self.role_dims:
            return self.role_dims[role]
        else:
            raise KeyError(f"❌ Dimension role '{role}' not found in Dim_Manager.")

    def registerRole(self, role, dim_value):
        """允許模組動態註冊自己的維度需求"""
        self.role_dims[role] = dim_value
