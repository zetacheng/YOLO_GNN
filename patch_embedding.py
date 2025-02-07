import torch
import torch.nn as nn
from dim_manager import Dim_Manager # ✅ 引入維度管理

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3):  # ✅ 調整 `patch_size`
        super(PatchEmbedding, self).__init__()
        self.dim_manager = Dim_Manager()
        embed_dim = self.dim_manager.enquireDimValue("patch_feature_dim")
        patch_size = self.dim_manager.enquireDimValue("Patch_Embedding")
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # Shape: (batch_size, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).contiguous()  # Shape: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)
        return x


