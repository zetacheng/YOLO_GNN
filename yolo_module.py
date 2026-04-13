import torch
import torch.nn as nn


class ConvBNSiLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNSiLU(in_channels, out_channels, 1)
        self.conv2 = ConvBNSiLU(in_channels, out_channels, 1)
        self.conv3 = ConvBNSiLU(out_channels, out_channels, 3, padding=1)
        self.conv4 = ConvBNSiLU(out_channels, out_channels, 1)
        self.conv5 = ConvBNSiLU(out_channels * 2, out_channels, 1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.conv3(y2)
        y2 = self.conv4(y2)
        return self.conv5(torch.cat([y1, y2], dim=1))


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNSiLU(in_channels, out_channels, 1)
        self.conv2 = ConvBNSiLU(out_channels * 4, out_channels, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class YOLO(nn.Module):
    """
    YOLO backbone for feature extraction.

    Outputs:
        object_logits:  [B, num_classes]  - rough classification for auxiliary loss
        node_features:  [B, 16, feature_dim] - spatial node features for the GNN
                        Each of the 16 nodes corresponds to one cell in the 4x4
                        spatial grid, preserving spatial layout information.
    """

    def __init__(self, input_size, num_classes, feature_dim=64):
        super(YOLO, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # 32x32 -> 16x16 -> 8x8 -> 4x4  (three stride-2 downsamples)
        self.backbone = nn.Sequential(
            ConvBNSiLU(3, 32, 3, padding=1),             # 32x32
            ConvBNSiLU(32, 64, 3, stride=2, padding=1),  # 16x16
            CSPLayer(64, 64),                             # 16x16
            ConvBNSiLU(64, 128, 3, stride=2, padding=1), # 8x8
            CSPLayer(128, 128),                           # 8x8
            ConvBNSiLU(128, 256, 3, stride=2, padding=1),# 4x4
            CSPLayer(256, 256),                           # 4x4
            SPPF(256, 256),                               # 4x4
        )

        # Rough classification head (auxiliary loss only)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.object_classifier = nn.Linear(256, num_classes)

        # Project each spatial location from 256-dim to feature_dim
        self.node_projector = nn.Linear(256, feature_dim)

    def forward(self, x):
        # Spatial feature map: [B, 256, 4, 4]
        feat_map = self.backbone(x)

        # Auxiliary classification via global average pooling
        pooled = torch.flatten(self.avgpool(feat_map), 1)   # [B, 256]
        object_logits = self.object_classifier(pooled)       # [B, num_classes]

        # Spatial node features: flatten 4x4 grid -> 16 nodes
        B, C, H, W = feat_map.shape                          # [B, 256, 4, 4]
        spatial = feat_map.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, 16, 256]
        node_features = self.node_projector(spatial)          # [B, 16, feature_dim]

        return object_logits, node_features
