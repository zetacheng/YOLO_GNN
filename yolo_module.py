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
    def __init__(self, input_size, num_classes, feature_dim=64):
        super(YOLO, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.features = nn.Sequential(
            ConvBNSiLU(3, 32, 3, padding=1),
            ConvBNSiLU(32, 64, 3, stride=2, padding=1),
            CSPLayer(64, 64),
            ConvBNSiLU(64, 128, 3, stride=2, padding=1),
            CSPLayer(128, 128),
            ConvBNSiLU(128, 256, 3, stride=2, padding=1),
            CSPLayer(256, 256),
            SPPF(256, 256)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.object_classifier = nn.Linear(256, num_classes)
        self.feature_extractor = nn.Linear(256, feature_dim)

    def forward(self, x):
        features = self.features(x)
        pooled = self.avgpool(features)
        pooled = torch.flatten(pooled, 1)
        object_logits = self.object_classifier(pooled)
        extracted_features = self.feature_extractor(pooled)
        return object_logits, extracted_features
