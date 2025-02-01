import torch
import torch.nn as nn

class YOLO(nn.Module):
    def __init__(self, input_size, num_classes, feature_dim=64):
        super(YOLO, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.features = nn.Sequential(
            self._make_layer(3, 16),
            self._make_layer(16, 32),
            self._make_layer(32, 64),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.object_classifier = nn.Linear(64, num_classes)
        self.feature_extractor = nn.Linear(64, feature_dim)

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        features = self.features(x)
        pooled = self.avgpool(features)
        pooled = torch.flatten(pooled, 1)
        object_logits = self.object_classifier(pooled)
        extracted_features = self.feature_extractor(pooled)
        return object_logits, extracted_features
