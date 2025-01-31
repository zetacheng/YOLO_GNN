# yolo_module.py
import torch
import torch.nn as nn

class YOLO(nn.Module):
    def __init__(self, input_size, num_classes, num_components=5):
        super(YOLO, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_components = num_components

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes + num_components * 64)  # 64-dim feature for each component
        )

    def forward(self, x):
        features = self.conv_layers(x)
        x = features.view(features.size(0), -1)
        x = self.fc_layers(x)
        
        class_logits = x[:, :self.num_classes]
        component_features = x[:, self.num_classes:].view(-1, self.num_components, 64)
        
        return class_logits, component_features, features
