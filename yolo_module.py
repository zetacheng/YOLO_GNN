import torch
import torch.nn as nn

class YOLO(nn.Module):
    def __init__(self, input_size, num_classes):
        super(YOLO, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
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
            nn.Linear(128, num_classes + 4),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        logits = x[:, :self.num_classes]
        bboxes = x[:, self.num_classes:]
        return logits, bboxes

