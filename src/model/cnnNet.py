import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNNet(nn.Module):
    """
    Input shape: (batch, num_channels, num_samples)
    Output: logits cho num_classes
    """
    def __init__(self, num_channels, num_samples, num_classes):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32 * (num_samples // 2), 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, num_channels, num_samples)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (batch, 16, num_samples//2)
        x = F.relu(self.bn2(self.conv2(x)))               # (batch, 32, num_samples//2)
        x = x.view(x.size(0), -1)                         # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
