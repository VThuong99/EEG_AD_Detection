import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel attention.
    Input: (batch, time_steps, channels)
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, time_steps, channels)
        # Global average pooling over the time dimension
        w = x.mean(dim=1)           # (batch, channels)
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))  # (batch, channels)
        w = w.unsqueeze(1)          # (batch, 1, channels)
        return x * w  # Scale the input by the attention weights

class GCNLayer(nn.Module):
    """
    A simple Graph Convolutional Network (GCN) layer.
    Input:
      - x: (batch, num_nodes, features)
      - adj: adjacency matrix of shape (num_nodes, num_nodes)
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        # x: (batch, num_nodes, features)
        x = torch.matmul(adj, x)  # Aggregate information from neighboring nodes
        x = self.linear(x)
        x = self.relu(x)
        return x
