import torch
import torch.nn as nn
from src.model.layer import SEBlock, GCNLayer

class TransformerEEG(nn.Module):
    """
    TransformerEEG - A Transformer Model for EEG Data.
    
    Processing Steps:
      - Input: (batch, num_channels, num_samples)
      - Transpose to (batch, num_samples, num_channels) and apply a linear projection for each time step.
      - Pass through a Transformer Encoder.
      - Apply global average pooling over the time dimension and classify via a fully connected layer.
    """
    def __init__(self, num_channels, num_samples, num_classes, d_model=64, nhead=8,
                 num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerEEG, self).__init__()
        self.input_proj = nn.Linear(num_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, num_channels, num_samples)
        x = x.transpose(1, 2)   # (batch, num_samples, num_channels)
        x = self.input_proj(x)  # (batch, num_samples, d_model)
        x = x.transpose(0, 1)   # (num_samples, batch, d_model)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)       # Global average pooling over time dimension
        x = self.fc(x)
        return x

class CAformer(nn.Module):
    """
    TransformerEEG with Channel Attention using SE Block.
    """
    def __init__(self, num_channels, num_samples, num_classes, d_model=64, nhead=8,
                 num_layers=2, dim_feedforward=128, dropout=0.1):
        super(CAformer, self).__init__()
        self.input_proj = nn.Linear(num_channels, d_model)
        self.se_block = SEBlock(d_model, reduction=8)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, num_channels, num_samples)
        x = x.transpose(1, 2)  # (batch, num_samples, num_channels)
        x = self.input_proj(x)  # (batch, num_samples, d_model)
        x = self.se_block(x)  # Apply SE block for channel attention
        x = x.transpose(0, 1)  # (num_samples, batch, d_model)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x

class GCNformer(nn.Module):
    """
    TransformerEEG with a GNN branch.
    Combines:
      - Transformer branch for temporal feature extraction.
      - GCN branch for learning channel relationships using an adjacency matrix.
    
    Requirement: Provide an adjacency matrix 'adj' with shape (num_channels, num_channels).
    """
    def __init__(self, num_channels, num_samples, num_classes, adj,
                 d_model=64, nhead=8, num_layers=2, dim_feedforward=128, dropout=0.1,
                 gcn_hidden=32, gcn_out=32):
        super(GCNformer, self).__init__()
        self.adj = adj  # Adjacency matrix for channels
        
        # Transformer branch (temporal)
        self.input_proj = nn.Linear(num_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # GCN branch (channel)
        self.gcn1 = GCNLayer(in_features=num_samples, out_features=gcn_hidden)
        self.gcn2 = GCNLayer(in_features=gcn_hidden, out_features=gcn_out)
        
        # Combine both branches
        self.fc = nn.Linear(d_model + gcn_out, num_classes)

    def forward(self, x):
        # x: (batch, num_channels, num_samples)
        # Transformer branch
        x_time = x.transpose(1, 2)      # (batch, num_samples, num_channels)
        x_time = self.input_proj(x_time)  # (batch, num_samples, d_model)
        x_time = x_time.transpose(0, 1)   # (num_samples, batch, d_model)
        x_time = self.transformer_encoder(x_time)
        x_time = x_time.mean(dim=0)       # (batch, d_model)
        
        # GCN branch
        adj = self.adj.to(x.device)
        x_gcn = self.gcn1(x, adj)    # (batch, num_channels, gcn_hidden)
        x_gcn = self.gcn2(x_gcn, adj)  # (batch, num_channels, gcn_out)
        x_gcn = x_gcn.mean(dim=1)         # Global average pooling over channels
        
        # Combine both branches
        x_combined = torch.cat([x_time, x_gcn], dim=1)  # (batch, d_model + gcn_out)
        out = self.fc(x_combined)
        return out

class Conv1dformer(nn.Module):
    """
    TransformerEEG with a Convolution module on the channel dimension.
    Uses Conv1d to capture relationships between channels before the Transformer processes temporal features.
    """
    def __init__(self, num_channels, num_samples, num_classes, d_model=64, nhead=8,
                 num_layers=2, dim_feedforward=128, dropout=0.1, kernel_size=3):
        super(Conv1dformer, self).__init__()
        self.channel_conv = nn.Conv1d(in_channels=num_channels, out_channels=d_model,
                                      kernel_size=kernel_size, padding=kernel_size//2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, num_channels, num_samples)
        x_conv = self.channel_conv(x)   # (batch, d_model, num_samples)
        x_conv = x_conv.transpose(1, 2)   # (batch, num_samples, d_model)
        x_conv = x_conv.transpose(0, 1)   # (num_samples, batch, d_model)
        x_tr = self.transformer_encoder(x_conv)
        x_tr = x_tr.mean(dim=0)           # (batch, d_model)
        out = self.fc(x_tr)
        return out

class DAformer(nn.Module):
    """
    Dual-Axis Transformer for EEG Data.
    Two branches:
      - Time branch: processes temporal information.
      - Channel branch: extracts channel features after pooling over time.
    The outputs of both branches are concatenated for classification.
    """
    def __init__(self, num_channels, num_samples, num_classes, d_model=64, nhead=8,
                 num_layers=2, dim_feedforward=128, dropout=0.1):
        super(DAformer, self).__init__()
        # Time branch
        self.input_proj = nn.Linear(num_channels, d_model)
        encoder_layer_time = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                        dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder_time = nn.TransformerEncoder(encoder_layer_time, num_layers=num_layers)
        
        # Channel branch: pooling over time and projecting to d_model
        self.channel_proj = nn.Linear(1, d_model)
        encoder_layer_channel = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                           dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder_channel = nn.TransformerEncoder(encoder_layer_channel, num_layers=num_layers)
        
        self.fc = nn.Linear(2 * d_model, num_classes)

    def forward(self, x):
        # x: (batch, num_channels, num_samples)
        # Time branch
        x_time = x.transpose(1, 2)         # (batch, num_samples, num_channels)
        x_time = self.input_proj(x_time)     # (batch, num_samples, d_model)
        x_time = x_time.transpose(0, 1)      # (num_samples, batch, d_model)
        x_time = self.transformer_encoder_time(x_time)
        x_time = x_time.mean(dim=0)          # (batch, d_model)
        
        # Channel branch
        x_channel = x.mean(dim=2, keepdim=True)  # (batch, num_channels, 1)
        x_channel = self.channel_proj(x_channel) # (batch, num_channels, d_model)
        x_channel = x_channel.transpose(0, 1)      # (num_channels, batch, d_model)
        x_channel = self.transformer_encoder_channel(x_channel)
        x_channel = x_channel.mean(dim=0)          # (batch, d_model)
        
        # Combine both branches
        x_combined = torch.cat([x_time, x_channel], dim=1)  # (batch, 2*d_model)
        out = self.fc(x_combined)
        return out
