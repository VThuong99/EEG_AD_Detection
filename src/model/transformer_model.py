import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEEG(nn.Module):
    """
    TransformerEEG - A Transformer Model for EEG Data.

    Processing Steps:

    Receive input in the format (batch, num_channels, num_samples).

    Transpose to (batch, num_samples, num_channels) and apply a linear projection to each time step.

    Feed the data into a Transformer Encoder.

    Perform global average pooling over the time dimension and classify via a fully connected layer.
    """
    def __init__(self, num_channels, num_samples, num_classes, d_model=64, nhead=8, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerEEG, self).__init__()
        self.input_proj = nn.Linear(num_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, num_channels, num_samples)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = x.transpose(0, 1)
        # Transformer Encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x
