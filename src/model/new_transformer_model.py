import torch
import torch.nn as nn

# 1. Spatial CNN + Transformer
class SpatialCNNTransformer(nn.Module):
    def __init__(self, num_channels=19, num_bands=5, num_classes=2, d_model=64, nhead=8, num_layers=2):
        super().__init__()
        self.conv = nn.Conv2d(1, d_model, kernel_size=(3,3), padding=1)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead), num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.pos_encoder = nn.Parameter(torch.randn(1, num_channels * num_bands, d_model))

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, 19, 5)
        x = self.conv(x)    # (batch, d_model, 19, 5)
        x = x.flatten(2).transpose(1,2)  # (batch, 19*5, d_model)
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)

# 3. Channel-wise MLP + Transformer
class ChannelMLPTransformer(nn.Module):
    def __init__(self, num_channels=19, num_bands=5, num_classes=2, d_model=64, nhead=8, num_layers=2):
        super().__init__()
        self.mlp = nn.Linear(num_bands, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead), num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.pos_encoder = nn.Parameter(torch.randn(1, num_channels, d_model))

    def forward(self, x):
        x = self.mlp(x)  # (batch, 19, d_model)
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)

# 5. Frequency-band-wise Transformer
class BandTransformer(nn.Module):
    def __init__(self, num_channels=19, num_bands=5, num_classes=2, d_model=64, nhead=8, num_layers=2):
        super().__init__()
        self.band_embed = nn.Linear(num_channels, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead), num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.pos_encoder = nn.Parameter(torch.randn(1, num_bands, d_model))

    def forward(self, x):
        x = x.transpose(1,2)  # (batch, 5, 19)
        x = self.band_embed(x)  # (batch, 5, d_model)
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)