import torch
import torch.nn as nn

class LargeConvFormer(nn.Module):
    """ ConvFormer model for experimenting with STFT/CWT features."""
    def __init__(self, num_channels, num_freqs, num_times, num_classes, 
                 embed_dim=128, num_heads=8, num_layers=2, dropout=0.1):
        super(LargeConvFormer, self).__init__()
        
        # Conv part
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_channels, num_freqs, num_times)
            cnn_output = self.cnn(dummy_input)
            _, _, h, w = cnn_output.shape
            self.seq_len = h * w
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification Head
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # x: (batch, num_channels, num_freqs, num_times)
        x = self.cnn(x)  # (batch, embed_dim, h, w)
        x = x.flatten(2)  # (batch, embed_dim, seq_len)
        x = x.permute(2, 0, 1)  # (seq_len, batch, embed_dim)
        x = self.transformer(x)  # (seq_len, batch, embed_dim)
        x = x.mean(dim=0)  # (batch, embed_dim)
        x = self.fc(x)  # (batch, num_classes)
        return x
