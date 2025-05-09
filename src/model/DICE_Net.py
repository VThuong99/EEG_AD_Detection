import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class DepthwiseConv2D(nn.Module):
    """
    Depthwise 2D convolution:
      - Input shape: (batch, T, B, C)
      - Applies a convolution over (T, B) for each channel independently.
      - Uses kernel_size=(5,5), stride=1, no padding.
      - groups=C for each channel independently.
      
      With T=30, B=5, kernel=(5,5) → Output shape: (batch, T'=26, B'=1, C)
    """
    def __init__(self, channels, kernel_size=(5,5)):
        super().__init__()
        self.channels = channels
        self.depthwise = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=(1,1),
            padding=0,
            groups=channels,
            bias=True
        )
        self.act = nn.GELU()

    def forward(self, x):
        # x: (batch, T, B, C)
        # Permute ot fed into Conv2d: (batch, C, T, B)
        x = x.permute(0, 3, 1, 2)
        x = self.depthwise(x)  # (batch, C, T', B')
        x = self.act(x)
        # Permute back: (batch, T', B', C)
        x = x.permute(0, 2, 3, 1)
        return x

class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding.
    Input shape: (batch, seq_len, d_model)
    """
    def __init__(self, max_len, d_model):
        super().__init__()
        # self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.06)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        div_len = (d_model + 1) // 2
        pe[:, 0::2] = torch.sin(position * div_term[:div_len])
        pe[:, 1::2] = torch.cos(position * div_term[:div_len]) 
 
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        # return x + self.pos_emb[:, :seq_len, :]
        return x + self.pe[:, :seq_len, :]

class ClassToken(nn.Module):
    """
    Adds a learnable CLS token at the beginning of the sequence.
    """
    def __init__(self, d_model):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size = x.size(0)
        cls_token = self.cls.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        return torch.cat((cls_token, x), dim=1)          # (batch, seq_len+1, d_model)

class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder block use nn.TransformerEncoder.
    """
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Input: (batch, seq_len, d_model)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)

class FFNClassifier(nn.Module):
    """
    Feed-Forward Network.
    """
    def __init__(self, input_dim=64, hidden_dim=24, num_classes=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

##########################################
# DICENet with Single Branch (RBP Only)
##########################################

class DICENetSingleBranchRBP(nn.Module):
    """
    DICE-Net model for automatic dementia detection using only RBP features.
    
    Input: (batch, T, B, C)
      - T: number of one-second segments (e.g., 30)
      - B: number of frequency bands (e.g., 5)
      - C: number of EEG channels (e.g., 19)
      
    Processing:
      1. Depthwise convolution over (T, B) → (T-4, B-4, C) = (26, 1, C) if T=30, B=5.
      2. Flatten from (26, 1, C) to (26, C).
      3. Add positional embedding for token series (length=26).
      4. Insert CLS token → (27, C).
      5. Transformer encoder → (27, C).
      6. Take CLS token and feed into FFN for classify.
    """
    def __init__(self, T=30, B=5, C=19, kernel_size=(5, 5), d_model=19, nhead=1, num_layers=2,
                 dim_feedforward=64, dropout=0.1, num_classes=2):
        super().__init__()
        self.T = T
        self.B = B
        self.C = C
        self.d_model = d_model
        
        # 1) Depthwise convolution
        self.depthwise_conv = DepthwiseConv2D(channels=C, kernel_size=kernel_size)
        
        # 2) Flatten from (T-4, B-4, C) to (T-4, C)
        # T=30, B=5, kernel=(5,5), no padding => T'=30-5+1=26, B'=5-5+1=1.
        # After Flatten: (batch, 26, C).
        
        # 3) Positional embedding
        # Sequence length = 26, then add CLS -> 27
        self.pos_emb = PositionalEncoding(max_len=50, d_model=d_model)
        
        # 4) CLS token
        self.cls_token = ClassToken(d_model)

        self.pre_norm = nn.LayerNorm(d_model)
        
        # 5) Transformer encoder
        self.transformer = TransformerEncoderBlock(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(d_model)
        
        # 6) FFN classifier: input is CLS token (d_model)
        self.ffn = FFNClassifier(input_dim=d_model, hidden_dim=24, num_classes=num_classes, dropout=dropout)

    def forward(self, x):
        """
        x: (batch, T, B, C)
           Với T=30, B=5, C=19.
        """
        batch_size = x.size(0)
        # 1) Depthwise convolution
        x = self.depthwise_conv(x)  # (batch, T', B', C) => (batch, 26, 1, C)
        
        # 2) Flatten: convert (batch, 26, 1, C) to (batch, 26, C)
        x = x.view(batch_size, -1, x.size(-1))
        
        # 3) Add positional embedding: (batch, 26, d_model)
        x = self.pos_emb(x)
        
        # 4) Add CLS token: (batch, 27, d_model)
        x = self.cls_token(x)

        x = self.pre_norm(x)
        
        # 5) Transformer encoder
        x = self.transformer(x)  # (batch, 27, d_model)
        x = self.norm(x)
        
        # 6) Token CLS to FFN
        cls_token = x[:, 0, :]   # (batch, d_model)
        logits = self.ffn(cls_token)
        return logits

##########################################
# Testing the DICENetSingleBranchRBP Model
##########################################

if __name__ == "__main__":
    # Dummy data: features shape (batch, T, B, C)
    batch_size = 38  
    T, B, C = 30, 5, 19
    num_classes = 2

    model = DICENetSingleBranchRBP(
        T=T,
        B=B,
        C=C,
        d_model=C,   
        nhead=1,     
        num_layers=2,
        dim_feedforward=64,
        dropout=0.1,
        num_classes=num_classes
    )
    
    x_dummy = torch.randn(batch_size, T, B, C)
    logits = model(x_dummy)
    print("Output shape:", logits.shape)  # Expected: (38, num_classes)
