import torch
import torch.nn as nn
import torch.nn.functional as F

""" This is the Pytorch implement for this model. https://github.com/ThreePoundUniverse/SZD2022JBHI"""

# GAP1Dtime: averages over the time dimension (here we assume time is dimension 2 in conv output)
class GAP1Dtime(nn.Module):
    def forward(self, x):
        # x: (B, hidden_size, T, W) -> average over T (time dimension)
        return x.mean(dim=2)

# ClassToken: prepends a learnable token along the sequence dimension
class ClassToken(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Create a parameter of shape (1, 1, hidden_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_size))
    
    def forward(self, x):
        # x: (B, seq_len, hidden_size)
        batch_size = x.size(0)
        cls_token = self.cls.expand(batch_size, -1, -1)  # (B, 1, hidden_size)
        return torch.cat((cls_token, x), dim=1)  # (B, seq_len+1, hidden_size)

# AddPositionEmbs: adds learnable positional embeddings
class AddPositionEmbs(nn.Module):
    def __init__(self, seq_len, hidden_size):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_size) * 0.06)
    
    def forward(self, x):
        return x + self.pos_embedding

# MultiHeadSelfAttention replicates the standard scaled dot-product attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.projection_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        # x: (B, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = x.size()
        Q = self.query(x)  # (B, seq_len, hidden_size)
        K = self.key(x)
        V = self.value(x)
        # Reshape and transpose for multi-head: (B, num_heads, seq_len, projection_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.projection_dim).transpose(1,2)
        K = K.view(batch_size, seq_len, self.num_heads, self.projection_dim).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_heads, self.projection_dim).transpose(1,2)
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.projection_dim ** 0.5)  # (B, num_heads, seq_len, seq_len)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # (B, num_heads, seq_len, projection_dim)
        context = context.transpose(1,2).contiguous().view(batch_size, seq_len, hidden_size)
        out = self.out(context)
        return out, attn

# TransformerBlock: standard block with LayerNorm, multi-head attention and an MLP block with GELU activation
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        y, attn = self.attn(self.norm1(x))
        x = x + self.dropout1(y)
        x = x + self.mlp(self.norm2(x))
        return x, attn

# CNN_TransSeizNet: the full model
class CNN_TransSeizNet(nn.Module):
    def __init__(self, channels, samplepoints, classes, num_layers=2, num_heads=4, 
                 hidden_size=128, mlp_dim=256, dropout=0.1, activation='sigmoid'):
        """
        Args:
            channels: number of input channels (e.g., EEG channels)
            samplepoints: number of time samples per trial
            classes: number of output classes
        """
        super().__init__()
        self.channels = channels
        self.samplepoints = samplepoints
        # --- Convolutional Feature Extractor ---
        # Input expected shape: (B, channels, samplepoints)
        # Permute to (B, samplepoints, channels) then unsqueeze to get (B, 1, samplepoints, channels)
        # We mimic the TF code which applies Conv2D with kernel (4,1) along time.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4,1), 
                               stride=(2,1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, hidden_size, kernel_size=(4,1), stride=(2,1), padding=(1, 0))

        # self.conv2 = nn.Conv2d(32, 64, kernel_size=(4,1), stride=(2,1), padding=(1, 0))
        # self.bn2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(64, hidden_size, kernel_size=(4,1), stride=(2,1), padding=(1, 0))
        # --- Global Average Pooling (over the time dimension) ---
        self.gap = GAP1Dtime()
        # After conv layers and GAP, we expect shape: (B, hidden_size, W) where W ~ channels.
        # Permute to (B, W, hidden_size) so that each token corresponds to one channel.
        # --- Class token and Positional Embedding ---
        self.class_token = ClassToken(hidden_size)
        # The sequence length will be (W + 1). We cannot fix W a priori because it depends on conv output.
        # We will initialize positional embeddings in forward if needed.
        self.pos_emb = None
        # --- Transformer Blocks ---
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_dim, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        # --- Classification Head ---
        self.head = nn.Linear(hidden_size, classes)
        if activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'softmax':
            self.activation = lambda x: torch.softmax(x, dim=-1)
        else:
            self.activation = None

    def forward(self, x):
        # Permute to (B, samplepoints, channels) then unsqueeze to (B, 1, samplepoints, channels)
        x = x.permute(0, 2, 1).unsqueeze(1)

        # --- Convolutional Layers ---
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = F.relu(x)

        # x = self.conv3(x)
        # print("After conv3:", x.shape)
        # x = F.relu(x)
        # x: (B, hidden_size, T, channels_out)

        # --- Global Average Pooling over T dimension ---
        x = self.gap(x)  # now shape: (B, hidden_size, channels_out)
        # Permute to (B, channels_out, hidden_size) so that each channel becomes a token.
        x = x.permute(0, 2, 1)

        # --- Append Class Token ---
        x = self.class_token(x)
        # --- Add Positional Embeddings ---
        if self.pos_emb is None or self.pos_emb.size(1) != x.size(1):
            self.pos_emb = nn.Parameter(torch.randn(1, x.size(1), x.size(2)) * 0.06).to(x.device)
        x = x + self.pos_emb

        # --- Transformer Blocks ---
        for i, block in enumerate(self.transformer_layers):
            x, attn = block(x)

        x = self.norm(x)
        # --- Extract Class Token ---
        x_cls = x[:, 0]  # (B, hidden_size)

        x_out = self.head(x_cls)

        if self.activation is not None:
            x_out = self.activation(x_out) if self.activation != torch.softmax else self.activation(x_out, dim=-1)
        return x_out


