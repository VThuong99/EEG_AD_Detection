import torch
import torch.nn as nn

class ConvFormer(nn.Module):
    def __init__(self, num_channels, num_freqs, num_times, num_classes, 
                 embed_dim=128, num_heads=8, num_layers=2, dropout=0.1):
        super(ConvFormer, self).__init__()
        
        # CNN Stem
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
        
        # Tính toán độ dài chuỗi sau CNN
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
        print("Input shape:", x.shape)
        x = self.cnn(x)
        print("After CNN:", x.shape)
        x = x.flatten(2)
        print("After flatten:", x.shape)
        x = x.permute(2, 0, 1)
        print("After permute:", x.shape)
        x = self.transformer(x)
        print("After transformer:", x.shape)
        x = x.mean(dim=0)
        print("After mean:", x.shape)
        x = self.fc(x)
        print("Output shape:", x.shape)
        return x

# Ví dụ khởi tạo mô hình
# Cho STFT: num_channels=19, num_freqs=251, num_times=30
stft_model = ConvFormer(num_channels=19, num_freqs=251, num_times=30, num_classes=2)
# Cho CWT: num_channels=19, num_freqs=22, num_times=2000
cwt_model = ConvFormer(num_channels=19, num_freqs=22, num_times=2000, num_classes=2)