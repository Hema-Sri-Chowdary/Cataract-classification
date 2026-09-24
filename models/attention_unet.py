import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """
    Attention Gate (AG) for Attention U-Net.
    Suppresses background regions and magnifies salient lens nucleus features.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Align spatial dimensions if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
            
        net = self.relu(g1 + x1)
        alpha = self.psi(net)
        return x * alpha

class ConvBlock(nn.Module):
    """Double Convolution Block"""
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpConvBlock(nn.Module):
    """Upsampling Block with Attention Gate"""
    def __init__(self, in_ch, out_ch):
        super(UpConvBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.ag = AttentionGate(F_g=out_ch, F_l=out_ch, F_int=out_ch // 2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, g, x):
        g_up = self.up(g)
        if g_up.shape[2:] != x.shape[2:]:
            g_up = F.interpolate(g_up, size=x.shape[2:], mode='bilinear', align_corners=True)
        x_att = self.ag(g=g_up, x=x)
        out = torch.cat((x_att, g_up), dim=1)
        return self.conv(out)

class AttentionUNet(nn.Module):
    """
    Attention U-Net for Ocular Lens Nucleus Segmentation.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(AttentionUNet, self).__init__()
        
        self.enc1 = ConvBlock(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = ConvBlock(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = ConvBlock(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = ConvBlock(features[2], features[3])
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[3], features[3] * 2)
        
        # Decoder
        self.up4 = UpConvBlock(features[3] * 2, features[3])
        self.up3 = UpConvBlock(features[3], features[2])
        self.up2 = UpConvBlock(features[2], features[1])
        self.up1 = UpConvBlock(features[1], features[0])
        
        # Output Head
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        b = self.bottleneck(p4)
        
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        
        mask = torch.sigmoid(self.final_conv(d1))
        return mask

if __name__ == '__main__':
    model = AttentionUNet()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("Attention U-Net output shape:", out.shape)
