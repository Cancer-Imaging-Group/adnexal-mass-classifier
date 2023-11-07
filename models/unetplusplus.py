import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/milesial/Pytorch-UNet/issues/18
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class NestedUp(nn.Module):
    """Upscaling then double conv with a nested skip connection"""
    
    def __init__(self, in_channels, out_channels, skip_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear \
                  else nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetPlusPlus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        self.up1 = NestedUp(1024, 512, 512, bilinear)
        self.up2 = NestedUp(512, 256, 256 + 128, bilinear)
        self.up3 = NestedUp(256, 128, 128 + 64, bilinear)
        self.up4 = NestedUp(128, 64, 64, bilinear)
        
        self.outc = OutConv(64, n_classes)
        

    def forward(self, x):
        # Encoder part
        x1 = self.inc(x)
        print(f"Shape of x1: {x1.shape}")
        x2 = self.down1(x1)
        print(f"Shape of x2: {x2.shape}")
        x3 = self.down2(x2)
        print(f"Shape of x3: {x3.shape}")
        x4 = self.down3(x3)
        print(f"Shape of x4: {x4.shape}")
        x5 = self.down4(x4)
        print(f"Shape of x5: {x5.shape}")

        # Decoder part
        # Up1
        x = self.up1(x5, x4)
        print(f"Shape after up1: {x.shape}")
        # Up2
        x = self.up2(x, torch.cat([x3, x2], dim=1))
        print(f"Shape after up2 (before cat): x3: {x3.shape}, x2: {x2.shape}, after cat: {x.shape}")
        # Up3
        x = self.up3(x, torch.cat([x2, x1], dim=1))
        print(f"Shape after up3 (before cat): x2: {x2.shape}, x1: {x1.shape}, after cat: {x.shape}")
        # Up4
        x = self.up4(x, x1)
        print(f"Shape after up4 (before cat): x1: {x1.shape}, after cat: {x.shape}")

        # Final convolution
        logits = self.outc(x)
        print(f"Shape of logits: {logits.shape}")
        return logits
