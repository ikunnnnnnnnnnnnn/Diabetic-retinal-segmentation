import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), # Add bias=False
        nn.BatchNorm2d(out_channels), # Add BatchNorm
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), # Add bias=False
        nn.BatchNorm2d(out_channels), # Add BatchNorm
        nn.ReLU(inplace=True)
    )

class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Encoder (ResNet34)
        resnet = models.resnet34(pretrained=True)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  # 64, /4
        self.encoder2 = resnet.layer1  # 64, /4
        self.encoder3 = resnet.layer2  # 128, /8
        self.encoder4 = resnet.layer3  # 256, /16
        self.encoder5 = resnet.layer4  # 512, /32

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2, bias=False) # -> /16
        self.bn1 = nn.BatchNorm2d(256)
        self.dec1 = double_conv(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False) # -> /8
        self.bn2 = nn.BatchNorm2d(128)
        self.dec2 = double_conv(128 + 128, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False) # -> /4
        self.bn3 = nn.BatchNorm2d(64)
        self.dec3 = double_conv(64 + 64, 64)

        self.up4 = nn.ConvTranspose2d(64, 64, 2, stride=2, bias=False) # -> /2 (相对于 /4)
        self.bn4 = nn.BatchNorm2d(64)
        self.dec4 = double_conv(64 + 64, 64)

        # Add one more upsampling layer to reach /1 -> 2048x2048
        self.up5 = nn.ConvTranspose2d(64, 64, 2, stride=2, bias=False) # -> /1
        self.bn5 = nn.BatchNorm2d(64)
        self.dec5 = double_conv(64 + 64, 64) # Consider concatenating with original input or a lower-level feature

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)  # /4
        x2 = self.encoder2(x1) # /4
        x3 = self.encoder3(x2) # /8
        x4 = self.encoder4(x3) # /16
        x5 = self.encoder5(x4) # /32

        # Decoder
        x = self.up1(x5) # /16
        x = self.bn1(x)
        x = torch.cat([x, x4], dim=1)
        x = self.dec1(x)

        x = self.up2(x) # /8
        x = self.bn2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec2(x)

        x = self.up3(x) # /4
        x = self.bn3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec3(x)

        x = self.up4(x) # /2
        x = self.bn4(x)

        # Upsample x1 to the size of x before concatenation
        upsample_x1 = nn.Upsample(size=x.shape[2:], mode='bilinear', align_corners=True)
        x1_upsampled = upsample_x1(x1)

        x = torch.cat([x, x1_upsampled], dim=1)
        x = self.dec4(x)

        x = self.up5(x) # /1
        x = self.bn5(x)
        # Consider concatenating with the original input 'x' (before encoding) if it helps with low-level features
        upsample_original = nn.Upsample(size=x.shape[2:], mode='bilinear', align_corners=True)
        original_upsampled = upsample_original(x) # Upsample original input to current feature map size
        x = torch.cat([x, original_upsampled], dim=1) # Concatenate with upsampled original input
        x = self.dec5(x)

        x = self.outc(x)
        # Remove the final_upsample layer
        # x = self.final_upsample(x)

        return x

if __name__ == '__main__':
    # Example usage
    model = ResUNet(in_channels=3, out_channels=5)
    input_tensor = torch.randn(1, 3, 256, 256)
    output_tensor = model(input_tensor)
    print("输入形状:", input_tensor.shape)
    print("输出形状:", output_tensor.shape)