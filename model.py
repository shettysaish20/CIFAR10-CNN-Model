import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=2, padding=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                             dilation=dilation, padding=padding)

    def forward(self, x):
        return self.conv(x)

class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        
        # C1 block - Initial feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),  # 32x32x16, RF=3
            nn.BatchNorm2d(16),
            nn.ReLU(),
            DepthwiseSeparableConv(16, 24, kernel_size=3),  # 32x32x24, RF=5
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 16x16x32, RF=7
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # C2 block
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(32, 48, kernel_size=3),  # 16x16x48, RF=11
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, padding=1, stride=2, bias=False),  # 8x8x64, RF=15
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # C3 block with Dilated Conv
        self.conv3 = nn.Sequential(
            DilatedConv(64, 64, kernel_size=3, dilation=2),  # 8x8x64, RF=31
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, stride=2, bias=False),  # 4x4x96, RF=47
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )

        # C4 block - Final features
        self.conv4 = nn.Sequential(
            DepthwiseSeparableConv(96, 128, kernel_size=3),  # 4x4x128, RF=55
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 48, kernel_size=1, bias=False),  # 4x4x48 (channel reduction)
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(48, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
