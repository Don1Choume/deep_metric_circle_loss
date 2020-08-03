import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        self.conv_bn_relu_pool_1 = nn.Sequential(
            nn.Conv2d(1,8,3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv_bn_relu_pool_2 = nn.Sequential(
            nn.Conv2d(8,16,3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv_3 = nn.Conv2d(16,64,3)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.norm = nn.LayerNorm

    def forward(self, x):
        x = self.conv_bn_relu_pool_1(x)
        x = self.conv_bn_relu_pool_2(x)
        x = self.conv_3(x)
        x = self.gap(x)
        x = self.norm(x)


class Decoder(nn.Module):
    def __init__(self):
        self.upconv_bn_relu_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4), # 4x4
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.upconv_bn_relu_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 4, 6, stride=2), # 12x12
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.upconv_3 = nn.ConvTranspose2d(4, 1, 6, stride=2) # 28x28
        self.hard_sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        x = self.upconv_bn_relu_1(x)
        x = self.upconv_bn_relu_2(x)
        x = self.upconv_3(x)
        x = self.hard_sigmoid(x)


class Classifier(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc(x)