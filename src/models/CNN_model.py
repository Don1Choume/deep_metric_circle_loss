import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
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
        self.norm = nn.LayerNorm(1)

    def forward(self, x):
        x = self.conv_bn_relu_pool_1(x)
        x = self.conv_bn_relu_pool_2(x)
        x = self.conv_3(x)
        x = self.gap(x)
        x = self.norm(x)
        # x = F.normalize(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
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
        return x


class CosLayer(nn.Module):
    def __init__(self, num_feature, num_class,
                loss_type='all',
                s=30.0,
                m0=1.35,
                m1=0.50,
                m2=0.35
                ):
        super(CosLayer, self).__init__()
        self.num_feature = num_feature
        self.n_classes = num_class
        self.loss_type = loss_type
        self.s = s
        self.m0 = m0 #SphereFace margin
        self.m1 = m1 #ArcFace margin
        self.m2 = m2 #CosFace margin
        self.W = nn.Parameter(torch.FloatTensor(num_class, num_feature))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, label=None):
        if (self.loss_type == 'softmax') or \
            (self.loss_type == 'adacos'):
            self.m0 = 1.0
            self.m1 = 0.0
            self.m2 = 0.0
        if self.loss_type == 'sphereface':
            self.m1 = 0.0
            self.m2 = 0.0
        if self.loss_type == 'arcface':
            self.m0 = 1.0
            self.m2 = 0.0
        if self.loss_type == 'cosface':
            self.m0 = 1.0
            self.m1 = 0.0

        # normalize features and weights
        x = F.normalize(x)
        W = F.normalize(self.W)

        # return output
        logits = F.linear(x, W)
        if label is None:
            return logits

        # calc margin
        theta = torch.acos(torch.clamp(logits, -1.0+1e-7, 1.0-1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        target_logits = torch.cos(self.m0 * theta + self.m1) - self.m2
        output = logits * (1 - one_hot) + target_logits * one_hot

        # calc optimal scale
        with torch.no_grad():
            pre_s = math.sqrt(2)*math.log(self.num_class-1)
            B_ave = torch.where(one_hot < 1, torch.exp(pre_s * logits),
                                             torch.zeros_like(logits))
            B_ave = torch.sum(B_ave) / x.size(0)
            theta_med = torch.median(theta[one_hot == 1])
            theta_med = torch.min(math.pi/4*torch.ones_like(theta_med),
                                  theta_med)
            opt_s = torch.log(B_ave) / torch.cos(theta_med)

        if (self.loss_type == 'adacos') or (self.loss_type == 'all'):
            self.s = opt_s

        # scale feature
        output *= self.s

        return output

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc(x)
        return x