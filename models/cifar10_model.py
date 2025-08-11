import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # InfoGAN 输入通道数：z_dim + dis_c_dim*num_dis_c + num_con_c
        # 这里先按 DCGAN 标准输入 128 维
        self.tconv1 = nn.ConvTranspose2d(140, 256, 4, 1, 0, bias=False)

        self.bn1 = nn.BatchNorm2d(256)

        self.tconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        img = torch.tanh(self.tconv4(x))
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
        return x

class DHead(nn.Module):
    """
    通用判别器头 (discriminator head)：
    - 自适应地把 feature map 池化到 1x1（适配任意输入分辨率）
    - 用 1x1 卷积映射到单通道，再 sigmoid
    - 返回 shape: (batch,) 的概率张量，可直接与 label (batch,) 做 BCE
    """
    def __init__(self, in_channels=256):
        super().__init__()
        # in_channels: 判别器最后一层输出的通道数（多数模型为 256）
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 任意 HxW -> 1x1
        self.conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        # optional: 一个小的线性替代也可以，但 Conv2d 保持在卷积风格的一致性

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.pool(x)            # (B, C, 1, 1)
        x = self.conv1x1(x)         # (B, 1, 1, 1)
        x = torch.sigmoid(x)        # (B, 1, 1, 1)
        x = x.view(x.size(0))       # (B,)
        return x

class QHead(nn.Module):
    def __init__(self, in_channels=256, num_dis_c=1, dis_c_dim=10, num_con_c=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        # 离散潜变量预测
        self.conv_disc = None
        if num_dis_c > 0:
            self.conv_disc = nn.Conv2d(128, num_dis_c * dis_c_dim, 1)

        # 连续潜变量预测
        self.conv_mu = None
        self.conv_var = None
        if num_con_c > 0:
            self.conv_mu = nn.Conv2d(128, num_con_c, 1)
            self.conv_var = nn.Conv2d(128, num_con_c, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)
        disc_logits, mu, var = None, None, None

        if self.conv_disc is not None:
            disc_logits = self.conv_disc(x).squeeze()

        if self.conv_mu is not None:
            mu = self.conv_mu(x).squeeze()
            var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var
