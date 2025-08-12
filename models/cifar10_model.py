# models/cifar10_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """
    参数化 Generator，输入通道自动根据 InfoGAN 配置计算：
      in_ch = num_z + num_dis_c * dis_c_dim + num_con_c
    out_channels: 3 for RGB, 1 for gray.
    """
    def __init__(self, num_z=64, num_dis_c=1, dis_c_dim=10, num_con_c=2, out_channels=3):
        super().__init__()
        in_ch = num_z + num_dis_c * dis_c_dim + num_con_c

        self.tconv1 = nn.ConvTranspose2d(in_ch, 256, 4, 1, 0, bias=False)  # 1x1 -> 4x4
        self.bn1 = nn.BatchNorm2d(256)

        self.tconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)    # 4x4 -> 8x8
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)     # 8x8 -> 16x16
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(64, out_channels, 4, 2, 1, bias=False)  # 16x16 -> 32x32

        # activation: tanh => output in [-1,1]
    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        img = torch.tanh(self.tconv4(x))
        return img


class Discriminator(nn.Module):
    """
    Backbone 判别器：输出 feature map (B,256,H,W).
    Use this with DHead below.
    """
    def __init__(self, in_channels=3, use_spectral_norm=False):
        super().__init__()
        conv = nn.utils.spectral_norm if use_spectral_norm else (lambda x: x)

        self.conv1 = conv(nn.Conv2d(in_channels, 64, 4, 2, 1))
        self.conv2 = conv(nn.Conv2d(64, 128, 4, 2, 1, bias=False))
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv(nn.Conv2d(128, 256, 4, 2, 1, bias=False))
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
        return x


class DHead(nn.Module):
    """
    自适应判别器头：任何输入分辨率都会被 pool->1x1，然后 conv1x1->sigmoid -> (B,)
    """
    def __init__(self, in_channels=256, use_spectral_norm=False):
        super().__init__()
        conv = nn.utils.spectral_norm if use_spectral_norm else (lambda x: x)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1 = conv(nn.Conv2d(in_channels, 1, kernel_size=1, bias=True))

    def forward(self, x):
        x = self.pool(x)
        x = self.conv1x1(x)
        x = torch.sigmoid(x)
        return x.view(x.size(0))


class QHead(nn.Module):
    """
    QHead 根据 num_dis_c / dis_c_dim / num_con_c 自动构建输出通道。
    返回 (disc_logits, mu, var)；当对应项不存在时，返回 None。
    """
    def __init__(self, in_channels=256, num_dis_c=1, dis_c_dim=10, num_con_c=2):
        super().__init__()
        self.num_dis_c = num_dis_c
        self.dis_c_dim = dis_c_dim
        self.num_con_c = num_con_c

        self.conv1 = nn.Conv2d(in_channels, 128, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        if num_dis_c > 0:
            self.conv_disc = nn.Conv2d(128, num_dis_c * dis_c_dim, 1)
        else:
            self.conv_disc = None

        if num_con_c > 0:
            self.conv_mu = nn.Conv2d(128, num_con_c, 1)
            self.conv_var = nn.Conv2d(128, num_con_c, 1)
        else:
            self.conv_mu = None
            self.conv_var = None

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)
        disc_logits, mu, var = None, None, None

        if self.conv_disc is not None:
            disc_logits = self.conv_disc(x).squeeze()  # shape: (B, num_dis_c*dis_c_dim) or similar

        if self.conv_mu is not None:
            mu = self.conv_mu(x).squeeze()             # shape: (B, num_con_c)
            var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var
