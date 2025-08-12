import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random

from models.mnist_model import Generator, Discriminator, DHead, QHead
from dataloader import get_data
from utils import *
from config import params

if(params['dataset'] == 'MNIST'):
    from models.mnist_model import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'CelebA'):
    from models.celeba_model import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'FashionMNIST'):
    from models.mnist_model import Generator, Discriminator, DHead, QHead
elif(params['dataset'] == 'CIFAR10'):
    from models.cifar10_model import Generator, Discriminator, DHead, QHead


# Set random seed for reproducibility.
seed = 1123
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Use GPU if available.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

dataloader = get_data(params['dataset'], params['batch_size'])

# 根据数据集设置InfoGAN的潜在变量参数（遵循原论文设置）
# num_z: 不可压缩噪声维度
# num_dis_c: 离散潜在变量数量
# dis_c_dim: 离散潜在变量维度
# num_con_c: 连续潜在变量数量
if(params['dataset'] == 'MNIST'):
    params['num_z'] = 62
    params['num_dis_c'] = 1
    params['dis_c_dim'] = 10
    params['num_con_c'] = 2
elif(params['dataset'] == 'CIFAR10'):
    params['num_z'] = 64            # 试试 64 比 128 更容易收敛
    params['num_dis_c'] = 1
    params['dis_c_dim'] = 10
    params['num_con_c'] = 2
elif(params['dataset'] == 'CelebA'):
    params['num_z'] = 128
    params['num_dis_c'] = 10
    params['dis_c_dim'] = 10
    params['num_con_c'] = 0

# 绘制训练图像样本
sample_batch = next(iter(dataloader))
plt.figure(figsize=(10, 10))
plt.axis("off")
# 生成网格图像并显示
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0].to(device)[ : 100], nrow=10, padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.savefig('Training Images {}'.format(params['dataset']))
plt.close('all')# 关闭所有图像窗口

# 初始化网络模型
netG = Generator().to(device)
netG.apply(weights_init)
print(netG)

discriminator = Discriminator().to(device)
discriminator.apply(weights_init)
print(discriminator)

netD = DHead().to(device)
netD.apply(weights_init)
print(netD)

netQ = QHead().to(device)
netQ.apply(weights_init)
print(netQ)

#定义损失函数
#二分类交叉熵损失（用于真假判断）
criterionD = nn.BCELoss()
# 交叉熵损失（用于离散潜在变量）
criterionQ_dis = nn.CrossEntropyLoss()
# 正态负对数似然损失（用于连续潜在变量）
criterionQ_con = NormalNLLLoss()


# 定义优化器（Adam优化器）
# 鉴别器相关参数（discriminator和netD）
optimD = optim.Adam([{'params': discriminator.parameters()}, {'params': netD.parameters()}], lr=params['d_learning_rate'], betas=(params['beta1'], params['beta2']))
# 生成器相关参数（netG和netQ）
optimG = optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}], lr=params['g_learning_rate'], betas=(params['beta1'], params['beta2']))

# 固定噪声（用于监控生成器训练过程）
z = torch.randn(100, params['num_z'], 1, 1, device=device)
fixed_noise = z
if(params['num_dis_c'] != 0):
    idx = np.arange(params['dis_c_dim']).repeat(10)
    dis_c = torch.zeros(100, params['num_dis_c'], params['dis_c_dim'], device=device)
    for i in range(params['num_dis_c']):
        dis_c[torch.arange(0, 100), i, idx] = 1.0

    dis_c = dis_c.view(100, -1, 1, 1)

    fixed_noise = torch.cat((fixed_noise, dis_c), dim=1)

if(params['num_con_c'] != 0):
    con_c = torch.rand(100, params['num_con_c'], 1, 1, device=device) * 2 - 1
    fixed_noise = torch.cat((fixed_noise, con_c), dim=1)
# 定义标签（真/假）
real_label = 0.9
fake_label = 0.1

# 存储训练结果的列表
img_list = []
G_losses = []
D_losses = []


# 打印训练开始信息
print("-"*25)
print("Starting Training Loop...\n")
print('Epochs: %d\nDataset: {}\nBatch Size: %d\nLength of Data Loader: %d'.format(params['dataset']) % (params['num_epochs'], params['batch_size'], len(dataloader)))
print("-"*25)
# 记录训练开始时间
start_time = time.time()
iters = 0


# 开始训练循环
for epoch in range(params['num_epochs']):
    epoch_start_time = time.time()
    # 遍历数据加载器
    for i, (data, _) in enumerate(dataloader, 0):
        # 获取批次大小
        b_size = data.size(0)
        # 将数据转移到设备
        real_data = data.to(device)

        # 更新鉴别器和DHead（保持1次更新）
        optimD.zero_grad()
        # 真实数据处理
        label = torch.full((b_size, ), real_label, device=device, dtype=torch.float32)
        output1 = discriminator(real_data)
        probs_real = netD(output1).view(-1)
        loss_real = criterionD(probs_real, label)
        loss_real.backward()

        # 虚假数据处理（用于判别器训练）
        label.fill_(fake_label)
        noise, idx = noise_sample(params['num_dis_c'], params['dis_c_dim'], params['num_con_c'], params['num_z'], b_size, device)
        fake_data = netG(noise)
        output2 = discriminator(fake_data.detach())
        probs_fake = netD(output2).view(-1)
        loss_fake = criterionD(probs_fake, label)
        loss_fake.backward()

        # 鉴别器总损失与更新
        D_loss = loss_real + loss_fake
        optimD.step()

        # 更新生成器和QHead（每个batch更新2次）
        for _ in range(2):  # 增加生成器更新次数（可调整为2/3次）
            optimG.zero_grad()
            
            # 每次更新生成器时重新生成噪声和虚假数据（使用当前生成器参数）
            noise, idx = noise_sample(params['num_dis_c'], params['dis_c_dim'], params['num_con_c'], params['num_z'], b_size, device)
            fake_data = netG(noise)
            
            # 将虚假数据视为真实数据来训练生成器
            output = discriminator(fake_data)
            label.fill_(real_label)
            probs_fake = netD(output).view(-1)
            gen_loss = criterionD(probs_fake, label)
            
            # Q网络输出与损失计算
            q_logits, q_mu, q_var = netQ(output)
            target = torch.LongTensor(idx).to(device)
            dis_loss = 0
            for j in range(params['num_dis_c']):
                dis_loss += criterionQ_dis(q_logits[:, j*10 : j*10 + 10], target[j])
            
            con_loss = 0
            if (params['num_con_c'] != 0):
                con_loss = criterionQ_con(noise[:, params['num_z']+ params['num_dis_c']*params['dis_c_dim'] : ].view(-1, params['num_con_c']), q_mu, q_var)*0.1
            
            # 生成器总损失与更新
            G_loss = gen_loss + dis_loss + con_loss
            G_loss.backward()
            optimG.step()

        # 每100次迭代打印一次训练进度
        if i != 0 and i%100 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch+1, params['num_epochs'], i, len(dataloader), 
                    D_loss.item(), G_loss.item()))

        # 保存损失值
        G_losses.append(G_loss.item())
        D_losses.append(D_loss.item())

        iters += 1  # 迭代次数加1

    epoch_time = time.time() - epoch_start_time
    print("Time taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))
    # 每个epoch结束后生成图像，用于后续制作动画
    with torch.no_grad():
        gen_data = netG(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True))

    # 在第1个epoch和中间epoch保存生成图像
    if((epoch+1) == 1 or (epoch+1) == params['num_epochs']/2):
        with torch.no_grad():
            gen_data = netG(fixed_noise).detach().cpu()
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1,2,0)))
        plt.savefig("Epoch_%d {}".format(params['dataset']) %(epoch+1))
        plt.close('all')

    # 定期保存模型 checkpoint
    import os
    if (epoch + 1) % params['save_epoch'] == 0:
        save_dir = os.path.join("checkpoint", params['dataset'])
        os.makedirs(save_dir, exist_ok=True)  # 自动创建目录

        save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}_{params['dataset']}.pth")
        torch.save({
            'netG': netG.state_dict(),
            'discriminator': discriminator.state_dict(),
            'netD': netD.state_dict(),
            'netQ': netQ.state_dict(),
            'optimD': optimD.state_dict(),
            'optimG': optimG.state_dict(),
            'params': params
        }, save_path)

# 计算总训练时间
training_time = time.time() - start_time
print("-"*50)
print('Training finished!\nTotal Time for Training: %.2fm' %(training_time / 60))
print("-"*50)

# 训练结束后生成最终图像
with torch.no_grad():
    gen_data = netG(fixed_noise).detach().cpu()
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1,2,0)))
plt.savefig("Epoch_%d_{}".format(params['dataset']) %(params['num_epochs']))

# 保存最终模型
torch.save({
    'netG' : netG.state_dict(),
    'discriminator' : discriminator.state_dict(),
    'netD' : netD.state_dict(),
    'netQ' : netQ.state_dict(),
    'optimD' : optimD.state_dict(),
    'optimG' : optimG.state_dict(),
    'params' : params
    }, 'checkpoint/model_final_{}'.format(params['dataset']))


# 绘制训练损失曲线
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("Loss Curve {}".format(params['dataset']))

# 制作生成器训练过程的动画
fig = plt.figure(figsize=(10,10))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
anim.save('infoGAN_{}.gif'.format(params['dataset']), dpi=80, writer='imagemagick')
plt.show()