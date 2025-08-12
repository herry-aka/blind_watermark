# cifar10_gan_wm_eval.py
import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from models.cifar10_model import Generator, Discriminator, DHead, QHead
from utils import weights_init
from config import params

# -------------------
# 1. 配置
# -------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

checkpoint_path = "checkpoint/CIFAR10/model_epoch_30_CIFAR10.pth"  # 训练好模型的路径
save_dir = "eval_results_cifar10"
os.makedirs(save_dir, exist_ok=True)

# -------------------
# 2. 加载模型
# -------------------
print(f"Loading checkpoint from {checkpoint_path} ...")
ckpt = torch.load(checkpoint_path, map_location=device)
params.update(ckpt['params'])  # 用训练时的参数

netG = Generator(
    num_z=params['num_z'],
    num_dis_c=params['num_dis_c'],
    dis_c_dim=params['dis_c_dim'],
    num_con_c=params['num_con_c'],
    out_channels=3
).to(device)
netG.load_state_dict(ckpt['netG'])
netG.eval()

print("Generator loaded.")

# -------------------
# 3. CIFAR-10 数据加载（方便做对比）
# -------------------
data_root = "data/cifar10"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

testset = datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

real_batch = next(iter(testloader))[0].to(device)

# -------------------
# 4. 生成固定噪声图片
# -------------------
print("Generating samples...")
z = torch.randn(100, params['num_z'], 1, 1, device=device)
fixed_noise = z

if params['num_dis_c'] != 0:
    idx = np.arange(params['dis_c_dim']).repeat(10)
    dis_c = torch.zeros(100, params['num_dis_c'], params['dis_c_dim'], device=device)
    for i in range(params['num_dis_c']):
        dis_c[torch.arange(0,100), i, idx] = 1.0
    dis_c = dis_c.view(100, -1, 1, 1)
    fixed_noise = torch.cat((fixed_noise, dis_c), dim=1)

if params['num_con_c'] != 0:
    con_c = torch.rand(100, params['num_con_c'], 1, 1, device=device) * 2 - 1
    fixed_noise = torch.cat((fixed_noise, con_c), dim=1)

with torch.no_grad():
    fake_images = netG(fixed_noise).detach().cpu()

# -------------------
# 5. 保存生成图像和真实图像对比
# -------------------
vutils.save_image(fake_images, os.path.join(save_dir, "fake_samples.png"), nrow=10, normalize=True)
vutils.save_image(real_batch.cpu(), os.path.join(save_dir, "real_samples.png"), nrow=10, normalize=True)

print(f"Saved fake samples to {save_dir}/fake_samples.png")
print(f"Saved real samples to {save_dir}/real_samples.png")

# -------------------
# 6. 可选：FID / IS 评估（需要 pytorch-fid / pytorch-gan-metrics）
# -------------------
try:
    from pytorch_fid import fid_score
    import torchvision

    fake_dir = os.path.join(save_dir, "fake_fid")
    real_dir = os.path.join(save_dir, "real_fid")
    os.makedirs(fake_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)

    # 保存 5000 张 fake / real 图像用于 FID
    print("Saving images for FID computation...")
    count = 0
    with torch.no_grad():
        while count < 5000:
            z = torch.randn(100, params['num_z'], 1, 1, device=device)
            noise = z
            if params['num_dis_c'] != 0:
                idx = np.random.randint(params['dis_c_dim'], size=100)
                dis_c = torch.zeros(100, params['num_dis_c'], params['dis_c_dim'], device=device)
                for i in range(params['num_dis_c']):
                    dis_c[torch.arange(0,100), i, idx] = 1.0
                dis_c = dis_c.view(100, -1, 1, 1)
                noise = torch.cat((noise, dis_c), dim=1)
            if params['num_con_c'] != 0:
                con_c = torch.rand(100, params['num_con_c'], 1, 1, device=device) * 2 - 1
                noise = torch.cat((noise, con_c), dim=1)

            batch_fake = netG(noise).detach().cpu()
            for img in batch_fake:
                torchvision.utils.save_image(img, os.path.join(fake_dir, f"{count}.png"), normalize=True)
                count += 1
                if count >= 5000:
                    break

    # 保存真实图像
    count = 0
    for img, _ in testset:
        torchvision.utils.save_image(img, os.path.join(real_dir, f"{count}.png"), normalize=True)
        count += 1
        if count >= 5000:
            break

    fid_value = fid_score.calculate_fid_given_paths([real_dir, fake_dir], batch_size=50, device=device, dims=2048)
    print(f"FID score: {fid_value:.4f}")

except ImportError:
    print("FID computation skipped. Install pytorch-fid to enable it.")
