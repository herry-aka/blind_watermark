import os
import sys
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from blind_watermark import WaterMark, att

# 获取当前脚本的绝对路径
current_script = os.path.abspath(__file__)
# 获取当前脚本所在的目录（examples 文件夹）
current_dir = os.path.dirname(current_script)
# 获取项目根目录（current_dir 的上级目录，即 blind_watermark 根目录）
project_root = os.path.dirname(current_dir)
# 将项目根目录加入模块搜索路径（放在最前面，确保优先搜索）
sys.path.insert(0, project_root)

# ==================== 配置 ====================
celeba_root = 'E:/Python Files/bwm/CelebA/Img'              # CelebA 数据集路径（img_align_celeba）
output_dir = './output/celeba_wm_results'
num_images = 1000
ckpt_path = 'checkpoint/model_final_CelebA'  # InfoGAN 权重
password_img = 1
password_wm = 1

# 攻击模式列表（调用 blind_watermark.att）
attack_modes = [
    ('jpeg_50', lambda arr: att.jpeg_att(arr, 50)),
    ('jpeg_30', lambda arr: att.jpeg_att(arr, 30)),
    ('gauss_5', lambda arr: att.gaussian_noise_att(arr, 5)),
    ('gauss_10', lambda arr: att.gaussian_noise_att(arr, 10)),
    ('salt_0.01', lambda arr: att.salt_pepper_att(arr, 0.01)),
    ('salt_0.05', lambda arr: att.salt_pepper_att(arr, 0.05)),
    ('resize_0.8', lambda arr: att.resize_att(arr, 0.8)),
    ('resize_0.5', lambda arr: att.resize_att(arr, 0.5)),
    ('crop_0.1', lambda arr: att.crop_att(arr, 0.1)),
    ('crop_0.25', lambda arr: att.crop_att(arr, 0.25)),
]

os.makedirs(output_dir, exist_ok=True)

# NCC 计算函数
def calc_ncc(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    numerator = np.sum((img1 - mean1) * (img2 - mean2))
    denominator = np.sqrt(np.sum((img1 - mean1) ** 2) * np.sum((img2 - mean2) ** 2))
    return numerator / denominator if denominator != 0 else 0

# 加载 InfoGAN 生成器 
from models.celeba_model import Generator

state_dict = torch.load(ckpt_path, map_location='cpu')
params = state_dict['params']

netG = Generator()
netG.load_state_dict(state_dict['netG'])
netG.eval()

z_dim = params['num_z']
dis_c_dim = params['dis_c_dim']
num_dis_c = params['num_dis_c']
num_con_c = params['num_con_c']

# 加载 CelebA 数据集
transform = transforms.Compose([
    transforms.Resize((128, 128)),
])
celeba_dataset = datasets.ImageFolder(celeba_root, transform=transform)
print(f"[Info] CelebA 样本数: {len(celeba_dataset)}")

#评测存储变量
psnr_list, ssim_list, ber_list, ncc_list = [], [], [], []
attack_results_ber = {name: [] for name, _ in attack_modes}
attack_results_ncc = {name: [] for name, _ in attack_modes}

# 主循环
for idx in range(num_images):
    cover_img, _ = celeba_dataset[idx]
    cover_path = os.path.join(output_dir, f"cover_{idx:04d}.png")
    cover_img.save(cover_path)

    #  1. 用 InfoGAN 生成动态扰动 
    z = torch.randn(1, z_dim, 1, 1)
    dis_c = torch.zeros(1, num_dis_c, dis_c_dim)
    dis_c[0, 0, np.random.randint(dis_c_dim)] = 1.0
    dis_c = dis_c.view(1, -1, 1, 1)
    con_c = torch.rand(1, num_con_c, 1, 1) * 2 - 1
    noise = torch.cat([z, dis_c, con_c], dim=1)

    with torch.no_grad():
        perturb = netG(noise).squeeze(0).squeeze(0).numpy()

    perturb_img = Image.fromarray((perturb * 255).astype(np.uint8)).resize(cover_img.size).convert('RGB')
    wm_path = os.path.join(output_dir, f"wm_{idx:04d}.png")
    perturb_img.save(wm_path)

    #  2. 嵌入水印 
    wm = WaterMark(password_img, password_wm)
    wm.read_img(cover_path)
    wm.read_wm(wm_path, mode='img')
    out_path = os.path.join(output_dir, f"watermarked_{idx:04d}.png")
    wm.embed(out_path)

    watermarked_img = Image.open(out_path)

    #  3. PSNR / SSIM
    psnr_val = psnr(np.array(cover_img), np.array(watermarked_img))
    ssim_val = ssim(np.array(cover_img), np.array(watermarked_img), channel_axis=2)
    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)

    #  4. 提取水印 & 计算 BER / NCC（无攻击）
    wm.set_path(out_path)
    extracted_path = os.path.join(output_dir, f"extract_{idx:04d}.png")
    wm.extract(extracted_path)

    orig_wm = np.array(Image.open(wm_path).convert('L'))
    ext_wm = np.array(Image.open(extracted_path).convert('L'))
    ber = np.mean(orig_wm != ext_wm)
    ncc = calc_ncc(orig_wm, ext_wm)
    ber_list.append(ber)
    ncc_list.append(ncc)

    #  5. 攻击测试（调用 att.py） 
    for name, attack_func in attack_modes:
        attacked_arr = attack_func(np.array(watermarked_img))
        attacked_img = Image.fromarray(attacked_arr)
        attacked_path = os.path.join(output_dir, f"attacked_{name}_{idx:04d}.png")
        attacked_img.save(attacked_path)

        wm.set_path(attacked_path)
        ext_attacked_path = os.path.join(output_dir, f"ext_{name}_{idx:04d}.png")
        wm.extract(ext_attacked_path)

        ext_attacked_wm = np.array(Image.open(ext_attacked_path).convert('L'))
        ber_attack = np.mean(orig_wm != ext_attacked_wm)
        ncc_attack = calc_ncc(orig_wm, ext_attacked_wm)
        attack_results_ber[name].append(ber_attack)
        attack_results_ncc[name].append(ncc_attack)

    if (idx+1) % 20 == 0:
        print(f"[{idx+1}/{num_images}] 已处理")

#  输出统计结果 
avg_psnr = np.mean(psnr_list)
avg_ssim = np.mean(ssim_list)
avg_ber = np.mean(ber_list)
avg_ncc = np.mean(ncc_list)

print(f"\n[结果统计] 平均PSNR: {avg_psnr:.2f} dB")
print(f"[结果统计] 平均SSIM: {avg_ssim:.4f}")
print(f"[结果统计] 平均BER(无攻击): {avg_ber:.4f}")
print(f"[结果统计] 平均NCC(无攻击): {avg_ncc:.4f}")
for name in attack_results_ber:
    print(f"{name} 攻击BER: {np.mean(attack_results_ber[name]):.4f} | 攻击NCC: {np.mean(attack_results_ncc[name]):.4f}")

# 保存到 CSV
csv_path = os.path.join(output_dir, 'results.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['PSNR', avg_psnr])
    writer.writerow(['SSIM', avg_ssim])
    writer.writerow(['BER_no_attack', avg_ber])
    writer.writerow(['NCC_no_attack', avg_ncc])
    for name in attack_results_ber:
        writer.writerow([f'BER_{name}', np.mean(attack_results_ber[name])])
        writer.writerow([f'NCC_{name}', np.mean(attack_results_ncc[name])])

print(f"\n[Info] 结果已保存到 {csv_path}")
