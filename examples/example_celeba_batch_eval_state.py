import os
import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from blind_watermark import WaterMark
import cv2

# ======== 配置 ========
celeba_root = 'E:\Python Files\CelebA\Img'              # CelebA 数据集路径（img_align_celeba）
output_dir = './output/celeba_wm_results'
num_images = 1000
wm_path = './pic/watermark.png'              # 固定水印图片路径（或GAN生成）
password_img = 1
password_wm = 1

os.makedirs(output_dir, exist_ok=True)

# ======== 攻击函数 ========
def attack_image(img, mode):
    arr = np.array(img)
    if mode.startswith('jpeg'):
        q = int(mode.split('_')[1])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        _, encimg = cv2.imencode('.jpg', arr, encode_param)
        arr = cv2.imdecode(encimg, 1)
    elif mode.startswith('gauss'):
        sigma = int(mode.split('_')[1])
        noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif mode.startswith('crop'):
        ratio = float(mode.split('_')[1])
        h, w, _ = arr.shape
        ch, cw = int(h*(1-ratio)), int(w*(1-ratio))
        arr = arr[(h-ch)//2:(h+ch)//2, (w-cw)//2:(w+cw)//2]
        arr = cv2.resize(arr, (w, h))
    elif mode.startswith('scale'):
        scale_factor = float(mode.split('_')[1])
        h, w, _ = arr.shape
        arr = cv2.resize(arr, (int(w*scale_factor), int(h*scale_factor)))
        arr = cv2.resize(arr, (w, h))
    elif mode == 'median':
        arr = cv2.medianBlur(arr, 3)
    return Image.fromarray(arr)

# ======== 1. 加载 CelebA 数据集 ========
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 缩小加快实验
])
celeba_dataset = datasets.ImageFolder(celeba_root, transform=transform)
print(f"[Info] CelebA 样本数: {len(celeba_dataset)}")

# ======== 2. 生成批量嵌入并评估 ========
psnr_list, ssim_list, ber_list = [], [], []
attack_modes = ['jpeg_50', 'jpeg_30', 'gauss_5', 'gauss_10', 'crop_0.1', 'crop_0.25', 'scale_0.8', 'scale_0.5', 'median']
attack_results = {m: [] for m in attack_modes}

wm = WaterMark(password_img, password_wm)
wm.read_wm(wm_path, mode='img')  # 固定水印（如需GAN生成，这里替换成GAN输出）

for idx in range(num_images):
    cover_img, _ = celeba_dataset[idx]
    cover_path = os.path.join(output_dir, f"cover_{idx:04d}.png")
    cover_img.save(cover_path)

    wm.read_img(cover_path)
    out_path = os.path.join(output_dir, f"watermarked_{idx:04d}.png")
    wm.embed(out_path)

    watermarked_img = Image.open(out_path)

    # PSNR / SSIM
    psnr_val = psnr(np.array(cover_img), np.array(watermarked_img))
    ssim_val = ssim(np.array(cover_img), np.array(watermarked_img), channel_axis=2)
    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)

    # 提取水印，计算BER
    wm.set_path(out_path)
    wm.extract(os.path.join(output_dir, f"extract_{idx:04d}.png"))
    orig_wm = np.array(Image.open(wm_path).convert('L'))
    ext_wm = np.array(Image.open(os.path.join(output_dir, f"extract_{idx:04d}.png")).convert('L'))
    ber = np.mean(orig_wm != ext_wm)
    ber_list.append(ber)

    # 攻击测试
    for mode in attack_modes:
        attacked = attack_image(watermarked_img, mode)
        attacked_path = os.path.join(output_dir, f"attacked_{mode}_{idx:04d}.png")
        attacked.save(attacked_path)

        wm.set_path(attacked_path)
        wm.extract(os.path.join(output_dir, f"ext_{mode}_{idx:04d}.png"))
        ext_attacked_wm = np.array(Image.open(os.path.join(output_dir, f"ext_{mode}_{idx:04d}.png")).convert('L'))
        ber_attack = np.mean(orig_wm != ext_attacked_wm)
        attack_results[mode].append(ber_attack)

    if (idx+1) % 50 == 0:
        print(f"[{idx+1}/{num_images}] done.")

# ======== 3. 打印统计结果 ========
print(f"平均PSNR: {np.mean(psnr_list):.2f} dB")
print(f"平均SSIM: {np.mean(ssim_list):.4f}")
print(f"平均BER(无攻击): {np.mean(ber_list):.4f}")
for mode in attack_modes:
    print(f"{mode} 攻击BER: {np.mean(attack_results[mode]):.4f}")