import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F # 引入F模块用于填充
from models import UNet
from utils import * # 假设 utils 中包含了 normalize, batch_PSNR, batch_SSIM 等函数

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- 1. 参数设置 ---
parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers (Placeholder, UNet depth is fixed)")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
opt = parser.parse_args()

# --- 2. 辅助函数 ---
def normalize(data):
    """将图像像素值归一化到 [0, 1]"""
    return data / 255.

# --- 3. 主函数 ---
def main():
    # 设定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # U-Net 现在有3层下采样，下采样因子 D=8 (2^3)
    # 这样可以处理更小的patch，提高训练和测试效率
    DOWNSAMPLE_FACTOR = 8 
    
    # Build model
    print('Loading model ...\n')
    net = UNet(channels=1)
    device_ids = [0]
    # 注意：加载模型时，如果训练时使用了 nn.DataParallel，测试时也要相应处理。
    # 因为在 train.py 中使用了 nn.DataParallel，这里也保留。
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    
    # FutureWarning 处理
    # 警告：这里假设 net.pth 存储的是 DataParallel 模型的 state_dict。
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth'), weights_only=True))
    model.eval()
    
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()
    
    # process data
    psnr_test = 0
    ssim_test = 0
    
    for f in files_source:
        # --- 图像读取和预处理 ---
        Img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        H_orig, W_orig = Img.shape # 记录原始尺寸
        Img = normalize(np.float32(Img))
        
        # --- 关键修复步骤：填充图像至 DOWNSAMPLE_FACTOR 的倍数 ---
        
        # 计算填充量
        H_pad = (H_orig + DOWNSAMPLE_FACTOR - 1) // DOWNSAMPLE_FACTOR * DOWNSAMPLE_FACTOR
        W_pad = (W_orig + DOWNSAMPLE_FACTOR - 1) // DOWNSAMPLE_FACTOR * DOWNSAMPLE_FACTOR
        
        # 如果原始尺寸已经是倍数，则不需要填充
        pad_h = H_pad - H_orig
        pad_w = W_pad - W_orig
        
        # 将 numpy 数组转换为 torch Tensor (C=1, H, W)
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img) 
        
        # 使用 F.pad 进行填充：(左, 右, 上, 下)
        # 注意：这里使用 'replicate'（复制填充），也可以使用 'reflect'
        ISource_padded = F.pad(ISource, (0, pad_w, 0, pad_h), mode='replicate')
        
        # --- 噪声和模型输入准备 ---
        
        # 噪声必须与填充后的 ISource_padded 尺寸相同
        noise = torch.FloatTensor(ISource_padded.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        INoisy = ISource_padded + noise
        
        # 创建噪声图（与 FFDNet 类似）
        noise_map = torch.FloatTensor(ISource_padded.size()).fill_(opt.test_noiseL/255.)
        model_input = torch.cat((INoisy, noise_map), 1)

        # --- 推理 ---
        
        # 将张量移动到设备
        ISource, INoisy = ISource.to(device), INoisy.to(device) # 注意：ISource 是未填充的原始图像块
        model_input = model_input.to(device)
        
        with torch.no_grad(): # this can save much memory
            # Out 是模型输出的去噪图像（填充尺寸）
            Out_padded = torch.clamp(INoisy - model(model_input), 0., 1.)
        
        # --- 关键修复步骤：裁剪回原始尺寸 ---
        
        # 裁剪掉之前填充的像素 (只保留 H_orig x W_orig 部分)
        # Out 的尺寸是 (1, C, H_orig, W_orig)
        Out = Out_padded[..., :H_orig, :W_orig] 
        
        # --- 评估 ---
        
        # ISource 是原始图像（未填充）
        current_psnr = batch_PSNR(Out, ISource, 1.)
        current_ssim = batch_SSIM(Out, ISource, 1.)
        psnr_test += current_psnr
        ssim_test += current_ssim
        print(f"{os.path.basename(f)} PSNR {current_psnr:.4f} SSIM {current_ssim:.4f}")
        
    # 计算平均值并打印
    ssim_test /= len(files_source)
    psnr_test /= len(files_source)
    print(f"\nPSNR on test data {psnr_test:.4f} SSIM on test data {ssim_test:.4f}")

if __name__ == "__main__":
    main()