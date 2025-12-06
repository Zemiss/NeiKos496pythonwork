import cv2
import os
import argparse
import glob
import numpy as np
import jittor as jt
from jittor import nn
from models import UNet 
from utils import * # 假设 utils 中包含了 batch_PSNR, batch_SSIM 等函数
jt.flags.use_cuda = 1

# --- 1. 参数设置 ---
parser = argparse.ArgumentParser(description="DnCNN_Test")

parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers (Placeholder)")
parser.add_argument("--logdir", type=str, default="logs", help='Directory of the log/model files (where trained weights are saved)')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or BSD68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')

opt = parser.parse_args()

# --- 2. 辅助函数 ---
def normalize(data):
    """将图像像素值归一化到 [0, 1]"""
    return data / 255.

# --- 3. 主函数 ---
def main():
    print(f"Using Jittor with CUDA: {jt.flags.use_cuda}")
    
    DOWNSAMPLE_FACTOR = 8 
    
    # Build model
    print('Loading model ...\n')
    net = UNet(channels=1)
    model = net
    
    model_path = os.path.join(opt.logdir, 'net.pkl')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
        
    model.load_state_dict(jt.load(model_path))
    model.eval() 
    
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()
    
    if not files_source:
        print(f"Error: No images found in data/{opt.test_data}")
        return

    # process data
    psnr_test = 0
    ssim_test = 0
    
    for f in files_source:
        # --- 图像读取和预处理 ---
        Img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        H_orig, W_orig = Img.shape 
        Img = normalize(np.float32(Img))
        
        # --- 填充图像至 DOWNSAMPLE_FACTOR 的倍数 ---
        
        H_pad_total = (H_orig + DOWNSAMPLE_FACTOR - 1) // DOWNSAMPLE_FACTOR * DOWNSAMPLE_FACTOR
        W_pad_total = (W_orig + DOWNSAMPLE_FACTOR - 1) // DOWNSAMPLE_FACTOR * DOWNSAMPLE_FACTOR
        
        pad_h = H_pad_total - H_orig
        pad_w = W_pad_total - W_orig
        
        # 将 Numpy 数组转换为 Jittor Tensor (B=1, C=1, H, W)
        Img = np.expand_dims(Img, 0) 
        Img = np.expand_dims(Img, 1) 
        # 注意：此处 ISource 仍然是 Jittor Tensor，但我们将在下一步使用 np.pad
        ISource_orig = jt.array(Img) 
        
        # *** 最终修正：使用 np.pad 进行填充 ***
        
        # 1. 提取 numpy 数组 (Jittor to Numpy)
        Img_np = ISource_orig.numpy()
        
        # 2. 定义填充参数 (只填充右侧和底部)
        # 格式: ((B_b, B_a), (C_b, C_a), (H_b, H_a), (W_b, W_a))
        # pad_h 和 pad_w 对应 H 和 W 维度的 "after" (尾部) 填充
        pad_params = ((0, 0), (0, 0), (0, pad_h), (0, pad_w))
        
        # 3. 使用 numpy 的 'edge' 模式进行复制边界填充
        Img_padded_np = np.pad(Img_np, pad_params, mode='edge')
        
        # 4. 转换回 Jittor Tensor
        ISource_padded = jt.array(Img_padded_np)
        
        # --- 噪声和模型输入准备 ---
        
        noise = jt.randn(ISource_padded.shape) * (opt.test_noiseL/255.)
        INoisy = ISource_padded + noise
        
        noise_map = jt.full_like(ISource_padded, opt.test_noiseL/255.)
        model_input = jt.concat([INoisy, noise_map], dim=1)

        # --- 推理 ---
        with jt.no_grad():
            predicted_noise = model(model_input)
            Out_padded = jt.clamp(INoisy - predicted_noise, 0., 1.)
        
        # --- 裁剪回原始尺寸 ---
        Out = Out_padded[..., :H_orig, :W_orig] 
        
        # --- 评估 ---
        current_psnr = batch_PSNR(Out, ISource_orig, 1.) # 评估使用未填充的原始图像
        current_ssim = batch_SSIM(Out, ISource_orig, 1.)
        
        psnr_test += current_psnr
        ssim_test += current_ssim
        print(f"{os.path.basename(f)} PSNR {current_psnr:.4f} SSIM {current_ssim:.4f}")
        
    # 计算平均值并打印
    num_files = len(files_source)
    ssim_test /= num_files
    psnr_test /= num_files
    print(f"\nAverage PSNR on {opt.test_data} (Noise={opt.test_noiseL}): {psnr_test:.4f}")
    print(f"Average SSIM on {opt.test_data} (Noise={opt.test_noiseL}): {ssim_test:.4f}")

if __name__ == "__main__":
    main()