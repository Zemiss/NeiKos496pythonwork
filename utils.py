import math
import jittor as jt
from jittor import nn
import numpy as np
# 导入 skimage 库中的峰值信噪比 (PSNR) 计算函数
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
from jittor import init

#初始权重函数
def weights_init_kaiming(m):
    # 使用 isinstance 来精确判断模块类型，避免对容器模块（如 DoubleConv）错误操作
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d):
        init.gauss_(m.weight, 1.0, 0.02) # 通常将 weight 初始化为1
        init.constant_(m.bias, 0.0)

#PSNR计算函数
def batch_PSNR(img, imclean, data_range):
    Img = img.numpy().astype(np.float32)
    Iclean = imclean.numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

#SSIM计算函数
def batch_SSIM(img, imclean, data_range):
    Img = img.numpy().astype(np.float32)
    Iclean = imclean.numpy().astype(np.float32)
    SSIM = 0
    for i in range(Img.shape[0]):
        # 对于单通道灰度图，我们需要移除通道维度
        SSIM += compare_ssim(Iclean[i,0,:,:], Img[i,0,:,:], data_range=data_range)
    return (SSIM/Img.shape[0])

#数据增强函数：对图像进行旋转、翻转
def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))
