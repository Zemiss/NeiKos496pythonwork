import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from skimage import io
from skimage.transform import resize


from utils import data_augmentation

def normalize(data):
    return data/255.

# 分割函数：将原图片分割成数个小图像块 (已优化)
def Im2Patch(img, win, stride=1):
    """
    使用 numpy.lib.stride_tricks 高效地从图像中提取 patch
    """
    from numpy.lib.stride_tricks import as_strided
    C, H, W = img.shape

    # 计算输出形状
    out_H = (H - win) // stride + 1
    out_W = (W - win) // stride + 1

    s_C, s_H, s_W = img.strides
    # 为新的 patch 数组视图定义 strides
    new_strides = (s_C, s_H * stride, s_W * stride, s_H, s_W)
    patches = as_strided(img, shape=(C, out_H, out_W, win, win), strides=new_strides)
    # 重塑为期望的输出格式: (C, win, win, num_patches)
    return patches.transpose(0, 3, 4, 1, 2).reshape(C, win, win, -1)

#数据处理函数：训练数据——多尺度、多增强的小块  测试数据——完整图像
def prepare_data(data_path, patch_size, stride, aug_times=1):
    # train

    # Helper function to ensure image dimensions are divisible by a factor
    def ensure_divisibility(img_array, factor=16):
        # img_array can be (H, W, C) or (H, W) or (1, H, W)
        if img_array.ndim == 3 and img_array.shape[0] == 1: # (1, H, W)
            H, W = img_array.shape[1], img_array.shape[2]
            is_chw_format = True
        else: # (H, W, C) or (H, W)
            H, W = img_array.shape[0], img_array.shape[1]
            is_chw_format = False

        new_H = (H // factor) * factor
        new_W = (W // factor) * factor

        if new_H == 0 or new_W == 0: # Check for zero dimension after cropping
            raise ValueError(f"Image dimension ({H}x{W}) is too small to be divisible by {factor} and result in non-zero size.")

        if new_H != H or new_W != W:
            if is_chw_format: # (1, H, W)
                img_array = img_array[:, :new_H, :new_W]
            else: # (H, W, C) or (H, W)
                img_array = img_array[:new_H, :new_W, ...] # Use ... to handle C dimension if present
        return img_array

    print('process training data')
    scales = [1, 0.9, 0.8, 0.7]
    files = glob.glob(os.path.join(data_path, 'train', '*.png'))
    files.sort()
    h5f = h5py.File('train.h5', 'w')
    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])
        # Original image (H, W, C)
        # No need to ensure divisibility here, as it will be resized and then processed
        for k in range(len(scales)):
            # Resize first, then ensure divisibility
            resized_img = cv2.resize(img, (int(img.shape[1]*scales[k]), int(img.shape[0]*scales[k])), interpolation=cv2.INTER_CUBIC)
            processed_img = ensure_divisibility(resized_img, factor=16) # Ensure (H, W, C) is divisible
            processed_img = np.float32(normalize(np.expand_dims(processed_img[:,:,0], 0))) # Convert to (1, H, W) and normalize
            patches = Im2Patch(processed_img, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:,:,:,n].copy()
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1,8))
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    train_num += 1
    h5f.close()
    # val
    print('\nprocess validation data')
    files.clear()
    files = glob.glob(os.path.join(data_path, 'Set12', '*.png'))
    files.sort()
    h5f = h5py.File('val.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])

        img = cv2.imread(files[i]) # (H, W, C)
        processed_img = ensure_divisibility(img, factor=16) # Ensure (H, W, C) is divisible
        processed_img = np.float32(normalize(np.expand_dims(processed_img[:,:,0], 0))) # Convert to (1, H, W) and normalize
        h5f.create_dataset(str(val_num), data=processed_img)
        val_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)

class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        self.h5f = None  # 文件句柄将在 __getitem__ 中为每个 worker 初始化，以保证多进程安全
        if self.train:
            self.h5_path = 'train.h5'
        else:
            self.h5_path = 'val.h5'
        
        # 在初始化时一次性读取所有 keys
        with h5py.File(self.h5_path, 'r') as h5f:
            self.keys = list(h5f.keys())
            random.shuffle(self.keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # 在多进程数据加载 (num_workers > 0) 场景下，__init__ 在主进程中调用，
        # 而 __getitem__ 在 worker 进程中调用。h5py 文件对象不可序列化 (pickle)，
        # 因此不能在 __init__ 中打开并在 __getitem__ 中使用。
        # 常见的模式是在 worker 进程中首次调用时打开文件，并缓存文件句柄。
        if self.h5f is None:
            self.h5f = h5py.File(self.h5_path, 'r')

        key = self.keys[index]
        data = np.array(self.h5f[key])
        return torch.Tensor(data)
