import os
import argparse
import numpy as np
import jittor as jt
from jittor import nn
from jittor import optim
from jittor.dataset import DataLoader
from tensorboardX import SummaryWriter #
from models import UNet
from dataset import prepare_data, Dataset
from utils import *
import jittor.transform as transform

# Jittor自动管理GPU，无需手动设置CUDA环境变量
jt.flags.use_cuda = 1  # 启用GPU

parser = argparse.ArgumentParser(description="DnCNN")
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser.add_argument("--preprocess", type=str2bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=6, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
opt = parser.parse_args()

def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    # drop_last=False 确保处理所有数据，即使最后一个批次不完整。
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True, drop_last=False)
    # 验证集图像大小不同，批大小必须为1
    loader_val = DataLoader(dataset=dataset_val, num_workers=4, batch_size=1, shuffle=False, drop_last=False)
    # 使用向上取整除法，以在 drop_last=False 时获得正确的批次数。
    num_batches_train = (len(dataset_train) + opt.batchSize - 1) // opt.batchSize
    # 当批大小为1时，批次数等于样本数
    num_batches_val = len(dataset_val)

    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = UNet(channels=1)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss()  # Jittor默认为sum reduction
    # Jittor自动管理GPU，无需手动移动模型
    model = net
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    noiseL_B=[0,55] # ingnored when opt.mode=='S'
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            optimizer.zero_grad()
            img_train = data
            
            # 生成噪声和对应的噪声水平图
            noise_map = jt.zeros(img_train.shape)
            if opt.mode == 'S':
                noise = jt.randn(img_train.shape) * (opt.noiseL/255.)
                noise_map = jt.full_like(img_train, opt.noiseL/255.)
            if opt.mode == 'B':
                noise = jt.zeros(img_train.shape)
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.shape[0])
                for n in range(noise.shape[0]):
                    sizeN = noise[0,:,:,:].shape
                    noise_std = stdN[n]/255.
                    noise[n,:,:,:] = jt.randn(sizeN) * noise_std
                    noise_map[n,:,:,:] = noise_std

            imgn_train = img_train + noise
            model_input = jt.concat([imgn_train, noise_map], dim=1)

            # Jittor自动管理GPU，无需手动移动张量
            predicted_noise = model(model_input)
            loss = criterion(predicted_noise, noise) / (imgn_train.shape[0]*2)
            optimizer.step(loss)
            
            # results - 使用同一次前向传播的结果计算指标，避免重复计算
            out_train = jt.clamp(imgn_train - predicted_noise, 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            ssim_train = batch_SSIM(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f SSIM_train: %.4f" %
                (epoch+1, i+1, num_batches_train, loss.item(), psnr_train, ssim_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
                writer.add_scalar('SSIM on training data', ssim_train, step)
            step += 1
        ## the end of each epoch
        model.eval()
        # validate
        psnr_val = 0
        ssim_val = 0
        # 使用 DataLoader 和 no_grad() 进行高效验证
        with jt.no_grad():
            for i, img_val in enumerate(loader_val):
                noise = jt.randn(img_val.shape) * (opt.val_noiseL/255.)
                imgn_val = img_val + noise
                # 为验证集创建噪声图和模型输入
                noise_map_val = jt.full_like(img_val, opt.val_noiseL/255.)
                model_input_val = jt.concat([imgn_val, noise_map_val], dim=1)

                # Jittor自动管理GPU
                predicted_noise_val = model(model_input_val)
                out_val = jt.clamp(imgn_val - predicted_noise_val, 0., 1.)
                psnr_val += batch_PSNR(out_val, img_val, 1.)
                ssim_val += batch_SSIM(out_val, img_val, 1.)
        psnr_val /= num_batches_val
        ssim_val /= num_batches_val
        print("\n[epoch %d] PSNR_val: %.4f SSIM_val: %.4f" % (epoch+1, psnr_val, ssim_val))

        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        writer.add_scalar('SSIM on validation data', ssim_val, epoch)
        # save model
        jt.save(model.state_dict(), os.path.join(opt.outf, 'net.pkl'))

if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=64, stride=10, aug_times=1) # 增加到64，适配3层下采样
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=64, stride=10, aug_times=2) # 增加到64，适配3层下采样
    main()
