import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter #
from models import UNet
from dataset import prepare_data, Dataset
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = UNet(channels=1)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(reduction='sum')
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
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
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            
            # 生成噪声和对应的噪声水平图
            noise_map = torch.zeros(img_train.size())
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
                noise_map.fill_(opt.noiseL/255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise_std = stdN[n]/255.
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=noise_std)
                    noise_map[n,:,:,:].fill_(noise_std)

            imgn_train = img_train + noise
            model_input = torch.cat((imgn_train, noise_map), 1)

            img_train, imgn_train = img_train.cuda(), imgn_train.cuda()
            noise = noise.cuda()
            model_input = model_input.cuda()
            out_train = model(model_input)
            loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            with torch.no_grad():
                out_train = torch.clamp(imgn_train-model(model_input), 0., 1.)
                psnr_train = batch_PSNR(out_train, img_train, 1.)
                ssim_train = batch_SSIM(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f SSIM_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train, ssim_train))
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
        with torch.no_grad():
            for k in range(len(dataset_val)):
                img_val = torch.unsqueeze(dataset_val[k], 0)
                noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
                imgn_val = img_val + noise
                # 为验证集创建噪声图和模型输入
                noise_map_val = torch.FloatTensor(img_val.size()).fill_(opt.val_noiseL/255.)
                model_input_val = torch.cat((imgn_val, noise_map_val), 1)

                img_val, imgn_val = img_val.cuda(), imgn_val.cuda()
                model_input_val = model_input_val.cuda()
                out_val = torch.clamp(imgn_val-model(model_input_val), 0., 1.)
                psnr_val += batch_PSNR(out_val, img_val, 1.)
                ssim_val += batch_SSIM(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        ssim_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f SSIM_val: %.4f" % (epoch+1, psnr_val, ssim_val))

        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        writer.add_scalar('SSIM on validation data', ssim_val, epoch)
        # log the images
        with torch.no_grad():
            out_train = torch.clamp(imgn_train-model(model_input), 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))

if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=64, stride=10, aug_times=1) # 增加到64，适配3层下采样
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=64, stride=10, aug_times=2) # 增加到64，适配3层下采样
    main()
