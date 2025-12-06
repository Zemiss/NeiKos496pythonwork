# DnCNN-PyTorch NeiKos496小组作业

这是 TIP2017 论文 Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising 的 PyTorch 实现(http://ieeexplore.ieee.org/document/7839189/)。

作者的 MATLAB 实现请见此处(https://github.com/cszn/DnCNN).

原代码是使用 PyTorch < 0.4 编写的，本次作业对低版本语法进行了优化使得适应 PyTorch 2.5.1

如何运行

1. 依赖项 
python 2.9.23
PyTorch(2.5.1)
torchvision
OpenCV for Python
HDF5 for Python
tensorboardX (PyTorch 的 TensorBoard 可视化工具)

2.训练 DnCNN-S/B (已知噪声水平的 DnCNN)

conda activate myenv
cd /home/xie/zzzmypython/DnCNNunet

第一次加上预训练
--preprocess True 

后续训练：
nohup python train.py --preprocess False --num_of_layers 17 --mode S --noiseL 15 --val_noiseL 15 --outf logs/DnCNN-S-15 > logs/DnCNN-S-15.log 2>&1 &

nohup python train.py --preprocess False --num_of_layers 17 --mode S --noiseL 25 --val_noiseL 25 --outf logs/DnCNN-S-25  > logs/DnCNN-S-25.log 2>&1 &

nohup python train.py --preprocess False --num_of_layers 17 --mode S --noiseL 50 --val_noiseL 50 --outf logs/DnCNN-S-50 > logs/DnCNN-S-50.log 2>&1 &

nohup python train.py --preprocess True --num_of_layers 20 --mode B --val_noiseL 25 --outf logs/DnCNN-B  > logs/DnCNN-B.log 2>&1 &

测试set68
python test.py --num_of_layers 17 --logdir logs/DnCNN-S-15 --test_data "Set68" --test_noiseL 15
python test.py --num_of_layers 17 --logdir logs/DnCNN-S-25 --test_data "Set68" --test_noiseL 25
python test.py --num_of_layers 17 --logdir logs/DnCNN-S-50 --test_data "Set68" --test_noiseL 50

python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set68" --test_noiseL 15
python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set68" --test_noiseL 25
python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set68" --test_noiseL 50

python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set68" --test_noiseL 75
python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set68" --test_noiseL 100

### BSD68 平均 PSNR
|:-----------:|:-------:|:-------:|
| Noise Level | DnCNN-S | DnCNN-B |
|:-----------:|:-------:|:-------:|
|     15      |  31.73  |  31.61  |
|     25      |  29.23  |  29.16  | 
|     50      |  26.23  |  26.23  | 
|:-----------:|:-------:|:-------:|

|:-----------:|:---------------:|:---------------:|:---------------:|:---------------:|
| Noise Level | DnCNN-S-PyTorch | DnCNN-B-PyTorch | DnCNN-S-ours    | DnCNN-B-ours    |
|:-----------:|:---------------:|:---------------:|:---------------:|:---------------:|
|     15      |      31.71      |      31.60      |      31.70      |     31.62       |
|     25      |      29.21      |      29.15      |      29.17      |     29.16       |
|     50      |      26.22      |      26.20      |      26.16      |     26.20       |
|:-----------:|:---------------:|:---------------:|:---------------:|:---------------:|

### BSD68 本次作业修改后的得分(1为使用u-net框架并借鉴了ffdnet)

|:-----------:|:---------------:|:---------------:|
| Noise Level | DnCNN-B         | DnCNN-B-1       |
|:-----------:|:---------------:|:---------------:|
|     50      |      26.20      |       26.11     |            
|     75      |      17.89      |       24.35     |            
|    100      |      13.65      |       22.91     |            
|:-----------:|:---------------:|:---------------:|

### BSD68 平均 SSIM
|:-----------:|:---------------:|:---------------:|:---------------:|:---------------:|
| Noise Level | DnCNN-S-PyTorch | DnCNN-B-PyTorch | DnCNN-S-ours    | DnCNN-B-ours    |
|:-----------:|:---------------:|:---------------:|:---------------:|:---------------:|
|     15      |      0.895      |      0.891      |      0.895      |      0.891      |
|     25      |      0.833      |      0.827      |      0.832      |      0.829      |
|     50      |      0.719      |      0.714      |      0.719      |      0.715      |
|:-----------:|:---------------:|:---------------:|:---------------:|:---------------:|

|:-----------:|:---------------:|:---------------:|
| Noise Level | DnCNN-B         | DnCNN-B-1       |
|:-----------:|:---------------:|:---------------:|
|     50      |      0.715      |       0.713     |            
|     75      |      0.294      |       0.614     |            
|    100      |      0.160      |       0.503     |            
|:-----------:|:---------------:|:---------------:|

测试set12
python test.py --num_of_layers 17 --logdir logs/DnCNN-S-15 --test_data "Set12" --test_noiseL 15
python test.py --num_of_layers 17 --logdir logs/DnCNN-S-25 --test_data "Set12" --test_noiseL 25
python test.py --num_of_layers 17 --logdir logs/DnCNN-S-50 --test_data "Set12" --test_noiseL 50

python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set12" --test_noiseL 15
python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set12" --test_noiseL 25
python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set12" --test_noiseL 50

python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set12" --test_noiseL 75
python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set12" --test_noiseL 100

### Set12 平均 PSNR
|:-----------:|:-------:|:-------:|
| Noise Level | DnCNN-S | DnCNN-B | 
|:-----------:|:-------:|:-------:|
|     15      | 32.859  | 32.680  |   
|     25      | 30.436  | 30.362  |  
|     50      | 27.178  | 27.206  | 
|:-----------:|:-------:|:-------:|

|:-----------:|:---------------:|:---------------:|:---------------:|:---------------:|
| Noise Level | DnCNN-S-PyTorch | DnCNN-B-PyTorch | DnCNN-S-ours    | DnCNN-B-ours    |
|:-----------:|:---------------:|:---------------:|:---------------:|:---------------:|
|     15      |     32.837      |     32.725      |     32.811      |     32.731      |
|     25      |     30.404      |     30.344      |     30.349      |     30.376      |
|     50      |     27.165      |     27.138      |     27.057      |     27.132      |
|:-----------:|:---------------:|:---------------:|:---------------:|:---------------:|

|:-----------:|:---------------:|:---------------:|
| Noise Level | DnCNN-B         | DnCNN-B-1       |
|:-----------:|:---------------:|:---------------:|
|     50      |      27.132     |       26.846    |            
|     75      |      18.113     |       24.749    |            
|    100      |      13.895     |       22.996    |            
|:-----------:|:---------------:|:---------------:|

### Set12 平均 SSIM
|:-----------:|:---------------:|:---------------:|:---------------:|:---------------:|
| Noise Level | DnCNN-S-PyTorch | DnCNN-B-PyTorch | DnCNN-S-ours    | DnCNN-B-ours    |
|:-----------:|:---------------:|:---------------:|:---------------:|:---------------:|
|     15      |     0.904       |     0.902       |     0.904       |     0.903       |
|     25      |     0.862       |     0.859       |     0.861       |     0.862       |
|     50      |     0.779       |     0.773       |     0.779       |     0.777       |
|:-----------:|:---------------:|:---------------:|:---------------:|:---------------:|


|:-----------:|:---------------:|:---------------:|
| Noise Level | DnCNN-B         | DnCNN-B-1       |
|:-----------:|:---------------:|:---------------:|
|     50      |      0.777      |      0.765      |            
|     75      |      0.290      |      0.668      |            
|    100      |      0.163      |      0.549      |            
|:-----------:|:---------------:|:---------------:|

tensorboard --logdir=C:\Users\12445\Desktop\DnCNNpytorch\logs\DnCNN-S-15
tensorboard --logdir=C:\Users\12445\Desktop\DnCNNpytorch\logs\DnCNN-S-25
tensorboard --logdir=C:\Users\12445\Desktop\DnCNNpytorch\logs\DnCNN-S-50
tensorboard --logdir=C:\Users\12445\Desktop\DnCNNpytorch\logs\DnCNN-B