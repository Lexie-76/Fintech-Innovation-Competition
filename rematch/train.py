import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset
from network import C3D_model

nEpochs = 101  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True # See evolution of the test set when training
nTestInterval = 20 # Run on test set every nTestInterval epochs
snapshot = 25 # Store a model every snapshot epochs
lr = 1e-5 # Learning rate
num_classes = 10 # 10类
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#保存文件的绝对路径
save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
#以斜杠为分隔符，保留后一部分
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

#构造模型保存文件夹
if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))

modelName = 'C3D'
saveName = modelName

def train_model(save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):

    model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
    train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                    {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]

    #交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    #随机梯度下降算法优化器  momentum动量因子  weight_decay权重衰退
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    #有序调整学习率 等间隔调整学习率，调整倍数为gamma倍，调整间隔为step_size，单位epoch。
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '-' + str(resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)
    #可视化输出
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    #取数据
    train_dataloader = DataLoader(VideoDataset(split='train',clip_len=16), batch_size=12, shuffle=True, num_workers=4)
    val_dataloader   = DataLoader(VideoDataset(split='val',  clip_len=16), batch_size=12, num_workers=4)
    test_dataloader  = DataLoader(VideoDataset(split='test', clip_len=16), batch_size=12, num_workers=4)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    for epoch in range(resume_epoch, num_epochs):
        for phase in ['train', 'val']:
            #运行时间
            start_time = timeit.default_timer()

            #重置损失和准确率
            running_loss = 0.0
            running_corrects = 0.0

            if phase == 'train':
                scheduler.step()
                # 启用BN(Batch Normalization)层或Dropout层
                # 保证BN层能够用到每一批数据的均值和方差；Dropout随机取一部分网络连接来训练更新参数
                model.train()
            else:
                # 不启用Batch Normalization和Dropout，利用到了所有网络连接，不进行随机舍弃神经元
                model.eval()
            #输入图片和对应标签
            for inputs, labels in tqdm(trainval_loaders[phase]):
                # requires_grad在计算中保留对应的梯度信息，反向传播时会自动求导
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                #梯度置0
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    #requires_grad=False，反向传播不会求导
                    with torch.no_grad():
                        outputs = model(inputs)

                #softmax得到所给矩阵的概率分布，在softmax操作之后在dim这个维度相加等于1
                probs = nn.Softmax(dim=1)(outputs)

                #按行取最大值
                preds = torch.max(probs, 1)[1]

                #求loss
                loss = criterion(outputs, labels.long())

                if phase == 'train':
                    # 反向传播，计算当前梯度
                    loss.backward()
                    # 根据梯度更新网络参数
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            # 数据保存在文件里面供可视化使用
            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels.long())

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()


if __name__ == "__main__":
    train_model()