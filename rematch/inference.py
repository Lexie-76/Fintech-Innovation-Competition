import os

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from dataloaders.dataset import VideoDataset
from network import C3D_model
import cv2
torch.backends.cudnn.benchmark = True
from mypath import Path

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("Device being used:", device)

    with open('./dataloaders/label.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    # 模型初始化
    model = C3D_model.C3D(num_classes=10)
    # 加载模型
    checkpoint = torch.load('D:\\aaa\\run\\run_10\\models\\C3D-49.pth.tar', map_location=lambda storage, loc: storage)
    """
    state_dict = model.state_dict()
    for k1, k2 in zip(state_dict.keys(), checkpoint.keys()):
        state_dict[k1] = checkpoint[k2]
    model.load_state_dict(state_dict)
    """
    model.load_state_dict(checkpoint['state_dict'])
    
    model.to(device)
    model.eval()

    resize_height = 480
    resize_width = 640
    retaining = True
    clip = []
    # 取文件夹 共6个
    fnames = []
    sample_dir = Path.test_dir()
    for fname in os.listdir(sample_dir):
        fnames.append(os.path.join(sample_dir, fname))
        tag = 0

    while tag < 6:
        curr_fname = fnames[tag]
        path_list = os.listdir(curr_fname)
        path_list.sort(key=lambda x:int(x.split("Img")[1].split('.')[0]))
        #按顺序取文件夹中的图片
        frames = [os.path.join(curr_fname, img) for img in path_list]
        #图片处理
        buffer = np.empty((len(frames), resize_height, resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame
        crop_size = 200
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)
        buffer = buffer[:, height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame
        buffer = np.expand_dims(buffer, axis=0)
        buffer = buffer.transpose(0, 4, 1, 2, 3)
        inputs = torch.from_numpy(buffer)
        #模型预测
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        probs = torch.nn.Softmax(dim=1)(outputs)
        label = torch.max(probs, 1)[1]
        tag+=1




if __name__ == '__main__':
    main()









