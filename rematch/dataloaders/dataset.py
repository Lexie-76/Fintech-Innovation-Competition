import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from mypath import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class VideoDataset(Dataset):
    #初始化参数
    def __init__(self, split='train', clip_len=16):
        #裁剪好的帧数据
        self.output_dir = Path.db_dir()
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split

        #修改图片大小
        self.resize_height = 480
        self.resize_width = 640
        self.crop_size = 300

        #获取文件名
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of {}: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        #读入数据
        buffer = self.load_frames(self.fnames[index]) #train 100/test 20 val 20
        #截16张图的窗口
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        # 数据扩充
        if self.split == 'test':
            buffer = self.randomflip(buffer)
        #色素通道均值化
        buffer = self.normalize(buffer)
        #转置，第四维度提前至第一维度
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)


    def randomflip(self, buffer):
        #图像水平翻转
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)
        return buffer

    #  RGB去均值化
    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame
        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        #读取每一张图片
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame
        return buffer

    def crop(self, buffer, clip_len, crop_size):
        #截取16帧图片窗口，随机第一帧
        time_index = np.random.randint(buffer.shape[0] - clip_len)

        #截取300*300的图片的初始位置
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        #buffer 16帧图片，每张图片300*300
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]
        return buffer


if __name__ == "__main__":
    train_data = VideoDataset(split='test', clip_len=16)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)
        if i == 1:
            break