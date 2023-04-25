import os
import torch
import tqdm
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from config import opt
from torch.autograd import Variable
import os.path as osp


class GenImg(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.test = test
        self.train = train
        if test:
            self.pose_set = None
        elif train:
            self.pose_set = np.load(os.path.join(opt.train_pose_root, '0-10000.npy'))
            for i in range(1, opt.pose_num):
                name = os.path.join(opt.train_pose_root, '{}-{}.npy'.format(i * 10000, (i + 1) * 10000))
                pose = np.load(name)
                self.pose_set = np.concatenate((self.pose_set, pose))
            self.train_data_cpu = np.load(os.path.join(opt.train_data_root,'dataset_uint8_1.npy'))

            for i in tqdm.tqdm(range(2, opt.pose_num + 1)):
                name = os.path.join(opt.train_data_root,'dataset_uint8_{}.npy'.format(i))
                img = np.load(name)
                self.train_data_cpu = np.concatenate((self.train_data_cpu, img))
        else:
            self.pose_set = np.load(os.path.join(opt.train_pose_root, 'validation(640).npy'))
            self.train_data = np.load(os.path.join(opt.data_path, 'dataset_uint8(validation640).npy'))

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        if self.test:
            return None, None
        elif self.train:
            if index < 1000000:
                data = self.train_data_cpu[index, :, :].astype(np.float32) / 255.
            else:
                file_index1 = index // 10000
                file_index2 = (index % 10000) // 100
                file_index3 = ((index % 10000) % 100) % 100
                file_path = osp.join(opt.train_data_root, str(file_index1 + 1), str(file_index2), str(file_index3) + '.npy')
                data = np.load(file_path).astype(np.float32) / 255.
            label_p = self.pose_set[index, :]
            u, v = label_p[10:]
            uv = [u, v]
            label = np.concatenate((label_p[:6], uv), 0)
        else:
            data = self.train_data[index, :, :].astype(np.float32) / 255.
            label_p = self.pose_set[index, :]
            u, v = label_p[10:]
            uv = [u, v]
            label = np.concatenate((label_p[:6], uv), 0)
        return data, label

    def __len__(self):
        return np.shape(self.pose_set)[0]
