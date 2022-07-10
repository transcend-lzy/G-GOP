import copy
import os

import numpy
import numpy as np
import os.path as osp
import cv2
import torch
from config import opt
import sys
from tqdm import tqdm
from utils import *


def create_new_pose(pose_set):
    for i in range(len(pose_set)):
        new_pose = min_max(pose_set[i])
        pose_set[i] = new_pose
    return pose_set


def min_max(pose):
    pose_new = np.zeros((12))
    pose_new[0] = (pose[0] - amin) / (amax - amin)
    pose_new[1] = (pose[1] - bmin) / (bmax - bmin)
    pose_new[2] = (pose[2] - gmin) / (gmax - gmin)
    pose_new[3] = (pose[3] - xmin) / (xmax - xmin)
    pose_new[4] = (pose[4] - ymin) / (ymax - ymin)
    pose_new[5] = (pose[5] - rmin) / (rmax - rmin)
    for i in range(6,12):
        pose_new[i] = pose[i]
    return pose_new


if __name__ == '__main__':
    pose_new1 = np.load('../data/6/pose_set_new/0-10000.npy')
    pose_new2 = np.load('../data/6/pose_set_new/validation(640).npy')
    pose_set = np.load(os.path.join(opt.train_pose_root, '0-10000.npy'))
    pose_set_new = '../data/6/pose_set_new'
    mkdir(pose_set_new)
    Arange = ranges['Arange'][0]
    Brange = ranges['Brange'][0]
    Grange = ranges['Grange'][0]
    Xrange = ranges['Xrange'][0]
    Zrange = ranges['Yrange'][0]
    Rrange = ranges['Rrange'][0]
    amin, amax = Arange[0] - 0.3, Arange[1] + 0.3
    bmin, bmax = Brange[0] - 0.3, Brange[1] + 0.3
    gmin, gmax = Grange[0] - 0.3, Grange[1] + 0.3
    xmin, xmax = Xrange[0] - 0.1, Xrange[1] + 0.1
    ymin, ymax = Zrange[0] - 0.1, Zrange[1] + 0.1
    rmin, rmax = Rrange[0] - 0.05, Rrange[1] + 0.05
    name = os.path.join(opt.train_pose_root, 'validation(640).npy')
    name_new = os.path.join(pose_set_new, 'validation(640).npy')
    pose = np.load(name)
    pose_new = np.zeros((len(pose), 12))
    for i in range(len(pose)):
        pose_new[i] = min_max(copy.deepcopy(pose[i]))
    np.save(name_new, pose_new)
    for i in tqdm(range(opt.pose_num)):
        name = os.path.join(opt.train_pose_root, '{}-{}.npy'.format(i * 10000, (i + 1) * 10000))
        name_new = os.path.join(pose_set_new, '{}-{}.npy'.format(i * 10000, (i + 1) * 10000))
        pose = np.load(name)
        # pose_new = np.load(name_new)
        pose_new = np.zeros((len(pose), 12))
        for j in range(len(pose)):
            pose_new[j] = min_max(copy.deepcopy(pose[j]))
        np.save(name_new, pose_new)








    # for i in tqdm(range(102, opt.pose_num + 1)):
    #     name = os.path.join(opt.train_data_root, 'dataset_uint8_{}.npy'.format(i))

    # for i in tqdm(range(2, opt.pose_num + 1)):
    #     name = os.path.join(opt.train_data_root, 'dataset_uint8_{}.npy'.format(i))
    #     pose = np.load(name)
    #     pose_set = np.concatenate((pose_set, pose))
    # train_new_data = osp.join(opt.data_path, 'train_data_new')
    # for i in tqdm(range(100)):
    #     if not osp.exists(osp.join(train_new_data, str(i))):
    #         os.mkdir(osp.join(train_new_data, str(i)))
    #     for j in range(100):
    #         if not osp.exists(osp.join(train_new_data, str(i), str(j))):
    #             os.mkdir(osp.join(train_new_data, str(i), str(j)))
    #         for k in range(100):
    #             np.save(osp.join(train_new_data, str(i), str(j), str(k) + '.npy'),
    #                     pose_set[i * 10000 + j * 100 + k])






