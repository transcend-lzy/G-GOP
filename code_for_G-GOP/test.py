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
from ruamel import yaml
import random


def update_bbox(xywh):
    x, y, w, h = xywh[0] + random.randint(-8, 0), xywh[1] + random.randint(-8, 0), \
                 xywh[2] + random.randint(0, 10), xywh[3] + random.randint(0, 10)
    return [x, y, w, h]


if __name__ == '__main__':
    bbox = read_yaml(osp.join(opt.test_data_root, 'video_data', 'bbox.yml'))
    bbox_new = {}
    obj_ids = [6, 22]
    for i in range(1, 216):
        bbox_single_new = []
        for j in range(2):
            xywh = bbox[str(i)][j]['xywh']
            bbox_single_new.append({'obj_id': obj_ids[j], 'xywh': update_bbox(xywh)})
        bbox_new[str(i)] = copy.deepcopy(bbox_single_new)

    yaml_file = osp.join('./', 'ssdResult.yml')
    file = open(yaml_file, 'w', encoding='utf-8')
    yaml.dump(bbox_new, file, Dumper=yaml.RoundTripDumper)
    file.close()

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
