import os
import os.path as osp
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from math import pi, cos, sin, sqrt, asin, atan
from config import opt
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import yaml


def read_yaml(yaml_path):
    f = open(yaml_path, 'r', encoding='utf-8')
    cfg = f.read()
    dic = yaml.safe_load(cfg)
    return dic


spe_points = read_yaml(osp.join(opt.data_path_root, 'special_points.yml'))[str(opt.obj_id)]
ranges = read_yaml(osp.join(opt.data_path_root, 'obj_scene_pose_range.yml'))[str(opt.obj_id)]
points = np.load(osp.join(opt.data_path_root, 'fibonacci.npy'))
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

def min_max(pose):
    pose_new = np.zeros(6)
    pose_new[0] = (pose[0] - amin) / (amax - amin)
    pose_new[1] = (pose[1] - bmin) / (bmax - bmin)
    pose_new[2] = (pose[2] - gmin) / (gmax - gmin)
    pose_new[3] = (pose[3] - xmin) / (xmax - xmin)
    pose_new[4] = (pose[4] - ymin) / (ymax - ymin)
    pose_new[5] = (pose[5] - rmin) / (rmax - rmin)
    return pose_new


def min_max_rollback(pose_new):
    pose = np.zeros(6)
    pose[0] = pose_new[0] * (amax - amin) + amin
    pose[1] = pose_new[1] * (bmax - bmin) + bmin
    pose[2] = pose_new[2] * (gmax - gmin) + gmin
    pose[3] = pose_new[3] * (xmax - xmin) + xmin
    pose[4] = pose_new[4] * (ymax - ymin) + ymin
    pose[5] = pose_new[5] * (rmax - rmin) + rmin
    return pose


def mkdir(path):
    if not osp.exists(path):
        os.makedirs(path)


def clip_and_scaling(img, xywh, bbox_range, mask=None, mask_small=None):
    b, g, r = cv2.split(img)

    b1 = cv2.equalizeHist(b)
    g1 = cv2.equalizeHist(g)
    r1 = cv2.equalizeHist(r)

    img = cv2.merge([b1, g1, r1]).astype(np.uint8)
    mask_ori = cv2.imread('../data/test_2448/scene23/mask/6_6.png', 0)
    if mask is not None:
        maskCopy = np.array(mask).astype(np.uint8)
        # show_photos([mask_ori, maskCopy])
        np.place(maskCopy, maskCopy > 0, 255)
        img = cv2.bitwise_and(img, img, mask=maskCopy)
        # show_photos([img, maskCopy])
    x, y, w, h = xywh[0], xywh[1], xywh[2], xywh[3]
    img = img[y: y + h, x: x + w, :]

    canny_img = cv2.Canny(img, 200, 300)

    # show_photos([canny_img])
    kernel = np.ones((2, 2), np.uint8)
    canny_img = cv2.dilate(canny_img, kernel, iterations=1)
    # show_photos([canny_img])
    canny_img = cv2.morphologyEx(canny_img, cv2.MORPH_OPEN, kernel)
    border_top_bottom = bbox_range - h
    if border_top_bottom % 2 == 0:
        top = bottom = border_top_bottom // 2
    else:
        top = border_top_bottom // 2 + 1
        bottom = border_top_bottom // 2
    y -= top
    border_left_right = bbox_range - w
    if border_left_right % 2 == 0:
        left = right = border_left_right // 2
    else:
        left = border_left_right // 2 + 1
        right = border_left_right // 2
    x -= left
    constant = cv2.copyMakeBorder(canny_img, top, bottom, left, right, cv2.BORDER_CONSTANT)
    if mask_small is not None:
        mask_small_copy = np.array(mask_small).astype(np.uint8)
        np.place(mask_small_copy, mask_small_copy > 0, 255)
        kernel = np.ones((10, 10), np.uint8)
        erosion = cv2.erode(mask_small_copy, kernel, iterations=1)
        constant = cv2.bitwise_and(constant, constant, mask=erosion)
        # show_photos([mask_small_copy, erosion, constant])
    res = cv2.resize(constant, (128, 128), interpolation=cv2.INTER_AREA)

    # show_photos([canny_img, constant, res])
    return x, y, res


def show_photos(photos):  # 展示照片
    for index, photo in enumerate(photos):
        if (photo.shape[0] > 1000):
            photo = cv2.resize(photo, (int(2448.0 // 2), int(2048.0 // 2)))
        cv2.imshow(str(index), photo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def init():
    # 创建各种文件夹
    path_list = [opt.main_path, opt.val_sample_path, opt.test_sample_path,
                 opt.data_path, opt.log_path, opt.log_train, opt.log_test]
    for i in path_list:
        if not os.path.exists(i):
            mkdir(i)


def compute_loss(x_res, x):  # 输入z是8维输入特征,x是训练图,输出由z经model的生成图像和x之间的loss
    cross_ent = F.binary_cross_entropy(x_res, x)
    rec_loss = torch.sum(cross_ent) / x.shape[0]

    return torch.mean(rec_loss)


def im_save(im, index, path=opt.val_sample_path):  # 保存图像
    im = im.cpu().detach().numpy()
    im = np.reshape(im, [-1, 128, 128])  # -1就是64,一个batchsize是64
    if np.max(im) <= 1:
        im = im * 255
    im = im.astype(np.uint8)
    if np.shape(im)[-1] != 3:
        sample_im = Image.new('L', (128 * 8, 128 * 8))
        for i, x in enumerate(range(0, 128 * 8, 128)):
            for j, y in enumerate(range(0, 128 * 8, 128)):
                I1 = im[i * 8 + j, :, :]
                I1 = np.reshape(I1, [-1, 128])
                I1 = Image.fromarray(I1, mode='L')
                sample_im.paste(I1, (x, y))

        sample_im.save(os.path.join(path, 'sample{}.png'.format(index)))
    else:
        cv2.imwrite(os.path.join(path, 'sample{}.png'.format(index)), im)


def show_example_im_double(im, im2):  # 为了生成 输入图像和生成图像的对比图   左边是target右边是生成图
    im2 = im2.cpu().detach().numpy()[:64, :, :]
    im2 = np.reshape(im2, [-1, 128, 128])  # -1就是64,一个batchsize是64
    im = im.cpu().detach().numpy()[:64, :, :]
    if np.max(im) <= 1:
        im = im * 255
    if np.max(im2) <= 1:
        im2 = im2 * 255

    im = im.astype(np.uint8)
    im2 = im2.astype(np.uint8)

    new_im = Image.new('L', (128 * 8 * 2, 128 * 8))

    for i, x in enumerate(range(0, 128 * 8 * 2, 128 * 2)):
        for j, y in enumerate(range(0, 128 * 8, 128)):
            I = im[i * 8 + j, :, :]
            I[-5:-1, :] = 255
            I = Image.fromarray(I, mode='L')
            new_im.paste(I, (x, y))

    for i, x in enumerate(range(0, 128 * 8 * 2, 128 * 2)):
        for j, y in enumerate(range(0, 128 * 8, 128)):
            I = im2[i * 8 + j, :, :]
            I[-5:-1, :] = 255
            I[:, -5:-1] = 255
            I = Image.fromarray(I, mode='L')
            new_im.paste(I, (x + 128, y))
    print(new_im)
    fig = plt.figure(figsize=(12, 6), dpi=128 * 8 / 6.)
    plt.imshow(np.asarray(new_im), cmap='gray')
    plt.axis('off')
    plt.show()


def save_example_im_double(im, im2, index, is_val=False, is_test=False):  # 为了生成 输入图像和生成图像的对比图   左边是target右边是生成图
    im2 = im2.cpu().detach().numpy()[:64, :, :]
    im2 = np.reshape(im2, [-1, 128, 128])  # -1就是64,一个batchsize是64
    if (type(im) != np.ndarray): im = im.cpu().detach().numpy()[:64, :, :]
    if np.max(im) <= 1:
        im = im * 255
    if np.max(im2) <= 1:
        im2 = im2 * 255

    im = im.astype(np.uint8)
    im2 = im2.astype(np.uint8)

    new_im = Image.new('L', (128 * 8 * 2, 128 * 8))

    for i, x in enumerate(range(0, 128 * 8 * 2, 128 * 2)):
        for j, y in enumerate(range(0, 128 * 8, 128)):
            I = im[i * 8 + j, :, :]
            I[-5:-1, :] = 255
            I = Image.fromarray(I, mode='L')
            new_im.paste(I, (x, y))

    for i, x in enumerate(range(0, 128 * 8 * 2, 128 * 2)):
        for j, y in enumerate(range(0, 128 * 8, 128)):
            I = im2[i * 8 + j, :, :]
            I[-5:-1, :] = 255
            I[:, -5:-1] = 255
            I = Image.fromarray(I, mode='L')
            new_im.paste(I, (x + 128, y))
    if is_val:
        new_im.save(os.path.join(opt.val_sample_path, 'sample{}_val.png'.format(index)))
    elif is_test:
        new_im.save(os.path.join(opt.test_sample_path, 'sample{}.png'.format(index)))
    else:
        new_im.save(os.path.join(opt.val_sample_path, 'sample{}.png'.format(index)))


def estimate_3D_to_2D(a, b, g, x_trans, z_trans, r, points):
    r_x = [[1, 0, 0], [0, cos(a), -sin(a)], [0, sin(a), cos(a)]]
    r_y = [[cos(b), 0, sin(b)], [0, 1, 0], [-sin(b), 0, cos(b)]]
    r_z = [[cos(g), -sin(g), 0], [sin(g), cos(g), 0], [0, 0, 1]]

    bag = np.matmul(np.matmul(r_y, r_x), r_z)  # np.matmul是做矩阵相乘

    rm = bag

    xx = np.array([rm[0, 0], rm[1, 0], rm[2, 0]])
    yy = np.array([rm[0, 1], rm[1, 1], rm[2, 1]])
    zz = np.array([rm[0, 2], rm[1, 2], rm[2, 2]])

    x = r * -zz[0]
    y = r * -zz[1]
    z = r * -zz[2]

    pos = np.array([x, y, z]) + x_trans * xx + z_trans * yy  # T 3×1

    worldOrientation = rm.T  # 旋转矩阵
    worldLocation = pos * 1000  # 平移矩阵

    rotationMatrix = worldOrientation.T
    translationVector = -np.matmul(worldLocation, worldOrientation.T)

    # np.matmul 是矩阵相乘   np.tile
    a = np.matmul(points, rotationMatrix) + np.tile(translationVector, [np.size(points, 0), 1])

    u = opt.ox - opt.fx * a[:, 0] / a[:, 2]  # 像素坐标系坐标
    v = opt.oy - opt.fy * a[:, 1] / a[:, 2]

    results = np.array([u, v]).T

    return results


def val_pose(a, b, g, x, z, r, u_tar, v_tar, is_create_val=False, spe_points2D=0):
    if not is_create_val:
        spe_points2D = estimate_3D_to_2D(a, b, g, x, z, r, spe_points)  # 把spe_points 全部转换为了2d图像坐标，左上角为（0，0）

    for k in range(np.size(spe_points2D, 0)):
        u = spe_points2D[k, 0]
        v = spe_points2D[k, 1]

        if u < u_tar or u >= u_tar + opt.bbox_len or v < v_tar or v >= v_tar + opt.bbox_len:
            return False
    return True


def val_xyr(x, y, r):
    if x > xmax or x < xmin:
        return False
    if y > ymax or y < ymin:
        return False
    if r > rmax or r < rmin:
        return False
    return True


def random_gen():
    g = np.random.uniform(gmin, gmax)
    x = np.random.uniform(xmin, xmax)
    y = np.random.uniform(ymin, ymax)
    r = np.random.uniform(rmin, rmax)
    return g, x, y, r


def val_ab(a, b):
    flagA = False
    flagB = False
    if amin <= a <= amax:
        flagA = True
    if not flagA:
        return False
    if bmin <= b <= bmax:
        flagB = True
    return flagB


def creatPose(u_tar, v_tar):
    try_times = 1
    while True:
        while True:  # 这里while true做的就是把xyz转化为了ab
            index = np.random.randint(len(points))  # sample_num是斐波那契球的全部数量

            x = points[index][0]
            y = points[index][1]
            z = points[index][2]

            beta = asin(y)
            alpha = 0

            if beta > pi / 2:
                beta = pi - beta

            L = abs(cos(beta))

            if x > 0 and z > 0:
                alpha = asin(z / L)
            elif x < 0 < z:
                alpha = pi - asin(z / L)
            elif x < 0 and z < 0:
                alpha = pi - asin(z / L)
            elif x > 0 > z:
                alpha = 2 * pi + asin(z / L)
            elif x > 0 and z == 0:
                alpha = 0
            elif x == 0 and z > 0:
                alpha = pi / 2
            elif x < 0 and z == 0:
                alpha = pi
            elif x == 0 and z < 0:
                alpha = pi * 3 / 2.

            if val_ab(alpha, beta):
                a = alpha
                b = beta
                break

        g, x, y, r = random_gen()

        trans = [(0, 0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1)]
        spe_points2D = estimate_3D_to_2D(a, b, g, x, z, r, spe_points)
        in_window = True
        for k in range(np.size(spe_points2D, 0)):
            u = spe_points2D[k, 0]
            v = spe_points2D[k, 1]

            if u < 0 or u >= opt.height or v < 0 or v >= opt.width:
                in_window = False
                break
        if not in_window:
            continue
        center = estimate_3D_to_2D(a, b, g, x, z, r, [[0, 0, 0]])
        center = center[0]
        # 零件正好在矩形框中心，这时候的矩形框中心点像素坐标
        center_x, center_y = int(np.round(center[0])), int(np.round(center[1]))
        if center_x <= opt.bbox_len // 2 or center_x >= opt.width - opt.bbox_len // 2 or center_y <= opt.bbox_len // 2 or center_y >= opt.height - opt.bbox_len // 2:
            continue
        return [a, b, g, x, y, r]

        # for i in range(try_times):
        #     x_new, y_new, r_new = 0, 0, 0
        #     for j in trans:
        #         x_new = x + j[0] * i * 2
        #         y_new = z + j[1] * i * 2
        #         r_new = r + j[1] * i * 4
        #         if val_pose(a, b, g, x_new, y_new, r_new, u_tar, v_tar, is_create_val=True, spe_points2D=spe_points2D):
        #             return [a, b, g, x_new, y_new, r_new]
        #     if not val_xyr(x_new, y_new, r_new):
        #         break
