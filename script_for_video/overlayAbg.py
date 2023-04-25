import os

import cv2
import numpy as np
from math import pi
from overlayUtils import *
from trans_pose import *


class OverLay():
    def __init__(self, obj_id):
        self.obj_id = str(obj_id)
        self.height = 2048
        self.width = 2448
        self.abgxyr = [0, 0, 0, 0, 0, 0]
        self.cad_path = osp.join('CADmodels\\stl', str(self.obj_id) + '.stl')  # stl文件
        self.save_path = '.\\vis\\abg.jpg'
        self.K = [[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]

    def init(self, k, width, height):
        pygame.init()
        display = (width, height)
        window = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        scale = 0.0001
        fx = k[0][2]  # 相机标定矩阵的值
        fy = k[1][2]
        cx = k[0][0]
        cy = k[1][1]
        glFrustum(-fx * scale, (width - fx) * scale, -(height - fy) * scale, fy * scale,
                  (cx + cy) / 2 * scale, 20)  # 透视投影
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)  # 设置深度测试函数
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SMOOTH)
        glPolygonMode(GL_FRONT, GL_FILL)
        glPolygonMode(GL_BACK, GL_FILL)
        return display, window

    def create_img(self):
        height = self.height
        width = self.width
        display, window = self.init(self.K, width, height)
        tri = stl_model(self.cad_path).tri
        a, b, g, x, y, r = self.abgxyr[0], self.abgxyr[1], self.abgxyr[2], self.abgxyr[3], self.abgxyr[4], self.abgxyr[
            5]
        data_path = './yml/eight_points.yml'
        aver_mm = read_yaml(data_path)[self.obj_id][-1]
        aver = np.multiply(aver_mm, 0.001)
        if r > 200:
            im = np.array(draw_cube_abg(a, b, g, x / 1000, y / 1000, r / 1000, tri, window, display, aver))
            im = np.array(draw_cube_abg(a, b, g, x / 1000, y / 1000, r / 1000, tri, window, display, aver))
        else:
            im = np.array(draw_cube_abg(a, b, g, x, y, r, tri, window, display, aver))
            im = np.array(draw_cube_abg(a, b, g, x, y, r, tri, window, display, aver))
        pose_mask = np.zeros((height, width, 3))
        for i in range(3):
            pose_mask[:, :, i] = im[:, :, i].T
        return pose_mask


# ranges = read_yaml(osp.join('.\\test_dataset\\000023', 'obj_scene_pose_range.yml'))['6']
# Arange = ranges['Arange'][0]
# Brange = ranges['Brange'][0]
# Grange = ranges['Grange'][0]
# Xrange = ranges['Xrange'][0]
# Zrange = ranges['Yrange'][0]
# Rrange = ranges['Rrange'][0]
# amin, amax = Arange[0] - 0.3, Arange[1] + 0.3
# bmin, bmax = Brange[0] - 0.3, Brange[1] + 0.3
# gmin, gmax = Grange[0] - 0.3, Grange[1] + 0.3
# xmin, xmax = Xrange[0] - 0.1, Xrange[1] + 0.1
# ymin, ymax = Zrange[0] - 0.1, Zrange[1] + 0.1
# rmin, rmax = Rrange[0] - 0.05, Rrange[1] + 0.05
#
# min_max_list = [[amin, amax],
#                 [bmin, bmax],
#                 [gmin, gmax],
#                 [xmin, xmax],
#                 [ymin, ymax],
#                 [rmin, rmax]]
#
#
# def min_max(pose):
#     pose_new = np.zeros(6)
#     pose_new[0] = (pose[0] - amin) / (amax - amin)
#     pose_new[1] = (pose[1] - bmin) / (bmax - bmin)
#     pose_new[2] = (pose[2] - gmin) / (gmax - gmin)
#     pose_new[3] = (pose[3] - xmin) / (xmax - xmin)
#     pose_new[4] = (pose[4] - ymin) / (ymax - ymin)
#     pose_new[5] = (pose[5] - rmin) / (rmax - rmin)
#     return pose_new
#
#
# def min_max_rollback(pose_new):
#     pose = np.zeros(6)
#     pose[0] = pose_new[0] * (amax - amin) + amin
#     pose[1] = pose_new[1] * (bmax - bmin) + bmin
#     pose[2] = pose_new[2] * (gmax - gmin) + gmin
#     pose[3] = pose_new[3] * (xmax - xmin) + xmin
#     pose[4] = pose_new[4] * (ymax - ymin) + ymin
#     pose[5] = pose_new[5] * (rmax - rmin) + rmin
#     return pose


if __name__ == '__main__':
    ox = 1239.951701787861  # 相机内参
    # oy = 1023.50823945964  # 相机内参
    # FocalLength_x = 2344.665256639324  # 相机内参
    # FocalLength_y = 2344.050648508343  # 相机内参
    #
    # obj_scene = read_yaml('.\\data\\obj_scenes.yml')
    # overlay = OverLay(6)
    # overlay.K = [[FocalLength_x, 0, ox],
    #              [0, FocalLength_y, oy],
    #              [0, 0, 1]]
    # # eng = matlab.engine.start_matlab()
    # abg = read_yaml('.\\test_dataset\\000023\\gt_abg_new.yml')['6']['abg']
    # pick_ori = np.load('.\\test_dataset\\result_pick_ori.npy')
    # pick = np.load('.\\test_dataset\\result_pick.npy')
    # for i in range(6):
    #     bottom = cv2.imread(osp.join('./test_dataset/000023', str(i + 1) + '.jpg'))
    #     overlay.abgxyr = pick[i]
    #     # overlay.abgxyr = np.load('data/0-10000.npy')[0][:6]
    #     img = overlay.create_img().astype(np.uint8)
    #     canny_im = cv2.Canny(img, 100, 200)
    #     canny_im = cv2.threshold(canny_im, 0, 255, cv2.THRESH_BINARY)[1]
    #     kernel = np.ones((3, 3), np.uint8)
    #     dilation = cv2.dilate(canny_im, kernel, iterations=1)
    #     _, small = cv2.threshold(dilation, 0, 255, cv2.THRESH_BINARY)
    #     im4 = np.copy(dilation)
    #     im3 = np.zeros((2048, 2448, 3))
    #     color = [0, 255, 0]
    #     for m in range(2048):
    #         for n in range(2448):
    #             if im4[m][n] != 0.0:
    #                 im3[m][n] = color
    #     bottom = cv2.addWeighted(bottom, 1, im3.astype(np.uint8), 0.5, 0)
    #     overlay.save_path = osp.join('.\\vis', str(i + 1) + '.jpg')
    #     cv2.imwrite(overlay.save_path, bottom)
