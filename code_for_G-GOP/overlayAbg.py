import os

import cv2
import numpy as np
from math import pi
from overlayUtils import *
from trans_pose import *
from utils import *


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
        self.eight_path = None

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
        data_path = './data'
        aver_mm = read_yaml(self.eight_path)[self.obj_id][-1]
        aver = np.multiply(aver_mm, 0.001)
        if r > 200:
            im = np.array(draw_cube_abg(a, b, g, x / 1000, y / 1000, r / 1000, tri, window, display, aver))
            # im = np.array(draw_cube_abg(a, b, g, x / 1000, y / 1000, r / 1000, tri, window, display, aver))
        else:
            im = np.array(draw_cube_abg(a, b, g, x, y, r, tri, window, display, aver))
            # im = np.array(draw_cube_abg(a, b, g, x, y, r, tri, window, display, aver))
        pose_mask = np.zeros((height, width, 3))
        for i in range(3):
            pose_mask[:, :, i] = im[:, :, i].T
        return pose_mask




if __name__ == '__main__':
    ox = 1239.951701787861  # 相机内参
    oy = 1023.50823945964  # 相机内参
    FocalLength_x = 2344.665256639324  # 相机内参
    FocalLength_y = 2344.050648508343  # 相机内参
    bottom = cv2.imread('./vis/6.jpg')
    obj_scene = read_yaml('.\\data\\obj_scenes.yml')
    overlay = OverLay(6)
    overlay.K = [[FocalLength_x, 0, ox],
                 [0, FocalLength_y, oy],
                 [0, 0, 1]]
    # eng = matlab.engine.start_matlab()
    abg = read_yaml('.\\test_dataset\\000023\\gt_abg_new.yml')['6']['abg']
    abg_np = min_max_rollback(np.load('.\\test_dataset\\result_pick.npy')[0][:6])
    overlay.abgxyr = abg_np
    # overlay.abgxyr = np.load('data/0-10000.npy')[0][:6]
    img = overlay.create_img().astype(np.uint8)
    canny_im = cv2.Canny(img, 100, 200)
    canny_im = cv2.threshold(canny_im, 0, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(canny_im, kernel, iterations=1)
    _, small = cv2.threshold(dilation, 0, 255, cv2.THRESH_BINARY)
    im4 = np.copy(dilation)
    im3 = np.zeros((2048, 2448, 3))
    color = [0, 255, 0]
    for m in range(2048):
        for n in range(2448):
            if im4[m][n] != 0.0:
                im3[m][n] = color
    bottom = cv2.addWeighted(bottom, 1, im3.astype(np.uint8), 0.5, 0)
    cv2.imwrite(overlay.save_path, bottom)
