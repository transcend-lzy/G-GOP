'''
共有四个位姿表达方式互相转换：
1、worldOrientation, worldLocation ： matlab的pnp位姿结果形式
2、rotationMatrix, translationVector： matlab定义的m2c位姿形式
（worldOrientation, worldLocation做cameraPoseToExtrinsics转换的结果）
3、m2c_r, m2c_t : opencv python  pnp得到的结果，也是我们所熟知的m2c形式
4、abgxyr: 一种新型的位姿表示形式
'''
import time

import numpy as np
from math import cos, sin


def trans_pyrt_to_matrt2(r, t):
    """
    将py m2c的rt 转换为matlab m2c的rt
    Args:
        r,t: py m2c的rt，就是pnp计算出来的结果

    Returns:
        r,t: matlab m2c的rt，是matlab pnp计算出来的结果再做cameraPoseToExtrinsics转换的结果
    """
    matrix = [[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]
    # 旋转矩阵先转置在前两列取反
    matr2 = np.dot(np.array(r, dtype=np.float32).T, np.array(matrix, dtype=np.float32))
    # 平移矩阵前两列取反
    matt2 = np.dot(np.multiply(np.array(t, dtype=np.float32), 1000), np.array(matrix, dtype=np.float32))
    return matr2, matt2


def trans_matr2_tomatr(r, t):
    """
    转换关系参考官网公式：https://ww2.mathworks.cn/help/vision/ref/cameraposetoextrinsics.html?requestedDomain=cn
    Args:
        r,t: matlab m2c的rt，是matlab pnp计算出来的结果再做cameraPoseToExtrinsics转换的结果

    Returns:
        r,t: matlab 的rt，是matlab pnp计算出来的结果
    """

    matr = np.array(r).T
    matr_inv = np.linalg.inv(r)
    matt = np.dot(np.array(-t), matr_inv)
    return matr, matt


def trans_matrt2_to_rt(r, t):
    """
    将rotationMatrix, translationVector 转换为m2c_r, m2c_t
    """
    matrix = [[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]
    # 旋转矩阵先前两列取反再转置
    m2c_r = np.dot(np.array(r, dtype=np.float32), np.array(matrix, dtype=np.float32)).T
    # 平移向量前两列取反
    m2c_t = np.dot(np.array(t, dtype=np.float32), np.array(matrix, dtype=np.float32))
    return m2c_r, m2c_t


def abg2rt(a, b, g, x_trans, z_trans, r):
    """
    retunrn:
    worldOrientation, worldLocation: matlab pnp得到的结果
    rotationMatrix, translationVector： matlab pnp结果后cameraPoseToExtrinsic的结果
    m2c_r, m2c_t : python pnp的结果 正常的m2c结果
    """
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

    worldOrientation = rm.T
    worldLocation = pos  # 平移矩阵

    rotationMatrix = worldOrientation.T
    translationVector = -np.matmul(worldLocation, worldOrientation.T)
    m2c_r, m2c_t = trans_matrt2_to_rt(rotationMatrix, translationVector)

    return worldOrientation, worldLocation, rotationMatrix, translationVector, m2c_r, m2c_t

#
# def rt2abg(m2c_r, m2c_t, eng):
#     """
#     将m2c_r, m2c_t 转换为 abgxyr
#     """
#     rotationMatrix, translationVector = trans_pyrt_to_matrt2(m2c_r, m2c_t)
#     # 先转换成worldOrientation, worldLocation
#     worldOrientation, worldLocation = trans_matr2_tomatr(rotationMatrix, translationVector)
#     # 做一次转换，matlab不允许直接使用np
#     r = matlab.double(np.array(worldOrientation).tolist())
#     t = matlab.double(worldLocation.tolist())
#     # 调用matlab函数，文件夹中要有一个同名同名rt2abg.m 里面写了function为rt2abg
#     # eng = matlab.engine.start_matlab()
#     # 输入的rt 是worldOrientation, worldLocation
#     ret = eng.rt2abg(r, t[0])
#     return ret[0]
