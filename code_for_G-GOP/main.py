import math
import os

import cv2
import numpy
import numpy as np
import torch
from utils import *
from model import *
from config import opt
from dataset import GenImg
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader

import warnings
from torchnet import meter
import random
import copy

warnings.filterwarnings('ignore')
from tensorboardX import SummaryWriter
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

init()
# 配置GPU或CPU设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer_train = SummaryWriter(opt.log_train)
writer_test = SummaryWriter(opt.log_test)
a_scope = ranges['Arange'][0][1] - ranges['Arange'][0][0]
b_scope = ranges['Brange'][0][1] - ranges['Brange'][0][0]
g_scope = ranges['Grange'][0][1] - ranges['Grange'][0][0]
x_scope = ranges['Xrange'][0][1] - ranges['Xrange'][0][0]
y_scope = ranges['Yrange'][0][1] - ranges['Yrange'][0][0]
r_scope = ranges['Rrange'][0][1] - ranges['Rrange'][0][0]
min_scope_abg = min(a_scope, b_scope, g_scope)
min_scope_xyr = min(x_scope, y_scope, r_scope)
a_scale = math.sqrt(a_scope / min_scope_abg)
b_scale = math.sqrt(b_scope / min_scope_abg)
g_scale = math.sqrt(g_scope / min_scope_abg)
abg_scale = [a_scale, b_scale, g_scale]
x_scale = math.sqrt(x_scope / min_scope_xyr)
y_scale = math.sqrt(y_scope / min_scope_xyr)
r_scale = math.sqrt(r_scope / min_scope_xyr)
xyr_scale = [x_scale, y_scale, r_scale]


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100 * (1 - super(SSIM_Loss, self).forward(img1, img2))


min_max_list = [[amin, amax],
                [bmin, bmax],
                [gmin, gmax],
                [xmin, xmax],
                [ymin, ymax],
                [rmin, rmax]]


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


def pose_iterative(model, img, xywh, loss_meter, is_refine=None, mask=None, mask_small=None):
    if is_refine:
        if mask is not None and mask_small is not None:
            u, v, target_img = clip_and_scaling(img, xywh, opt.bbox_len, mask, mask_small)
        else:
            raise "must have mask"
    else:
        u, v, target_img = clip_and_scaling(img, xywh, opt.bbox_len)
    target_img = target_img.astype(np.float32) / 255.
    batch = 64
    z2 = np.zeros((batch, 6))
    z1 = np.zeros((batch, 2))
    x2 = np.zeros((batch, 128, 128))
    for i in range(batch):
        print("process {} init".format(i + 1))
        z2[i] = np.array([min_max(creatPose(u, v))])
        z1[i] = np.array([[u, v]])
        x2[i] = target_img

    z1 = torch.as_tensor(z1, dtype=torch.float32)
    z2 = torch.as_tensor(z2, dtype=torch.float32)
    x2 = torch.as_tensor(x2, dtype=torch.float32)
    z1 = Variable(z1).to(device)
    z2 = Variable(z2).to(device)
    x2 = Variable(x2).to(device)
    z1.requires_grad = False
    z2.requires_grad = True
    optimizer = torch.optim.Adam([z2], lr=opt.learning_rate_test)
    # 这里是为了配置优化器不优化model的参数
    for i in model.parameters():
        i.requires_grad = False
    index = 1
    update_times = {}
    tolerant_thr = 10
    save_index = 0
    for i in tqdm(range(opt.test_ite)):
        for j in range(batch):
            z2_numpy = copy.deepcopy(z2).cpu().detach().numpy()
            pose_abg = min_max_rollback(z2_numpy[j])
            a, b, g, x, y, r = pose_abg[0], pose_abg[1], pose_abg[2], pose_abg[3], pose_abg[4], pose_abg[5]
            if not val_pose(a, b, g, x, y, r, u, v):
                if str(j) in update_times.keys():
                    update_times[str(j)] += 1
                    continue
                else:
                    update_times[str(j)] = 0
                # print("update pose for process {}".format(j))
                with torch.no_grad():
                    z2[j] = Variable(torch.as_tensor(min_max(creatPose(u, v)), dtype=torch.float32)).to(device)
        z2.requires_grad = True
        output = model(torch.cat((z2, z1), 1))
        # if i == 0:
        #     save_example_im_double(x2, output, 0, is_test=True)
        loss = F.binary_cross_entropy(output, x2)
        # 反向传播与优化
        # 清空上一步的残余更新参数值
        optimizer.zero_grad()
        # 误差反向传播, 计算参数更新值
        loss.mean().backward()
        # 按照范围更新梯度
        # for dimen in range(len(z2)):
        #     for abg_dim in range(3):
        #         z2.grad[dimen][abg_dim] = z2.grad[dimen][abg_dim] / abg_scale[abg_dim]
        #     for xyr_dim in range(3, 6):
        #         z2.grad[dimen][xyr_dim] = z2.grad[dimen][xyr_dim] / xyr_scale[xyr_dim - 3]
        # 将参数更新值施加到VAE model的parameters上
        optimizer.step()
        loss_meter.add(loss.data.cpu())
        if is_refine:
            writer_test.add_scalar('loss', loss_meter.value()[0], i + opt.test_ite)
        else:
            writer_test.add_scalar('loss', loss_meter.value()[0], i)
        update_times_copy = copy.deepcopy(update_times)
        for key in update_times_copy.keys():
            if update_times[key] > tolerant_thr:
                del update_times[key]
        if i % opt.adjust == 0:
            if is_refine:
                save_example_im_double(x2, output, save_index + 10000, is_test=True)
            else:
                save_example_im_double(x2, output, save_index, is_test=True)
            save_index += 1
            writer_test.add_scalar('lr', optimizer.param_groups[0]['lr'], index)
            index += 1
    return z2, z1, x2, u, v


def create_mask(contour_numpy, u, v):
    contour_numpy = np.uint8(numpy.multiply(contour_numpy[0], np.array([255])))
    # show_photos([contour_numpy])
    mask = contour_numpy.copy()
    # _, binaryzation = cv2.threshold(contour_numpy, 100, 255, cv2.THRESH_BINARY_INV)
    # show_photos([binaryzation])
    # 找到所有的轮廓
    contours, _ = cv2.findContours(contour_numpy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    area = []

    # 找到最大的轮廓
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))

    # 填充最大的轮廓
    mask = cv2.drawContours(mask, contours, max_idx, 255, cv2.FILLED)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    top = v
    bottom = 2048 - opt.bbox_len - v
    left = u
    right = 2448 - opt.bbox_len - u
    dilation = cv2.resize(dilation, (opt.bbox_len, opt.bbox_len), interpolation=cv2.INTER_AREA)
    constant = cv2.copyMakeBorder(dilation, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return constant, dilation


def get_result(loss_meter, z2, z1, x2, model, is_refine=None):
    loss_meter.reset()
    result = torch.cat((z2, z1), 1)
    scores = np.zeros((len(result), 1))
    loss_min = 10000000
    loss_min_index = 0
    for dim in range(len(result)):
        output = model(result[dim].view((1, 8)))
        loss = F.binary_cross_entropy(output, x2[0].view((1, 128, 128)))
        if loss < loss_min:
            loss_min_index = dim
            loss_min = loss
        loss_meter.add(loss.data.cpu())
        scores[dim] = loss.data.cpu()
    result_numpy = result.cpu().detach().numpy()
    if is_refine:
        np.save('../data/result_all.npy', result_numpy)
    del_index = []
    for index, score in enumerate(scores):
        if score > loss_meter.value()[0].cpu().numpy():
            del_index.append(index)
    best = result_numpy[loss_min_index]
    contour_best = model(result[loss_min_index].view((1, 8))).cpu().detach().numpy()
    result_pick = np.concatenate([np.delete(result_numpy, del_index, axis=0), best.reshape((1, 8))], axis=0)
    if is_refine:
        np.save('../data/result_pick.npy', result_pick)
    return contour_best


def test(scene_id):
    scene_path = osp.join(opt.test_data_root, 'scene' + str(scene_id))
    img = cv2.imread(osp.join(scene_path, 'rgb', '6.jpg'))
    random_bbox = read_yaml(osp.join(scene_path, 'bboxOffset.yml'))
    xywh = None
    for i in random_bbox['10']:
        if i['obj_id'] == opt.obj_id:
            xywh = i['xywh']
            break

    model = VAE_NEW(opt.latent_dim, opt.dim)
    model = nn.DataParallel(model).to(device)
    if osp.exists(opt.save_model_path):
        pths = [int(pth.split(".")[0]) for pth in os.listdir(opt.save_model_path) if
                "pth" in pth and "optim" not in pth]
        pth = 0
        if len(pths) != 0:
            pth = max(pths)
            print("Load model: {}".format(os.path.join(opt.save_model_path, "{}.pth".format(pth))))
            model.load_state_dict(
                torch.load(os.path.join(opt.save_model_path, str(pth) + ".pth")))
    model.eval()
    loss_meter = meter.AverageValueMeter()
    z2, z1, x2, u, v = pose_iterative(model, img, xywh, loss_meter)

    mask, mask_small = create_mask(get_result(loss_meter, z2, z1, x2, model), u, v)
    z2, z1, x2, u, v = pose_iterative(model, img, xywh, loss_meter, True, mask, mask_small)
    get_result(loss_meter, z2, z1, x2, model, True)


def val(val_dataloader, model, epoch):
    model.eval()
    for (data, label) in tqdm(val_dataloader):
        input = torch.as_tensor(label, dtype=torch.float32)
        input = Variable(input).to(device)
        target = Variable(data).to(device)
        output = model(input)

        loss_cross = F.binary_cross_entropy(output, target)

    save_example_im_double(target, output, epoch, True)

    return loss_cross


def train():
    # 构造实例对象
    model = VAE_NEW(opt.latent_dim, opt.dim)
    model = nn.DataParallel(model).to(device)
    writer_train = SummaryWriter(opt.log_train)
    # 选择优化器，并传入VAE模型参数和学习率  实验weight_decay=0.005
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, mode='min', patience=4,
                                                              cooldown=3, min_lr=1e-9, verbose=True)
    pth = 0
    if osp.exists(opt.save_model_path):
        pths = [int(pth.split(".")[0]) for pth in os.listdir(opt.save_model_path) if
                "pth" in pth and "optim" not in pth]
        pth = 0
        if len(pths) != 0:
            pth = max(pths)
            print("Load model: {}".format(os.path.join(opt.save_model_path, "{}.pth".format(pth))))
            print("Load optimizer: {}".format(os.path.join(opt.save_model_path, "optim{}.pth".format(pth))))
            model.load_state_dict(
                torch.load(os.path.join(opt.save_model_path, str(pth) + ".pth")))
            optimizer.load_state_dict(
                torch.load(os.path.join(opt.save_model_path, "optim" + str(pth) + ".pth")))

    train_data = GenImg(opt.train_data_root, train=True)
    val_data = GenImg(opt.train_data_root, train=False)

    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_data, opt.batch_size,
                                shuffle=False, num_workers=opt.num_workers)

    loss_meter = meter.AverageValueMeter()
    index = pth * 2
    ssim_loss = SSIM_Loss(win_size=5, win_sigma=1.5, data_range=1, size_average=True, channel=1, nonnegative_ssim=True)
    for j in range(pth, opt.max_epoch):
        loss_meter.reset()
        for i, (data, label) in tqdm(enumerate(train_dataloader)):
            input = torch.as_tensor(label, dtype=torch.float32)
            input = Variable(input).to(device)
            target = Variable(data).to(device)
            output = model(input)  # 这时候已经是二值图像了
            # 重构损失

            loss = F.binary_cross_entropy(output, target)
            # loss = ssim_loss(output.unsqueeze(1), target.unsqueeze(1))
            # 反向传播与优化
            # 清空上一步的残余更新参数值
            optimizer.zero_grad()
            # 误差反向传播, 计算参数更新值
            loss.mean().backward()
            # 将参数更新值施加到VAE model的parameters上
            optimizer.step()
            # 每迭代一定步骤，打印结果值
            loss_meter.add(loss.data.cpu())
            if i % opt.print_freq == opt.print_freq - 1:
                save_example_im_double(target, output, index)
                writer_train.add_scalar('loss_cross', loss_meter.value()[0], index)
                index += 1
        writer_train.add_scalar('lr', optimizer.param_groups[0]['lr'], j)
        val_loss = val(val_dataloader, model, j)
        writer_train.add_scalar('val_loss', val_loss, j)
        scheduler_lr.step(loss_meter.value()[0])

        if j % opt.save_fre == opt.save_fre - 1:
            torch.save(model.state_dict(),
                       os.path.join(opt.save_model_path, str(j + 1) + ".pth"))
            torch.save(optimizer.state_dict(),
                       os.path.join(opt.save_model_path, "optim" + str(j + 1) + ".pth"))


if __name__ == '__main__':
    test(23)
    # train()
