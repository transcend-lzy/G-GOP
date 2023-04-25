import argparse
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
from multiprocessing import Pool
import warnings
from torchnet import meter
import random
import copy
from ruamel import yaml

warnings.filterwarnings('ignore')
from tensorboardX import SummaryWriter

init()
# 配置GPU或CPU设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer_train = SummaryWriter(opt.log_train)
writer_test = SummaryWriter(opt.log_test)


def pick_loss(output, target):
    np.place(output, output > 0.4, 1)
    np.place(target, target > 0, 1)
    loss_match = np.sum(output.astype(np.uint8) & target.astype(np.uint8))
    loss_not_match = np.sum(np.logical_xor(output.astype(np.uint8), target.astype(np.uint8)))
    loss_final = loss_match - loss_not_match
    return -loss_final.astype(np.long), loss_not_match, loss_match


def pose_iterative(model, img, xywh, loss_meter, is_refine=None, mask=None, mask_small=None, batch=64):
    if is_refine:
        if mask is not None and mask_small is not None:
            u, v, target_img = clip_and_scaling(img, xywh, opt.bbox_len, mask, mask_small)
        else:
            raise "must have mask"
    else:
        u, v, target_img = clip_and_scaling(img, xywh, opt.bbox_len)
    target_img = target_img.astype(np.float32) / 255.
    z2 = np.zeros((batch, 6))
    z1 = np.zeros((batch, 2))
    x2 = np.zeros((batch, 128, 128))
    for i in tqdm(range(batch)):
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
    for ite in tqdm(range(opt.test_ite)):
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
        loss = F.binary_cross_entropy(output, x2)
        # 反向传播与优化
        # 清空上一步的残余更新参数值
        optimizer.zero_grad()
        # 误差反向传播, 计算参数更新值
        loss.mean().backward()
        # 将参数更新值施加到VAE model的parameters上
        optimizer.step()
        loss_meter.add(loss.data.cpu())
        # if is_refine:
        #     writer_test.add_scalar('loss', loss_meter.value()[0], i + opt.test_ite)
        # else:
        #     writer_test.add_scalar('loss', loss_meter.value()[0], i)
        update_times_copy = copy.deepcopy(update_times)
        for key in update_times_copy.keys():
            if update_times[key] > tolerant_thr:
                del update_times[key]
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


def get_result(loss_meter, z2, z1, x2, model, img_index, is_refine=None):
    output_all = []
    loss_meter.reset()
    result = torch.cat((z2, z1), 1)
    output_all = model(result)
    save_example_im_double(x2, output_all, int(img_index), is_test=True)
    loss_min = 10000000
    loss_min_index = 0
    for dim in range(len(result)):
        output = model(result[dim].view((1, 8)))
        # loss = F.binary_cross_entropy(output, x2[0].view((1, 128, 128)))
        loss, match_loss, not_match_loss = pick_loss(output[0].cpu().detach().numpy(),
                                                     x2[0].view((128, 128)).cpu().detach().numpy())
        output_show = copy.deepcopy(output[0].cpu().detach().numpy())
        np.place(output_show, output_show > 0.4, 255)
        # cv2.imwrite(osp.join(opt.vis_path, str(dim) + '.jpg'), output_show)
        # print("result %s loss is %s loss2 is %s loss3 is %s" % (dim, loss, match_loss, not_match_loss))
        if loss < loss_min:
            loss_min_index = dim
            loss_min = loss

    best = result.cpu().detach().numpy()[loss_min_index]
    contour_best = model(result[loss_min_index].view((1, 8))).cpu().detach().numpy()
    img_best = output_all.cpu().detach().numpy()[loss_min_index]
    result_pick = min_max_rollback(best.reshape((1, 8))[0][:6])

    return contour_best, result_pick, img_best


def mutil_test(scene_path, img_index, model, loss_meter, only_one=False):
    img = cv2.imread(osp.join(scene_path, 'rgb', str(img_index) + '.jpg'))
    show_photos([img])
    random_bbox = read_yaml(osp.join(scene_path, 'ssdResult.yml'))
    xywh = None
    for i in random_bbox[str(img_index)]:
        if int(i['obj_id']) == opt.obj_id:
            xywh = i['xywh']
            break
    z2, z1, x2, u, v = pose_iterative(model, img, xywh, loss_meter)
    if only_one:
        _, result_pick, img_best = get_result(loss_meter, z2, z1, x2, model, img_index, True)
        return result_pick, img_best, img_index, u, v
    mask, mask_small = create_mask(get_result(loss_meter, z2, z1, x2, model, img_index)[0], u, v)
    z2, z1, x2, u, v = pose_iterative(model, img, xywh, loss_meter, True, mask, mask_small)
    _, result_pick, img_best = get_result(loss_meter, z2, z1, x2, model, True)
    return result_pick, img_best, img_index, u, v


def save_yaml(dic, file_name):
    yaml_file = osp.join(opt.data_path, str(file_name) + '.yml')
    file = open(yaml_file, 'w', encoding='utf-8')
    yaml.dump(dic, file, Dumper=yaml.RoundTripDumper)
    file.close()


def test(scene_dir_name):
    scene_path = osp.join(opt.test_data_root, scene_dir_name)
    model = VAE_NEW(opt.latent_dim, opt.dim)
    model = nn.DataParallel(model).to(device)
    if osp.exists(opt.save_model_path):
        pths = [int(pth.split(".")[0]) for pth in os.listdir(opt.save_model_path) if
                "pth" in pth and "optim" not in pth]
        if len(pths) != 0:
            pth = max(pths)
            print("Load model: {}".format(os.path.join(opt.save_model_path, "{}.pth".format(pth))))
            model.load_state_dict(
                torch.load(os.path.join(opt.save_model_path, str(pth) + ".pth")))
    model.eval()
    loss_meter = meter.AverageValueMeter()
    pool = Pool(processes=3)
    res_l = []
    for i in tqdm(range(1)):
        ret = pool.apply_async(mutil_test, args=(scene_path, i + 1, model, loss_meter))
        res_l.append(ret)
    pool.close()
    pool.join()
    img_all = np.zeros((len(res_l), 128, 128))
    result_pick_all = np.zeros((len(res_l), 1, 6))
    uv_all = np.zeros((len(res_l), 2))
    for res in res_l:
        result_pick, img_best, img_index, u, v = res.get()
        img_all[int(img_index) - 1] = img_best
        result_pick_all[int(img_index) - 1] = result_pick
        uv_all[int(img_index) - 1] = [u, v]
    np.save(osp.join(opt.data_path, 'result_pick_all.npy'), result_pick_all)
    np.save(osp.join(opt.data_path, 'img_all.npy'), img_all)
    np.save(osp.join(opt.data_path, 'uv_all.npy'), uv_all)


def val(val_dataloader, model, epoch):
    target, output, loss_cross = None, None, None
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


def overlay_img():
    img_all = np.load(osp.join(opt.data_path, 'img_all.npy'), allow_pickle=True)
    uv_all = np.load(osp.join(opt.data_path, 'uv_all.npy'), allow_pickle=True)
    for i in tqdm(range(215)):
        u, v = int(uv_all[i][0]), int(uv_all[i][1])
        top = v
        bottom = 2048 - opt.bbox_len - v
        left = u
        right = 2448 - opt.bbox_len - u
        show_photos([img_all[i]])
        np.place(img_all[i], img_all[i] > 0.4, 255)
        ret, binary = cv2.threshold(img_all[i], 10, 255, cv2.THRESH_BINARY)
        # show_photos([binary])
        dilation = cv2.resize(binary, (opt.bbox_len, opt.bbox_len), interpolation=cv2.INTER_AREA)
        constant = cv2.copyMakeBorder(dilation, top, bottom, left, right, cv2.BORDER_CONSTANT)
        img = cv2.imread(osp.join(opt.test_data_root, 'video_data', 'rgb', str(i + 1) + '.jpg'))
        height, width = 2048, 2448
        im3 = np.zeros((height, width, 3))
        pix_where = np.argwhere(constant)
        color = [0, 255, 0]
        for pixel in pix_where:
            im3[pixel[0]][pixel[1]] = color
        overlay = cv2.addWeighted(img, 1, im3.astype(np.uint8), 0.5, 0)
        # show_photos([overlay])
        cv2.imwrite(osp.join(opt.test_data_root, 'video_overlay', str(i + 1) + '.jpg'), overlay)
    # show_photos([overlay])


if __name__ == '__main__':
    # overlay_img()
    torch.multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--scene_dir_name", type=str, default='scene23')

    args = parser.parse_args()
    if args.train:
        train()
    else:
        test(args.scene_dir_name)
