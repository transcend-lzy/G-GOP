import argparse
from utils import clip_and_scaling
import os
import os.path as osp
import cv2
import numpy
import numpy as np
import torch
from model import *
from config import opt, mkdir, read_yaml
from tqdm import tqdm
from torch.autograd import Variable
from multiprocessing import Pool
from overlayAbg import OverLay as overlay_abg
import warnings
from torchnet import meter
import copy

warnings.filterwarnings('ignore')

# 配置GPU或CPU设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    z1 = Variable(torch.as_tensor(z1, dtype=torch.float32)).to(device)
    z2 = Variable(torch.as_tensor(z2, dtype=torch.float32)).to(device)
    x2 = Variable(torch.as_tensor(x2, dtype=torch.float32)).to(device)
    z1.requires_grad = False
    z2.requires_grad = True
    optimizer = torch.optim.Adam([z2], lr=opt.learning_rate_test)
    # 这里是为了配置优化器不优化model的参数
    for i in model.parameters():
        i.requires_grad = False
    index = 1
    update_times = {}
    tolerant_thr = 10
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
        update_times_copy = copy.deepcopy(update_times)
        for key in update_times_copy.keys():
            if update_times[key] > tolerant_thr:
                del update_times[key]
    return z2, z1, x2, u, v


def create_mask(contour_numpy, u, v):
    contour_numpy = np.uint8(numpy.multiply(contour_numpy[0], np.array([255])))
    mask = contour_numpy.copy()
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


def get_result(loss_meter, z2, z1, x2, model):
    loss_meter.reset()
    result = torch.cat((z2, z1), 1)
    output_all = model(result)
    loss_min = 10000000
    loss_min_index = 0
    for dim in range(len(result)):
        output = model(result[dim].view((1, 8)))
        loss, match_loss, not_match_loss = pick_loss(output[0].cpu().detach().numpy(),
                                                     x2[0].view((128, 128)).cpu().detach().numpy())
        output_show = copy.deepcopy(output[0].cpu().detach().numpy())
        np.place(output_show, output_show > 0.4, 255)
        if loss < loss_min:
            loss_min_index = dim
            loss_min = loss

    best = result.cpu().detach().numpy()[loss_min_index]
    contour_best = model(result[loss_min_index].view((1, 8))).cpu().detach().numpy()
    img_best = output_all.cpu().detach().numpy()[loss_min_index]
    result_pick = min_max_rollback(best.reshape((1, 8))[0][:6])

    return contour_best, result_pick, img_best


def mutil_test(scene_path, img_index, model, loss_meter, only_one=False):
    img = cv2.imread(osp.join(scene_path, 'rgb', str(img_index) + '.png'))
    random_bbox = read_yaml(osp.join(scene_path, 'ssdResult.yml'))
    xywh = None
    for i in random_bbox[str(img_index)]:
        if int(i['obj_id']) == opt.obj_id:
            xywh = i['xywh']
            break
    z2, z1, x2, u, v = pose_iterative(model, img, xywh, loss_meter)
    if only_one:
        _, result_pick, img_best = get_result(loss_meter, z2, z1, x2, model)
        return result_pick, img_best, img_index, u, v
    mask, mask_small = create_mask(get_result(loss_meter, z2, z1, x2, model)[0], u, v)
    z2, z1, x2, u, v = pose_iterative(model, img, xywh, loss_meter, True, mask, mask_small)
    _, result_pick, img_best = get_result(loss_meter, z2, z1, x2, model)
    return result_pick, img_best, img_index, u, v


def test():
    scene_path = opt.video_img_data
    model = VAE_NEW(opt.latent_dim, opt.dim)
    model = nn.DataParallel(model).to(device)
    if osp.exists(opt.model_path):
        pths = [int(pth.split(".")[0]) for pth in os.listdir(opt.model_path) if
                "pth" in pth and "optim" not in pth]
        if len(pths) != 0:
            pth = max(pths)
            print("Load model: {}".format(os.path.join(opt.model_path, "{}.pth".format(pth))))
            model.load_state_dict(
                torch.load(os.path.join(opt.model_path, str(pth) + ".pth")))
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
    np.save(osp.join('./', 'result_pick_all.npy'), result_pick_all)
    np.save(osp.join('./', 'img_all.npy'), img_all)
    np.save(osp.join('./', 'uv_all.npy'), uv_all)


def overlay_img():
    img_all = np.load(osp.join('./', 'img_all.npy'), allow_pickle=True)
    uv_all = np.load(osp.join('./', 'uv_all.npy'), allow_pickle=True)
    for i in tqdm(range(215)):
        u, v = int(uv_all[i][0]), int(uv_all[i][1])
        top = v
        bottom = 2048 - opt.bbox_len - v
        left = u
        right = 2448 - opt.bbox_len - u
        np.place(img_all[i], img_all[i] > 0.4, 255)
        ret, binary = cv2.threshold(img_all[i], 10, 255, cv2.THRESH_BINARY)
        dilation = cv2.resize(binary, (opt.bbox_len, opt.bbox_len), interpolation=cv2.INTER_AREA)
        constant = cv2.copyMakeBorder(dilation, top, bottom, left, right, cv2.BORDER_CONSTANT)
        img = cv2.imread(osp.join(opt.video_img_data, 'rgb', str(i + 1) + '.png'))
        height, width = 2048, 2448
        im3 = np.zeros((height, width, 3))
        pix_where = np.argwhere(constant)
        color = [0, 255, 0]
        for pixel in pix_where:
            im3[pixel[0]][pixel[1]] = color
        overlay = cv2.addWeighted(img, 1, im3.astype(np.uint8), 0.5, 0)
        mkdir(osp.join(opt.video_img_data, 'video_overlay'))
        if not osp.exists('./final_img'):
            mkdir('./final_img')
        cv2.imwrite(osp.join('./final_img', str(i + 1) + '.png'), overlay)


def overlay_img_opengl(bottom_current, obj_id, abg):
    height, width = 2048, 2448
    overlay = overlay_abg(obj_id)
    overlay.abgxyr = abg
    mtx = np.zeros((3, 3))
    mtx[0][0], mtx[1][1], mtx[0][2], mtx[1][2], mtx[2][2] = opt.fx, opt.fy, opt.ox, opt.oy, 1.0
    overlay.K = mtx
    overlay.cad_path = opt.cad_path
    img = overlay.create_img().astype(np.uint8)
    canny_im = cv2.Canny(img, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(canny_im, kernel, iterations=1)
    im4 = np.copy(dilation)
    im3 = np.zeros((height, width, 3))
    color = [0, 255, 0]
    for m in range(height):
        for n in range(width):
            if im4[m][n] != 0.0:
                im3[m][n] = color
    bottom_current = cv2.addWeighted(bottom_current, 1, im3.astype(np.uint8), 0.5, 0)
    return bottom_current


def crop_img():
    img_path_ori = './final_img'
    img_path_crop = './final_img_crop'
    if not osp.exists(img_path_crop):
        os.makedirs(img_path_crop)
    for i in tqdm(range(1, 216)):
        img = cv2.imread(osp.join(img_path_ori, str(i) + '.png'))
        img_new = img[512:1536, 612:1836]
        cv2.imwrite(osp.join(img_path_crop, str(i) + '.png'), img_new)


def overlay_by_opengl():
    pick_all = np.load('./result_pick_all.npy')
    obj_ids = [6]
    img_path_dst = opt.video_img_data
    for img_index in tqdm(range(1, 216)):
        for obj_id in obj_ids:
            abg = pick_all[img_index - 1][0]
            img_path = osp.join(img_path_dst, 'rgb', str(img_index) + '.png')
            bottom_final = cv2.imread(img_path)
            bottom_final = overlay_img_opengl(bottom_final, abg=abg, obj_id=obj_id)
            if not osp.exists('./final_img'):
                mkdir('./final_img')
            cv2.imwrite(osp.join('./final_img', str(img_index) + '.png'), bottom_final)


def img_2_video():
    file_dir = './final_img_crop'
    list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            list.append(file)  # 获取目录下文件名列表

    video = cv2.VideoWriter('./video_new.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30,
                            (1224, 1024))
    for i in tqdm(range(1, 216)):
        img = cv2.imread(osp.join(file_dir, str(i) + '.png'))
        video.write(img)

    video.release()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--need_opengl", type=bool, default=False)
    args = parser.parse_args()
    need_opengl = args.need_opengl
    # test()
    if need_opengl:
        overlay_by_opengl()
    else:
        overlay_img()
    crop_img()
    img_2_video()
