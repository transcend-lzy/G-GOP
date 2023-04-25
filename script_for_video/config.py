# coding:utf8
import warnings
import os
import os.path as osp
import yaml
def mkdir(path):
    if not osp.exists(path):
        os.makedirs(path)

def read_yaml(yaml_path):
    f = open(yaml_path, 'r', encoding='utf-8')
    cfg = f.read()
    dic = yaml.safe_load(cfg)
    return dic

class DefaultConfig(object):
    bbox_len = 384
    ox = 1239.951701787861  # 相机内参
    oy = 1023.50823945964  # 相机内参
    fx = 2344.665256639324
    fy = 2344.050648508343
    learning_rate_test = 0.005
    test_ite = 300
    obj_id = 6
    video_img_data = './video_img'
    latent_dim = 8
    dim = 32 * 7  # 实验得出
    model_path = './checkpoint'
    cad_path = './cad/6.stl'
    height, width = 2048, 2448


def parse(self, kwargs):
    """
    根据字典kwargs 更新 config参数
    """
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
# opt.parse = parse
