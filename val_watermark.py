import numpy as np
import sys
import re
import os
import torch

from models.wide_resnet import Wide_ResNet
from train_watermark import get_layer_by


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def get_param_from_name(name):
    """从给定的字符串 name 中提取特定的参数值"""

    match = re.search(
        'N(\d+)K(\d+)[^_]+_TBLK(\d+)_layer(\d+)', name)

    if match:
        N = int(match.group(1))
        k = int(match.group(2))
        target_blk_id = int(match.group(3))
        target_layer_id = int(match.group(4))
        return N, k, target_blk_id, target_layer_id
    return None


def get_layer_weights_and_predicted(weight_fname, wparam_fname, nb_classes):
    N, k, target_blk_id, target_layer_id = get_param_from_name(wparam_fname)
    # create model and load weights
    model = Wide_ResNet(N * 6 + 4, k, num_classes=nb_classes)
    model.load_state_dict(torch.load(weight_fname))

    w = np.load(wparam_fname)

    # get signature from model weight and matrix
    target_layer = get_layer_by(model, target_layer_id)  # 目标层的权重参数
    weight = torch.mean(target_layer, 0)
    pred_b_param = np.dot(weight.view(1, -1).detach().numpy(), w)  # dot product
    pred_b_param = 1 / (1 + np.exp(-pred_b_param))  # apply sigmoid
    return target_layer.detach().numpy(), pred_b_param


if __name__ == '__main__':
    weight_fname = sys.argv[1]
    wparam_fname = sys.argv[2]
    oprefix = sys.argv[3]

    # cifar10 10 classes
    nb_classes = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    # get file and parameters
    layer_weights, pred_bparam = get_layer_weights_and_predicted(weight_fname, wparam_fname, nb_classes)
    print(layer_weights,pred_bparam)

    np.save('{}_predict_bparam.npy'.format(oprefix), pred_bparam)
    np.save('{}_layer_weight.npy'.format(oprefix), layer_weights)
