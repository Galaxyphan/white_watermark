import numpy as np
import sys
import json
import os
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from models.wide_resnet import Wide_ResNet

# from utils import random_index_generator

RESULT_PATH = './result'


def random_index_generator(count):
    """生成一系列随机的、不重复的索引值,范围是 0 到 count - 1"""
    indices = np.arange(0, count)
    print(indices)
    np.random.shuffle(indices)
    print(indices)

    for idx in indices:
        yield idx


def get_layer_by(model, target_blk_num):
    """从模型中提取指定层的权重参数"""

    return model.get_parameter(f'layer{target_blk_num}.0.conv2.weight')


def train(model, optimizer, dataloader, w, b, k, nb_epoch, target_blk_num):
    """训练函数:
        w:watermark matrix
        b:全为1的矩阵
        k:模型超参
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss().cuda()

    device = next(model.parameters()).device  # 获取模型参数所在的设备
    b = torch.tensor(b, dtype=torch.float32).to(device)
    w = torch.tensor(w, dtype=torch.float32).to(device)

    for ep in tqdm(range(nb_epoch)):
        for img, label in dataloader:
            img = img.to(device)  # data
            label = label.to(device)  # label
            optimizer.zero_grad()  # 除之前批次的梯度信息
            predict_class = model(img)
            loss = criterion(predict_class, label)
            regularized_loss = 0

            # 嵌入水印
            if target_blk_num > 0:
                p = get_layer_by(model, target_blk_num)
                x = torch.mean(p, dim=0)
                y = x.view(1, -1)  # 将 x 改变形状为一行的张量
                regularized_loss = k * torch.sum(
                    F.binary_cross_entropy(input=torch.sigmoid(torch.matmul(y, w)), target=b))

            (loss + regularized_loss).backward()  # 向后传播
            optimizer.step()  # 梯度下降


def build_wm(model, target_blk_num, embed_dim, wtype):
    """根据水印类型 wtype 创建相应的水印矩阵"""

    if target_blk_num == 0:
        '''嵌入第0层，就是不嵌入，也就不构造水印矩阵'''
        return np.array([])

    # get param
    p = get_layer_by(model, target_blk_num)  # 设置水印正则化器的参数p，得到模型的参数信息
    w_rows = p.size()[1:4].numel()  # 设置水印矩阵的行
    w_cols = embed_dim  # 设置水印矩阵的列
    if wtype == 'random':
        w = np.random.randn(w_rows, w_cols)  # 矩阵由随机数填充

    elif wtype == 'direct':
        '''矩阵将被初始化为全零，并在每一列中随机选择一个元素设置为1。'''
        w = np.zeros((w_rows, w_cols), dtype=None)

        rand_idx_gen = random_index_generator(w_rows)
        for col in range(w_cols):
            w[next(rand_idx_gen)][col] = 1.

    elif wtype == 'diff':
        '''矩阵将被初始化为全零，并在每一列中随机选择两个元素，一个设置为1，一个设置为-1'''
        w = np.zeros((w_rows, w_cols), dtype=None)

        rand_idx_gen = random_index_generator(w_rows)
        for col in range(w_cols):
            w[next(rand_idx_gen)][col] = 1.
            w[next(rand_idx_gen)][col] = -1.
    else:
        raise Exception('wtype="{}" is not supported'.format(wtype))
    return w


def save_wmark_signatures(prefix, target_blk_id, w, b):
    """保存权重矩阵(w)和水印的签名信息(b) ，b通常包含了水印的标识或其他信息"""

    fname_w = prefix + '_layer{}_w.npy'.format(target_blk_id)
    fname_b = prefix + '_layer{}_b.npy'.format(target_blk_id)
    np.save(fname_w, w)
    np.save(fname_b, b)



if __name__ == '__main__':

    device = torch.device('cuda')

    settings_json_fname = r"C:\Users\zhangzhun\Desktop\snn\DNN-watermark-torch-master\config\train_random_min.json"  # sys.argv[1]
    train_settings = json.load(open(settings_json_fname))

    if not os.path.isdir(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    # read parameters
    batch_size = train_settings['batch_size']
    nb_epoch = train_settings['epoch']
    scale = train_settings['scale']
    embed_dim = train_settings['embed_dim']
    N = train_settings['N']
    k = train_settings['k']
    target_blk_id = train_settings['target_blk_id']
    base_modelw_fname = train_settings['base_modelw_fname']
    wtype = train_settings['wmark_wtype']
    randseed = train_settings['randseed'] if 'randseed' in train_settings else 'none'

    #  模型名称前缀
    hist_hdf_path = 'WTYPE_{}/DIM{}/SCALE{}/N{}K{}B{}EPOCH{}/TBLK{}'.format(
        wtype, embed_dim, scale, N, k, batch_size, nb_epoch, target_blk_id)
    modelname_prefix = os.path.join(RESULT_PATH, 'wrn_' + hist_hdf_path.replace('/', '_'))

    # load dataset for learning
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    # initialize process for Watermark
    b = np.ones((1, embed_dim))
    model = Wide_ResNet(10, 4)
    model = model.to(device)

    # build wm
    w = build_wm(model, target_blk_id, embed_dim, wtype)

    # training process
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)

    # 根据训练的不同阶段来动态地调整学习率。
    # [60, 120, 160]:epoch的列表，当训练达到这些epoch时，学习率将进行调整。
    # gamma = 0.2: 学习率缩放因子，学习率将在每个epoch（60, 120, 160）处乘以这个因子以减小学习率。
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 120, 160], gamma=0.2)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    # 检查是否有预训练模型的权重参数文件可供加载
    if len(base_modelw_fname) > 0:
        model.load_state_dict(torch.load(base_modelw_fname))

    print("Finished building")
    print(w)  # water_mark

    train(model, optimizer, trainloader, w, b, k, nb_epoch, target_blk_id)

    # validate training accuracy
    model.eval()
    loss_meter = 0
    acc_meter = 0
    with torch.no_grad():
        for d, t in testloader:
            data = d.to(device)  # data
            target = t.to(device)  # label
            predict_class = model(data)
            loss_meter += F.cross_entropy(predict_class, target, reduction='sum').item()
            predict_class = predict_class.max(1, keepdim=True)[1]
            acc_meter += predict_class.eq(target.view_as(predict_class)).sum().item()

    print('Test loss:', loss_meter)

    print('Test accuracy:', acc_meter / len(testloader.dataset))

    # write model parameters to file
    torch.save(model.state_dict(), modelname_prefix + '.pth')

    # write watermark matrix and embedded signature to file
    if target_blk_id > 0:
        save_wmark_signatures(modelname_prefix, target_blk_id, w, b)
