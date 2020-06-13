import os
import torch
from torch import nn
from torchvision import models
from skimage import io
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from interpretability.guided_back_propagation import GuidedBackPropagation


def get_net(net_name, weight_path=None):
    """
    加载网络模型
    :param net_name: 网络名称
    :param weight_path: 预训练权重路径
    :return
    """
    pretrain = weight_path is None  # 未指定权重路径，加载默认预训练权重
    if net_name in ['vgg', 'vgg16']:
        net = models.vgg16(pretrained=pretrain)
        net.classifier[-1] = nn.Linear(4096, 2)
    elif net_name == 'vgg19':
        net = models.vgg19(pretrained=pretrain)
    elif net_name in ['resnet', 'resnet50']:
        net = models.resnet50(pretrained=pretrain)
        net.fc = nn.Sequential(nn.Linear(2048, 2))
    elif net_name == 'resnet_drop':
        net = models.resnet50(pretrained=pretrain)
        net.fc = nn.Sequential(nn.Linear(2048, 1000),
                               nn.ReLU(inplace=True),
                               nn.Dropout(0.5),
                               nn.Linear(1000, 2),
                               )
    else:
        raise ValueError('invalid network name {}'.format(net_name))

    if weight_path is not None:
        net.load_state_dict(torch.load(weight_path))  # 加载指定路径的权重参数

    return net


def get_last_conv_name(net):
    """
    获取最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, i in net.named_modules():
        if isinstance(i, nn.Conv2d):
            layer_name = name
    return layer_name


def preprocess_image(img):
    """
    将输入图像进行归一化处理
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = img.copy()
    image -= mean
    image /= std

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # transpose HWC > CHW 通道优先
    image = image[np.newaxis, ...]  # 增加batch维
    input = torch.tensor(image, requires_grad=True)

    return input



def show_cam_on_image(img, mask):
    """
    生成CAM图
    :param img: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return :tuple(cam,heatmap)
    """
    # 将mask转换为heatmap
    mask = np.uint8(255 * mask)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    # heatmap[np.where(mask <= 150)] = 0  # 设置热力图阈值
    heatmap = np.float32(heatmap) / 255

    # 将heatmap叠加到原图
    # cam = heatmap + np.float32(img)
    cam = cv2.addWeighted(src1=img, alpha=0.8, src2=heatmap, beta=0.4, gamma=0)
    # cam = cam - np.max(np.min(cam), 0)
    cam = cam / np.max(cam)
    cam = cam[:, :, ::-1]  # gbr to rgb
    heatmap = heatmap[:, :, ::-1]

    # cv2.imwrite("./results/cam_on_1.jpg", np.uint8(255 * cam))
    # cv2.imwrite("./results/heatmap.jpg", np.uint8(255 * heatmap))

    return cam, heatmap


def get_gb(grad):
    """
    转换梯度通道
    :param grad: tensor,[3,H,W]
    :return:
    """
    grad = grad.data.numpy()  # 标准化处理
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, input_image_name, network, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)


def main(args):

    # 输入图像预处理
    img = io.imread(args.image_path)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    inputs = preprocess_image(img)

    image_dict = {}  # 将输出CAM图像存为字典
    net = get_net(args.network, args.weight_path)  # 获取网络
    layer_name = get_last_conv_name(net) if args.layer_name is None else args.layer_name

    # Grad-CAM
    grad_cam = GradCAM(net, layer_name)  # 类实例化
    mask = grad_cam(inputs, args.class_id)  # 调用该对象得到 mask
    image_dict['cam'], image_dict['heatmap'] = show_cam_on_image(img, mask)
    # cam, heatmap = show_cam_on_image(img, mask)
    grad_cam.remove_handlers()

    # Grad-CAM++
    # grad_cam_plus_plus = GradCamPlusPlus(net, layer_name)
    # mask_plus_plus = grad_cam_plus_plus(inputs, args.class_id)  # cam mask
    # image_dict['cam++'], image_dict['heatmap++'] = show_cam_on_image(img, mask_plus_plus)
    # grad_cam_plus_plus.remove_handlers()

    # Guided BP
    gbp = GuidedBackPropagation(net)
    inputs.grad.zero_()  # 梯度置零
    grad = gbp(inputs)
    gb = get_gb(grad)
    image_dict['gb'] = gb

    # 生成Guided Grad-CAM
    cam_gb = gb * mask[..., np.newaxis]  # CAM
    # cam_gb = gb * mask_plus_plus[..., np.newaxis]  # CAM

    cam_gb = cam_gb.copy()
    cam_gb = cam_gb - np.max(np.min(cam_gb), 0)
    cam_gb = cam_gb / np.max(cam_gb)
    cam_gb = np.uint8(255 * cam_gb)
    image_dict['cam_gb'] = cam_gb
    # cv2.imwrite("./results/GBP_beat.jpg", cam_gb)

    save_image(image_dict, os.path.basename(args.image_path), args.network, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='resnet',
                        help='ImageNet classification network')
    parser.add_argument('--image-path', type=str, default='./sample/weapons/wp_1.jpg',
                        help='input image path')
    parser.add_argument('--weight-path', type=str, default='res50.pkl',
                        help='weight path of the model')
    parser.add_argument('--layer-name', type=str, default=None,
                        help='last convolutional layer name')
    parser.add_argument('--class-id', type=int, default=1,
                        help='class id')  # 1是暴力
    parser.add_argument('--output-dir', type=str, default='results',
                        help='output directory to save results')
    arguments = parser.parse_args()

    main(arguments)
