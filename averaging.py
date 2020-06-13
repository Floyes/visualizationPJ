import os, random, shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
from torchvision import models, transforms, datasets
import copy
import time
from preprocess.se_modules import se_resnet50
from preprocess.regularization import Regularization


plt.ion()   # interactive mode
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# 测试函数
def test(testloader, model1, model2):
    """

    :param testloader:
    :param model1: VGG16
    :param model2: SEResNet
    :return:
    """
    since = time.time()
    model1.eval()
    model2.eval()

    correct1 = correct2 = correct3 = 0
    precision1 = precision2 = precision3 = 0.0
    recall1 = recall2 = recall3 = 0.0
    acc1 = acc2 = acc3 = 0.0

    confusion_matrix1 = torch.zeros(2, 2)
    confusion_matrix2 = torch.zeros(2, 2)
    confusion_matrix3 = torch.zeros(2, 2)
    w1 = 0.8
    w2 = 0.2

    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        output1 = model1(images)
        output2 = model2(images)
        output3 = output1 * w1 + output2 * w2

        _, pred1 = torch.max(output1, 1)  # VGG16
        _, pred2 = torch.max(output2, 1)  # SERes
        _, pred3 = torch.max(output3, 1)  # 加权

        # VGG16
        for t, p in zip(pred1.view(-1), labels.view(-1)):
            confusion_matrix1[t.long(), p.long()] += 1

        precision1 = (confusion_matrix1.diag() / confusion_matrix1.sum(1))[1]
        recall1 = (confusion_matrix1.diag() / confusion_matrix1.sum(0))[1]

        correct1 += torch.sum(pred1 == labels.data)
        acc1 = correct1.double() / dataset_sizes['test']

        # SERes50
        for t, p in zip(pred2.view(-1), labels.view(-1)):
            confusion_matrix2[t.long(), p.long()] += 1

        precision2 = (confusion_matrix2.diag() / confusion_matrix2.sum(1))[1]
        recall2 = (confusion_matrix2.diag() / confusion_matrix2.sum(0))[1]

        correct2 += torch.sum(pred2 == labels.data)
        acc2 = correct2.double() / dataset_sizes['test']

        # Mixing
        for t, p in zip(pred3.view(-1), labels.view(-1)):
            confusion_matrix3[t.long(), p.long()] += 1

        precision3 = (confusion_matrix3.diag() / confusion_matrix3.sum(1))[1]
        recall3 = (confusion_matrix3.diag() / confusion_matrix3.sum(0))[1]

        correct3 += torch.sum(pred3 == labels.data)
        acc3 = correct3.double() / dataset_sizes['test']

    time_elapsed = time.time() - since

    print('-' * 10)
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("VGG16 p: {:.4f} r: {:.4f} acc: {:.4f}".format(precision1, recall1, acc1))
    print("SERes p: {:.4f} r: {:.4f} acc: {:.4f}".format(precision2, recall2, acc2))
    print("Mixing p: {:.4f} r: {:.4f} acc: {:.4f}".format(precision3, recall3, acc3))


# import data
data_dir = 'data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True)
               for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes


# 固定卷积层参数
vggnet = models.vgg16(pretrained=False)
vggnet.classifier[-1] = nn.Linear(4096, 2)
vggnet = vggnet.to(device)


se_resnet = se_resnet50(num_classes=1_000, pretrained=False)  # 在Imagenet上预训练的参数
se_resnet.fc = nn.Sequential(nn.Linear(2048, 2))
se_resnet = se_resnet.to(device)


# # Add regularization
# weight_decay = 0
# if weight_decay > 0:
#     reg_loss = Regularization(se_resnet, weight_decay, p=1).to(device)
# else:
#     print('No regularization')


# test
vggnet.load_state_dict(torch.load('vgg16_L1.pkl'))
se_resnet.load_state_dict(torch.load('seres50_L1.pkl'))

test(dataloaders['test'], vggnet, se_resnet)

