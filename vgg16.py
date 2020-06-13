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
from preprocess.regularization import Regularization


plt.ion()  # interactive mode
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


# Randomly pick sample to training sets
def move_file(origin_dir, aim_dir, rate):
    path = os.listdir(origin_dir)  # 获取图像的原始路径
    file_num = len(path)
    pick_num = int(file_num * rate)
    sample = random.sample(path, pick_num)  # 随机抽取指定数量的样本

    if not os.path.exists(aim_dir):
        os.makedirs(aim_dir)

    for name in sample:
        shutil.copy(origin_dir + name, aim_dir + name)
        # print('remove ' + name + ' to target folder')
    return


def delete_file(aim_dir):
    for path in aim_dir:
        shutil.rmtree(path)  # 删除该路径下的文件夹
        os.mkdir(path)  # 重建该文件夹


# Define training function
def train_model(model, criterion, optimizer, epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch{}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            tp = tn = 0.0
            fn = fp = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if weight_decay > 0:
                        loss = loss + reg_loss(model)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                # running_loss += loss.item()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 0是非暴力，1是暴力
                # predict 和 label 同时为1
                tp += ((preds == 1) & (labels.data == 1)).cpu().sum()
                # predict 和 label 同时为0
                tn += ((preds == 0) & (labels.data == 0)).cpu().sum()
                # predict 0 label 1
                fn += ((preds == 0) & (labels.data == 1)).cpu().sum()
                # predict 1 label 0
                fp += ((preds == 1) & (labels.data == 0)).cpu().sum()

            if phase == 'train':
                # scheduler.step()  # Adjust LR
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Acc', epoch_acc, epoch)

            # deep copy the model
            else:
                precision = tp / (tp + fp)  # 在预测为暴力的场景中真正暴力场景的比率
                recall = tp / (tp + fn)  # 所有暴力场景中被判断为暴力场景的比率
                # F1 = 2 * precision * recall / (precision + recall)
                acc = (tp + tn) / (tp + tn + fp + fn)
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Precision: {:.4f} Recall: {:.4f}'.format(phase, precision, recall))
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                writer.add_scalar('Val/Loss', epoch_loss, epoch)
                writer.add_scalar('Val/Acc', epoch_acc, epoch)

                if acc > best_acc:
                    best_acc = acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    writer.close()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# 测试函数
def test(testloader, model):
    since = time.time()
    correct = 0
    total = 0
    tp = tn = 0.0
    fp = fn = 0.0
    model.eval()

    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)  # 从预测值中选出概率最大的那个
        total += labels.size(0)  # 计算标签总数
        correct += (preds == labels.data).sum()  # 计算正确结果个数

        # 0是非暴力，1是暴力
        # predict 和 label 同时为1
        tp += ((preds == 1) & (labels.data == 1)).cpu().sum()
        # predict 和 label 同时为0
        tn += ((preds == 0) & (labels.data == 0)).cpu().sum()
        # predict 0 label 1
        fn += ((preds == 0) & (labels.data == 1)).cpu().sum()
        # predict 1 label 0
        fp += ((preds == 1) & (labels.data == 0)).cpu().sum()

    time_elapsed = time.time() - since
    precision = tp / (tp + fp)  # 在预测为暴力的场景中真正暴力场景的比率
    recall = tp / (tp + fn)  # 所有暴力场景中被判断为暴力场景的比率
    acc = (tp + tn) / (tp + tn + fp + fn)

    print('-' * 10)
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Accuracy on the test set: %d %%" % (100 * correct/total))  # 100为百分比
    print('Acc: {:.4f}'.format(100 * acc))
    print('Precision: {:.4f} Recall: {:.4f}'.format(precision, recall))


# import data
data_dir = 'data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
writer = SummaryWriter('VGG16_adam')
print('Data Ready!')


# 固定卷积层参数
vggnet = models.vgg16(pretrained=True)
for param in vggnet.parameters():
    param.requires_grad = False

# 修改后的模型结构参数会默认更新
vggnet.classifier = nn.Sequential(
    nn.Linear(512*7*7, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 2),)

CUDA = torch.cuda.is_available()
if CUDA:
    vggnet = vggnet.cuda()

# Add regularization
weight_decay = 0  # 0.001
if weight_decay > 0:
    reg_loss = Regularization(vggnet, weight_decay, p=1).to(device)
else:
    print('No regularization')

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()

# optimizer_ft = optim.SGD(vggnet.classifier.parameters(), lr=0.001, momentum=0.9)
optimizer_fn = optim.Adam(vggnet.classifier.parameters(), lr=0.001, weight_decay=1e-5)  # 1e-5

# Training and evaluation
model_ft = train_model(vggnet, loss_fn, optimizer_fn, epochs=25)
torch.save(model_ft.state_dict(), 'vgg16_adam.pkl')


# test_model = models.vgg16(pretrained=False)
# # test_model.classifier[-1] = nn.Linear(4096, 2)
#
# test_model.classifier = nn.Sequential(
#     nn.Linear(512*7*7, 4096),
#     nn.ReLU(inplace=True),
#     nn.Dropout(),
#     nn.Linear(4096, 4096),
#     nn.ReLU(inplace=True),
#     nn.Dropout(),
#     nn.Linear(4096, 2),)
#
# test_model.load_state_dict(torch.load('vgg16_adam.pkl'))
# CUDA = torch.cuda.is_available()
# if CUDA:
#     test_model = test_model.cuda()
#
# testset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
# testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True)
#
# test(testloader, test_model)
