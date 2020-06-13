import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter
from torchvision import models, transforms, datasets
import copy
import time
from preprocess.regularization import Regularization


plt.ion()   # interactive mode
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
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


# Define training function
def train_model(model, criterion, optimizer, scheduler, epochs=25):
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
                model.eval()  # Lock BN & Dropout

            running_loss = 0.0
            running_corrects = 0
            confusion_matrix = torch.zeros(2, 2)

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

                    for t, p in zip(preds.view(-1), labels.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

                    loss = criterion(outputs, labels)
                    if weight_decay > 0:
                        loss = loss + reg_loss(model)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                scheduler.step(epoch_acc)  # Adjust LR

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                # writer.add_scalar('Train/Loss', epoch_loss, epoch)
                # writer.add_scalar('Train/Acc', epoch_acc, epoch)

            # deep copy the model
            else:
                precision = (confusion_matrix.diag() / confusion_matrix.sum(1))[1]
                recall = (confusion_matrix.diag() / confusion_matrix.sum(0))[1]
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                print('{} Precision: {:.4f} Recall: {:.4f}'.format(phase, precision, recall))
                # writer.add_scalar('Val/Loss', epoch_loss, epoch)
                # writer.add_scalar('Val/Acc', epoch_acc, epoch)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # writer.close()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# 测试函数
def test(testloader, model):
    since = time.time()
    model.eval()

    acc = 0.0
    running_corrects = 0
    a_precision = a_recall = 0.0
    b_precision = b_recall = 0.0
    confusion_matrix = torch.zeros(2, 2)

    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)  # 从预测值中选出概率最大的那个

        for t, p in zip(preds.view(-1), labels.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        a_precision = (confusion_matrix.diag() / confusion_matrix.sum(1))[1]
        a_recall = (confusion_matrix.diag() / confusion_matrix.sum(0))[1]

        b_precision = (confusion_matrix.diag() / confusion_matrix.sum(1))[0]
        b_recall = (confusion_matrix.diag() / confusion_matrix.sum(0))[0]

        running_corrects += torch.sum(preds == labels.data)
        acc = running_corrects.double() / dataset_sizes['test']

    time_elapsed = time.time() - since

    print('-' * 10)
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print("Accuracy on the test set: %d %%" % (100 * correct/total))  # 100为百分比
    print("Accuracy on the test set: {:.4f}".format(acc))  # 100为百分比
    print('Vio Precision: {:.4f} Recall: {:.4f}'.format(a_precision, a_recall))
    print('non-Vio Precision: {:.4f} Recall: {:.4f}'.format(b_precision, b_recall))


# import data
data_dir = 'data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
               for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes
# writer = SummaryWriter('ResNet50')


# 固定卷积层参数
resnet = models.resnet50(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False

# 修改后的模型结构参数会默认更新
# resnet.fc = nn.Sequential(nn.Linear(2048, 2))
resnet.fc = nn.Sequential(nn.Linear(2048, 1000),
                          nn.ReLU(inplace=True),
                          nn.Dropout(0.5),
                          nn.Linear(1000, 1000),
                          nn.ReLU(inplace=True),
                          nn.Dropout(0.5),
                          nn.Linear(1000, 2),
                          )

resnet = resnet.to(device)

# Add regularization
weight_decay = 1e-5
if weight_decay > 0:
    reg_loss = Regularization(resnet, weight_decay, p=1).to(device)
else:
    print('No regularization')


# Loss function & optimizer

loss_fn = nn.CrossEntropyLoss()
# optimizer_fn = optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)
optimizer_fn = optim.Adam(resnet.fc.parameters(), lr=0.001, weight_decay=0)  # 1e-5

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_fn, mode='max', patience=5, factor=0.5)

# Training and evaluation
# model_ft = train_model(resnet, loss_fn, optimizer_fn, scheduler, epochs=50)
# torch.save(model_ft.state_dict(), 'res50_L1_dp.pkl')

# test
resnet.load_state_dict(torch.load('res50_L1_dp.pkl'))
test(dataloaders['test'], resnet)
