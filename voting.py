import os
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models, transforms, datasets
import time
from preprocess.se_modules import se_resnet50,se_vgg16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建预处理数据集的字典
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# 测试函数
def test(test_loader, model1, model2, model3):
    since = time.time()

    acc = 0.0
    running_corrects = 0
    precision = recall = 0.0
    confusion_matrix = torch.zeros(2, 2)

    model1.eval()
    model2.eval()
    model3.eval()

    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        output1 = model1(images)
        output2 = model2(images)
        output3 = model3(images)

        outputs = [output1, output2, output3]

        prediction = []
        for output in outputs:
            _, predict = torch.max(output, 1)
            prediction.append(predict)

        # 返回预测值中票数更多的结果
        if torch.equal(prediction[0], prediction[1]):
            best_pred = prediction[0]
        elif torch.equal(prediction[0], prediction[2]):
            best_pred = prediction[2]
        else:
            best_pred = prediction[1]

        for t, p in zip(best_pred.view(-1), labels.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        precision = (confusion_matrix.diag() / confusion_matrix.sum(1))[1]
        recall = (confusion_matrix.diag() / confusion_matrix.sum(0))[1]

        # running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(best_pred == labels.data)
        acc = running_corrects.double() / dataset_sizes

    time_elapsed = time.time() - since
    print('-' * 10)
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Accuracy on the test set: {:.4f}".format(acc))  # 100为百分比
    print('Vio Precision: {:.4f} Recall: {:.4f}'.format(precision, recall))


data_directory = 'data'
test_set = datasets.ImageFolder(os.path.join(data_directory, 'test'), data_transforms['test'])
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
dataset_sizes = len(test_set)

print('data ready.')

# 加载模型
vggnet = models.vgg16(pretrained=False)
vggnet.classifier[-1] = nn.Linear(4096, 2)
vggnet.load_state_dict(torch.load('vgg16_L1.pkl'))
vggnet = vggnet.to(device)

sevgg = se_vgg16(pretrained=False)
sevgg.classifier[-1] = nn.Linear(4096, 2)
sevgg.load_state_dict(torch.load('sevgg16.pkl'))
sevgg = sevgg.to(device)


res50 = models.resnet50(pretrained=False)
res50.fc = nn.Sequential(nn.Linear(2048, 2))
res50.load_state_dict(torch.load('res50_L1.pkl'))
res50 = res50.to(device)

# seres50 = se_resnet50(num_classes=1_000, pretrained=False)
# seres50.fc = nn.Sequential(nn.Linear(2048, 2))
# seres50.load_state_dict(torch.load('seres50_L1.pkl'))
# seres50 = seres50.to(device)

# 测试
test(testloader, vggnet, sevgg, res50)
