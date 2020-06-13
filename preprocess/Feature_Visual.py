
import torch
from torch import nn
from skimage import io
import skimage.transform
import torchvision
from torchvision import models, transforms, datasets
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 自定义网络  Input:(150, 150)
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6272, 512),
            nn.Linear(512, 2))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展开成一维向量
        x = self.classifier(x)

        return x


# 特征图提取
class FeaturesExtractor:
    features = None

    def __init__(self, model, layer_num):
        """
        :param model: 可视化的模型 (需包含features容器)
        :param layer_num: 目标层 (可由model.features查看)
        """
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


def preprocess_img(image, transform):
    img = skimage.transform.resize(image, (150, 150))
    img = np.asarray(img, dtype=np.float32)
    return transform(img)


# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 读取图片并处理为4D张量
image_dir = './images/horse.jpg'
img = io.imread(image_dir)
transform = transforms.ToTensor()
inputs = preprocess_img(img, transform)

inputs = inputs.unsqueeze(0)
inputs = inputs.to(device)
# print(inputs.shape)  # [1,3,150,150]

# 加载模型
model = torch.load('MyNet.pth')
model.to(device)  # 将模型放入GPU
conv_out = FeaturesExtractor(model.features, 0)  # 提取卷积层输出(0, 4, 7, 10)
x = model(inputs)

conv_out.remove()
activation = conv_out.features
print(activation.shape)

# Visualization
# fig = plt.figure(figsize=(20, 20))
# fig.subplots_adjust(wspace=0.5, hspace=0.5)
# for i in range(30):
#     ax = fig.add_subplot(5, 6, i+1, xticks=[], yticks=[])
#     ax.imshow(activation[0][i].detach().numpy())
#
# plt.show()

image_per_row = 16
n_features = activation.shape[1]  # 32
size = activation.shape[-1]  # 148

n_cols = n_features // image_per_row  # 4
display_grid = np.zeros((size * n_cols, size * image_per_row))


for col in range(n_cols):
    for row in range(image_per_row):
        channel_image = activation[0][col * image_per_row + row].detach().numpy()
        channel_image -= channel_image.mean()
        channel_image /= channel_image.std()
        channel_image *= 64
        channel_image += 128
        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
        display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

    scale = 1./size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title("Conv_4")
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto')
    plt.show()

