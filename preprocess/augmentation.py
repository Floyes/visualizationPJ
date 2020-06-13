import os,random
import numpy as np
import cv2
from PIL import Image


# 随机噪声
def add_noise(file_path):
    image = cv2.imread(file_path)
    file_name = os.path.splitext(file_path)[0]
    for i in range(1000):
        image[random.randint(0, image.shape[0]-1)][random.randint(0,image.shape[1]-1)][:] = 255
    cv2.imwrite(file_name + '_noise.jpg', image)
    print('Finish!')


def clamp(pv):
    """防止溢出"""
    if pv > 255:
        return 255
    elif pv < 0:
        return 0
    else:
        return pv


# 高斯噪声
def gaussian_noise_demo(image):
    h, w, c = image.shape
    for row in range(0, h):
        for col in range(0, w):
            s = np.random.normal(0, 15, 3)  # 产生随机数，每次产生三个
            b = image[row, col, 0]  # blue
            g = image[row, col, 1]  # green
            r = image[row, col, 2]  # red
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])

    cv2.imwrite('gau_panda.jpg', image)


def data_augmentation(sourceFileName, file_path):

    # 图像路径
    pic_path = os.path.join(file_path, sourceFileName + '.jpg')
    outputDirName = './data/val_augmentation/'

    # 如果文件目录不存在则创建目录
    if not os.path.exists(outputDirName):
        os.makedirs(outputDirName)

    image = cv2.imread(pic_path)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_resize = cv2.resize(image, (224, 224))  # 修改尺寸
    # image_flip = cv2.flip(image, 1)  # 水平翻转
    image_flip = cv2.flip(image, 0)  # 垂直翻转
    cv2.imwrite(outputDirName + sourceFileName + '_yflip_' + '.jpg', image_flip)


def rotate_bound(sourceFileName, file_path, angle):
    # 图像路径
    pic_path = os.path.join(file_path, sourceFileName + '.jpg')
    outputDirName = './database/rotate/'

    # 如果文件目录不存在则创建目录
    if not os.path.exists(outputDirName):
        os.makedirs(outputDirName)

    image = cv2.imread(pic_path)

    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    rotated = cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))
    cv2.imwrite(outputDirName + sourceFileName + '_rotate_' + '.jpg', rotated)


# 主函数入口
if __name__ == '__main__':
    im_file = './database/train/non_violence'

    # 遍历文件夹内的所有图片
    for im_name in os.listdir(im_file):
        suffix_file = os.path.splitext(im_name)[-1]
        if suffix_file == '.jpg':
            sourceFileName = os.path.splitext(im_name)[0]
            print(sourceFileName + ' process on')
            rotate_bound(sourceFileName, im_file, angle=5)
            # data_augmentation(sourceFileName, im_file)
        else:
            pass

    print('Finish!')
