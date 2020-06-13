import cv2
import os


# 从.avi 类型的视频中提取图像
def split_frames(sourceFileName):

    # 视频路径
    video_path = os.path.join('./data/train/blood', sourceFileName + '.avi')
    # outPutDirName = './data/fight/fight_frame/' + sourceFileName + '/'
    outPutDirName = './data/train/blood_set/'

    # 如果文件目录不存在则创建目录
    if not os.path.exists(outPutDirName):
        os.makedirs(outPutDirName)

    camera = cv2.VideoCapture(video_path)  # 打开视频文件
    time = 0
    count = 0
    frequency = 12  # 设置帧率（这里每1秒提取一帧）

    while True:
        success, image = camera.read()  # success表示是否读取到视频，image是当前帧的图像数据
        if not success:
            break
        time = time + 1
        if time % frequency == 0:
            count = count + 1
            cv2.imwrite(outPutDirName + sourceFileName + '_' + str(time) + '.jpg', image)
            print(time)
    camera.release()


# 从mp4视频中提取图像
def splitFrames_mp4(sourceFileName):

    # 视频路径
    video_path = os.path.join('./data1/test/new', sourceFileName + '.mp4')
    # outputDirName = './data1/val/blood_frame/' + sourceFileName + '/'
    outputDirName = './data1/test/new_set/'

    # 如果文件目录不存在则创建目录
    if not os.path.exists(outputDirName):
        os.makedirs(outputDirName)

    camera = cv2.VideoCapture(video_path)  # 打开视频文件
    time = 0
    count = 0
    frequency = 6  # 设置帧率（这里每1秒提取一帧）

    while True:
        success, image = camera.read()  # success表示是否读取到视频，image是当前帧的图像数据
        if not success:
            break
        time = time + 1
        if time % frequency == 0:
            count = count + 1
            cv2.imwrite(outputDirName + sourceFileName + '_' + str(time) + '.jpg', image)
            print(time)
    camera.release()


# 主函数入口
if __name__ == '__main__':
    im_file = './data1/test/new'

    # 遍历文件夹内的所有视频
    for im_name in os.listdir(im_file):
        suffix_file = os.path.splitext(im_name)[-1]
        if suffix_file == '.mp4':
            print('~~~~~~~~~~ 从.mp4 视频提取图像 ~~~~~~~~~~~~~~~')
            sourceFileName = os.path.splitext(im_name)[0]
            splitFrames_mp4(sourceFileName)

        elif suffix_file == '.avi':
            print('~~~~~~~~~~ 从.avi 视频提取图像 ~~~~~~~~~~~~~~~')

            sourceFileName = os.path.splitext(im_name)[0]
            split_frames(sourceFileName)

    print('Finish!')
