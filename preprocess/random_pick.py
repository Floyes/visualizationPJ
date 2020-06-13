from statistics import mode
import os, random, shutil


def move_file(origin_dir, aim_dir, pick_num):
    path = os.listdir(origin_dir)  # 获取图像的原始路径
    # file_num = len(path)
    # pick_num = int(file_num * rate)
    sample = random.sample(path, pick_num)  # 随机抽取指定数量的样本

    if not os.path.exists(aim_dir):
        os.makedirs(aim_dir)

    for name in sample:
        shutil.move(origin_dir + name, aim_dir + name)
        print('remove ' + name + ' to ' + aim_dir)
    return


def delete_file(aim_dir):
    for path in aim_dir:
        shutil.rmtree(path)  # 删除该路径下的文件夹
        os.mkdir(path)  # 重建该文件夹


if __name__ == '__main__':

    # # 从训练集中分出验证集
    # file_dir = ['./database/violence/train/', './database/no_violence/train/']
    # tar_dir = ['./data/val/violence/', './data/val/no_violence/']
    #
    # for pth in file_dir:
    #     if pth == './database/violence/train/':
    #         tar_folder = tar_dir[0]
    #         num = 2000  # violence val 2000 sets
    #     else:
    #         tar_folder = tar_dir[1]
    #         num = 2000  # no_violence val 2000 sets
    #     move_file(pth, tar_folder, num)
    # delete_file(tar_dir)

    # 从测试总集中分出测试集
    test_dir = ['./database/no_violence/test/']
    aim_dir = ['./data/val/no_violence/', './data/test/no_violence/']

    for pth in test_dir:
        if pth == './database/no_violence/test/':
            aim_folder = aim_dir[1]
            num = 2000  # violence test 2000 sets
        else:
            aim_folder = aim_dir[1]
            num = 2000  # no_violence test 2000 sets
        move_file(pth, aim_folder, num)

