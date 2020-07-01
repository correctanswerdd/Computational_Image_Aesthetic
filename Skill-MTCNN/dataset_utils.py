import numpy as np
import cv2
import os
import pickle
import configparser
from progressbar import ProgressBar


def split(root_dir="./", test_prob=0.2, val_prob=0.05, train_prob=0.75):
    """
    函数功能：
    将test.txt中的image url划分为train/test/val set，每一个set都含输入x和输出标签y
    其中train set的x依然为image url，test和val set的x为AVA_dataset/images中的jpg文件，读取后缩放为227x227x3的array
    三个set的y均为(num, 24)的矩阵

    函数结果：
    1- 新建dataset文件夹
    2- 创建数据集文件
        dataset/train_set_x.pkl(shape=(train_size,))
        dataset/train_set_y.pkl(shape=(train_size, 24))
        dataset/test_set_x.pkl(shape=(test_size, 227, 227, 3))
        dataset/test_set_y.pkl(shape=(test_size, 24))
        dataset/val_set_x.pkl(shape=(val_size, 227, 227, 3))
        dataset/val_set_x.pkl(shape=(val_size, 24))
    """

    index2score_dis = {}
    with open(root_dir + 'AVA_dataset/AVA_check.txt', "r") as f:
        lines = f.readlines()
        for line in lines:
            seg = line.split(" ")
            seg = list(map(int, seg))
            score_dis = np.array(seg[2: 12]) / sum(seg[2: 12])
            index2score_dis[seg[1]] = score_dis

    with open(root_dir + "AVA_dataset/style_image_lists/" + "test.txt", "r") as f:
        test_x = f.readlines()
    X = np.array([int(i[0:-2]) for i in test_x])
    dis = np.array([index2score_dis[int(i[0:-2])] for i in test_x])
    with open(root_dir + "AVA_dataset/style_image_lists/" + "test_y.txt", "r") as f:
        lines = f.readlines()
        sty = []
        for line in lines:
            seg = line.split(" ")
            seg = list(map(int, seg))
            sty.append(seg)
        sty = np.array(sty)
    Y = np.hstack((dis, sty))

    # shuffle
    total = X.shape[0]
    index = [i for i in range(total)]
    np.random.shuffle(index)
    np.random.shuffle(index)
    np.random.shuffle(index)

    print("train set: 0->{end}/{total}".format(end=int(total * train_prob), total=total))
    train_set_x = X[0: int(total * train_prob)]
    train_set_y = Y[0: int(total * train_prob)]

    # url to image
    print("loading test images ... {st}->{ed}".format(st=int(total * train_prob),
                                                      ed=int(total * (train_prob + test_prob))))
    test_set_x = urls_to_images_no_check(
        X[int(total * train_prob): int(total * (train_prob + test_prob))],
        size=227
    )
    test_set_y = Y[int(total * train_prob): int(total * (train_prob + test_prob))]

    print("loading validation images ... {st}->{ed}".format(
        st=int(total * (train_prob + test_prob)),
        ed=int(total * (train_prob + test_prob + val_prob))))
    val_set_x = urls_to_images_no_check(
        X[int(total * (train_prob + test_prob)): int(total * (train_prob + test_prob + val_prob))],
        size=227
    )
    val_set_y = Y[int(total * (train_prob + test_prob)):
                       int(total * (train_prob + test_prob + val_prob))]

    data = train_set_x, train_set_y, test_set_x, test_set_y, val_set_x, val_set_y
    save_data(save_dir="./dataset/", data=data)


def split_v2(root_dir, test_prob=0.2, val_prob=0.05, train_prob=0.75):
    """
    函数功能：
    将test.txt+train.txt中的image url划分为train/test/val set，每一个set都含输入x和输出标签y
    其中train set的x依然为image url，test和val set的x为AVA_dataset/images中的jpg文件，读取后缩放为227x227x3的array
    三个set的y均为(num, 24)的矩阵

    函数结果：
    1- 新建dataset文件夹
    2- 创建数据集文件
        dataset/train_set_x.pkl(shape=(train_size,))
        dataset/train_set_y.pkl(shape=(train_size, 24))
        dataset/test_set_x.pkl(shape=(test_size, 227, 227, 3))
        dataset/test_set_y.pkl(shape=(test_size, 24))
        dataset/val_set_x.pkl(shape=(val_size, 227, 227, 3))
        dataset/val_set_x.pkl(shape=(val_size, 24))
    """

    index2score_dis = {}
    with open(root_dir + 'AVA_dataset/AVA_check.txt', "r") as f:
        lines = f.readlines()
        for line in lines:
            seg = line.split(" ")
            seg = list(map(int, seg))
            score_dis = np.array(seg[2: 12]) / sum(seg[2: 12])
            index2score_dis[seg[1]] = score_dis

    with open(root_dir + "AVA_dataset/style_image_lists/" + "test.jpgl", "r") as f:
        x1 = f.readlines()
    with open(root_dir + "AVA_dataset/style_image_lists/" + "train.jpgl", "r") as f:
        x2 = f.readlines()
    X = np.array([int(i[0:-1]) for i in x1+x2])
    dis = np.array([index2score_dis[int(i[0:-1])] for i in x1+x2])
    with open(root_dir + "AVA_dataset/style_image_lists/" + "test.multilab", "r") as f:
        lines = f.readlines()
        sty1 = []
        for line in lines:
            seg = line.split(" ")
            seg = list(map(int, seg))
            sty1.append(seg)
    with open(root_dir + "AVA_dataset/style_image_lists/" + "train.lab", "r") as f:
        lines = f.readlines()
        style = np.eye(14)
        sty2 = []
        for line in lines:
            sty2.append(style[int(line[:-1])-1])
    Y = np.hstack((dis, np.array(sty1+sty2)))

    # shuffle
    total = X.shape[0]
    index = [i for i in range(total)]
    np.random.shuffle(index)
    np.random.shuffle(index)
    np.random.shuffle(index)

    print("train set: 0->{end}/{total}".format(end=int(total * train_prob), total=total))
    train_set_x = X[0: int(total * train_prob)]
    train_set_y = Y[0: int(total * train_prob)]

    # url to image
    print("loading test images ... {st}->{ed}".format(st=int(total * train_prob),
                                                      ed=int(total * (train_prob + test_prob))))
    test_set_x = urls_to_images_no_check(
        X[int(total * train_prob): int(total * (train_prob + test_prob))],
        root_dir=root_dir,
        size=227
    )
    test_set_y = Y[int(total * train_prob): int(total * (train_prob + test_prob))]

    print("loading validation images ... {st}->{ed}".format(
        st=int(total * (train_prob + test_prob)),
        ed=int(total * (train_prob + test_prob + val_prob))))
    val_set_x = urls_to_images_no_check(
        X[int(total * (train_prob + test_prob)): int(total * (train_prob + test_prob + val_prob))],
        root_dir=root_dir,
        size=227
    )
    val_set_y = Y[int(total * (train_prob + test_prob)):
                  int(total * (train_prob + test_prob + val_prob))]

    data = train_set_x, train_set_y, test_set_x, test_set_y, val_set_x, val_set_y
    save_data(save_dir="dataset/", data=data)


def split_test_set(root_dir, save_dir='testbatch/', batch_size=32, if_write=False):
    test_set_x, test_set_y = load_data(root_dir, flag="test")

    folder = os.path.exists(save_dir)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(save_dir)  # makedirs 创建文件时如果路径不存在会创建这个路径

    i = 0
    total = test_set_x.shape[0]
    it = total // batch_size
    if total % batch_size == 0:
        it -= 1
    batch_index_max = it  # it是最后一个batch的起始index

    progress = ProgressBar(it)
    progress.start()
    while i < it:
        with open(root_dir + save_dir + "test_set_x_" + str(i) + ".pkl", "wb") as f:
            pickle.dump(test_set_x[i * batch_size: (i + 1) * batch_size, :, :, :], f)
        with open(root_dir + save_dir + "test_set_y_" + str(i) + ".pkl", "wb") as f:
            pickle.dump(test_set_y[i * batch_size: (i + 1) * batch_size], f)
        i += 1
        progress.show_progress(i)
    progress.end()

    print('last block!')
    with open(root_dir + save_dir + "test_set_x_" + str(i) + ".pkl", "wb") as f:
        pickle.dump(test_set_x[i * batch_size: total, :, :, :], f)
    with open(root_dir + save_dir + "test_set_y_" + str(i) + ".pkl", "wb") as f:
        pickle.dump(test_set_y[i * batch_size: total], f)

    if if_write:
        data = batch_size, batch_index_max, test_set_x.shape[0]
        write_conf(data, task="TestBatch")


def urls_to_images_no_check(urls, root_dir, file_dir='AVA_dataset/images/', size=224, flag=1):
    images = []
    i = 0

    if flag == 1:
        progress = ProgressBar(urls.shape[0])
        progress.start()
        for url in urls:
            img = cv2.imread(root_dir + file_dir + str(url) + ".jpg")
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
            images.append(img)
            i += 1
            progress.show_progress(i)
        progress.end()
    else:
        for url in urls:
            img = cv2.imread(root_dir + file_dir + str(url) + ".jpg")
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
            images.append(img)
            i += 1
    return np.array(images) / 225.


def load_data(read_dir='dataset/', flag="val"):
    """
    read data from dir;
    :param read_dir:
    :param flag: 0 -> do not read training set; 1 -> training set only
    :return:
    """
    if flag == "train":
        with open(read_dir + 'train_set_x.pkl', 'rb') as f:
            train_set_x = pickle.load(f)
        with open(read_dir + 'train_set_y.pkl', 'rb') as f:
            train_set_y = pickle.load(f)
        return train_set_x, train_set_y
    elif flag == "test":
        with open(read_dir + 'test_set_x.pkl', 'rb') as f:
            test_set_x = pickle.load(f)
        with open(read_dir + 'test_set_y.pkl', 'rb') as f:
            test_set_y = pickle.load(f)
        return test_set_x, test_set_y
    elif flag == "val":
        with open(read_dir + 'val_set_x.pkl', 'rb') as f:
            val_set_x = pickle.load(f)
        with open(read_dir + 'val_set_y.pkl', 'rb') as f:
            val_set_y = pickle.load(f)
        return val_set_x, val_set_y
    elif flag == "Th":
        with open(read_dir + 'Th_x.pkl', 'rb') as f:
            Th_x = pickle.load(f)
        with open(read_dir + 'Th_y.pkl', 'rb') as f:
            Th_y = pickle.load(f)
        return Th_x, Th_y


def save_data(data, save_dir='dataset/', save_train=True, save_test=True, save_val=True):
    folder = os.path.exists(save_dir)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(save_dir)  # makedirs 创建文件时如果路径不存在会创建这个路径

    train_set_x, train_set_y, test_set_x, test_set_y, val_set_x, val_set_y = data

    if save_train:
        with open(save_dir + "train_set_x.pkl", "wb") as f:
            pickle.dump(train_set_x, f)
        with open(save_dir + "train_set_y.pkl", "wb") as f:
            pickle.dump(train_set_y, f)
    if save_test:
        with open(save_dir + "test_set_x.pkl", "wb") as f:
            pickle.dump(test_set_x, f)
        with open(save_dir + "test_set_y.pkl", "wb") as f:
            pickle.dump(test_set_y, f)
    if save_val:
        with open(save_dir + "val_set_x.pkl", "wb") as f:
            pickle.dump(val_set_x, f)
        with open(save_dir + "val_set_y.pkl", "wb") as f:
            pickle.dump(val_set_y, f)


def write_conf(data, task="Skill-MTCNN"):
    # 创建管理对象
    conf = configparser.ConfigParser()
    conf.read("config.ini", encoding="utf-8")

    if task == "TestBatch":
        batch_size, batch_index_max, total = data
        # 修改某key的value
        conf.set(task, "batch_index_max", str(batch_index_max))
        conf.set(task, "batch_size", str(batch_size))
        conf.set(task, "total", str(total))
    else:
        batch_size, batch_index_max = data
        # 修改某key的value
        conf.set(task, "batch_index_max", str(batch_index_max))
        conf.set(task, "batch_size", str(batch_size))

    conf.write(open("config.ini", "a"))  # 删除原文件重新写入   "a"是追加模式


def create_train_set(batch_size,
                     root_dir,
                     size=224,
                     if_write=True,
                     read_dir='dataset/',
                     train_set_dir='train_raw/'):
    """
    函数功能：
    将train set的x按照batch size划分，并把每一个batch都从image url转换为array存储

    函数结果：
    1- 创建文件夹dataset/train_raw
    2- 创建数据集文件。最后一个batch的size不一定为batch size
        dataset/train_raw/train_set_x_0.pkl(shape=(batch_size, 227, 227, 3))
        dataset/train_raw/train_set_y_0.pkl(shape=(batch_size, 24))
        ... ...
        dataset/train_raw/train_set_x_?.pkl(shape=(last_batch_size, 227, 227, 3))
        dataset/train_raw/train_set_y_?.pkl(shape=(last_batch_size, 24))
        ? -> 最后一个batch的index。若total%batch_size==0，?=total//batch_size - 1；否则为total//batch_size
    3- 填写config.ini文件中的last_batch_index和batch_size
    """
    folder = os.path.exists(read_dir + train_set_dir)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(read_dir + train_set_dir)  # makedirs 创建文件时如果路径不存在会创建这个路径

    train_set_x, train_set_y = load_data(read_dir=read_dir, flag="train")  # training set only

    i = 0
    total = train_set_x.shape[0]
    it = total // batch_size
    if total % batch_size == 0:
        it -= 1
    batch_index_max = it  # it是最后一个batch的起始index

    progress = ProgressBar(it)
    progress.start()
    while i < it:
        train_block = urls_to_images_no_check(
            train_set_x[i * batch_size: (i+1) * batch_size],
            root_dir=root_dir,
            size=size, flag=0)
        with open(read_dir + train_set_dir + "train_set_x_" + str(i) + ".pkl", "wb") as f:
            pickle.dump(train_block, f)
        with open(read_dir + train_set_dir + "train_set_y_" + str(i) + ".pkl", "wb") as f:
            pickle.dump(train_set_y[i * batch_size: (i+1) * batch_size], f)
        i += 1
        progress.show_progress(i)
    progress.end()

    print('last block!')
    train_block = urls_to_images_no_check(
        train_set_x[i * batch_size: total],
        root_dir=root_dir,
        size=size, flag=0)
    with open(read_dir + train_set_dir + "train_set_x_" + str(i) + ".pkl", "wb") as f:
        pickle.dump(train_block, f)
    with open(read_dir + train_set_dir + "train_set_y_" + str(i) + ".pkl", "wb") as f:
        pickle.dump(train_set_y[i * batch_size: total], f)

    if if_write:
        data = batch_size, batch_index_max
        write_conf(data)


def create_setx_for_Th(root_dir, read_dir='dataset/', size=227, Th_prob=0.1):
    """
    函数功能：
    从train set中选出0.1作为Th

    函数结果：
    1- 创建Th文件
        dataset/Th_x.pkl(shape=(train_size*0.1, 227, 227, 3))
        dataset/Th_y.pkl(shape=(train_size*0.1, 24))
    """
    train_set_x, train_set_y = load_data(read_dir=read_dir, flag="train")  # training set only
    total = train_set_x.shape[0]
    index = [i for i in range(total)]
    np.random.shuffle(index)
    np.random.shuffle(index)
    np.random.shuffle(index)
    setx = train_set_x[0: int(total * Th_prob)]
    sety = train_set_y[0: int(total * Th_prob)]
    setx = urls_to_images_no_check(setx, root_dir=root_dir, size=size, flag=1)
    with open(read_dir + "Th_x.pkl", "wb") as f:
        pickle.dump(setx, f)
    with open(read_dir + "Th_y.pkl", "wb") as f:
        pickle.dump(sety, f)


def dis2mean(score_distribution):
    assert score_distribution.shape[1] == 10

    mean = score_distribution[:, 0] * 1 \
           + score_distribution[:, 1] * 2 \
           + score_distribution[:, 2] * 3 \
           + score_distribution[:, 3] * 4 \
           + score_distribution[:, 4] * 5 \
           + score_distribution[:, 5] * 6 \
           + score_distribution[:, 6] * 7 \
           + score_distribution[:, 7] * 8 \
           + score_distribution[:, 8] * 9 \
           + score_distribution[:, 9] * 10
    return mean / np.sum(score_distribution, axis=1)


def cal_prob(data):
    """
    函数功能：
    分别计算train/test/val set中的正负样本比例
    """
    train_set_y, test_set_y, val_set_y = data
    y_mean = dis2mean(train_set_y[:, 0: 10])
    y = np.int64(y_mean >= 5)
    print('high quality images in train set: {num}'.format(
        num=np.sum(y) / y.shape[0]))
    y_mean = dis2mean(test_set_y[:, 0: 10])
    y = np.int64(y_mean >= 5)
    print('high quality images in test set: {num}'.format(
        num=np.sum(y) / y.shape[0]))
    y_mean = dis2mean(val_set_y[:, 0: 10])
    y = np.int64(y_mean >= 5)
    print('high quality images in validation set: {num}'.format(
        num=np.sum(y) / y.shape[0]))


def select_img_of_same_skill(skill_index=10):
    """
    从test set中，选出使用某特定摄影技巧的所有图片
    :return:
    """
    test_set_x, test_set_y = load_data(flag="test")
    x = []
    y = []
    count = 0
    for i in range(test_set_y.shape[0]):
        if test_set_y[i][10 + skill_index] == 1:
            count += 1
            x.append(test_set_x[i])
            y.append(test_set_y[i])
    print("total={n}".format(n=count))
    return np.array(x, dtype=float), np.array(y, dtype=float)


def get_index2score_dis(root_dir):
    with open(root_dir + "AVA_dataset/AVA_check.txt", "r") as f_ava:
        line_ava = f_ava.readlines()
    index2score_dis = {}
    for line in line_ava:
        seg = line.split(" ")
        seg = list(map(int, seg))
        score_dis = np.array(seg[2: 12]) / sum(seg[2: 12])
        index2score_dis[seg[1]] = score_dis
    return index2score_dis