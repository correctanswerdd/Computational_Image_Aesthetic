import cv2
import numpy as np
import pickle
import configparser
import os
from progressbar import ProgressBar


def read_challenges(filedir="AVA_dataset/challenges.txt"):
    f = open(filedir, "r")
    lines = f.readlines()
    for line in lines:
        print(line)


def read_tags(filedir="AVA_dataset/tags.txt"):
    f = open(filedir, "r")
    lines = f.readlines()
    for line in lines:
        print(line)


class AVAImages:
    def __init__(self):
        self.image_url = []
        self.score = []
        self.cat = []
        self.challenge = []
        self.batch_index = 0
        self.batch_index_max = 0
        self.batch_num = 0
        self.batch_size = 0

        # split
        self.train_set_x = 0
        self.train_set_y = 0
        self.test_set_x = 0
        self.test_set_y = 0
        self.val_set_x = 0
        self.val_set_y = 0

    def check_data(self,
                   filedir="AVA_dataset/AVA.txt",
                   newfiledir="AVA_dataset/AVA_check.txt",
                   imgdir="AVA_dataset/images/"):
        with open(filedir, "r") as f:
            with open(newfiledir, "a") as fw:
                lines = f.readlines()
                i = 0

                progress = ProgressBar(len(lines))
                progress.start()
                for line in lines:
                    seg = line.split(" ")
                    url = seg[1]
                    img = cv2.imread(imgdir + url + ".jpg")
                    if img is not None:
                        fw.write(line)
                    i += 1
                    progress.show_progress(i)
                progress.end()
                print("finish the check!")

    def create_train_set(self, batch_size,
                         read_dir='AVA_data_score_bi/',
                         train_set_dir='train_raw/'):
        self.batch_size = batch_size
        self.read_data(read_dir=read_dir, flag=1)  # training set only

        i = 0
        total = self.train_set_x.shape[0]
        it = total // batch_size
        self.batch_index_max = it
        while i < it:
            train_block = self.urls_to_images_no_check(
                self.train_set_x[i * batch_size: (i+1) * batch_size], flag=0)
            with open(read_dir + train_set_dir + "train_set_x_" + str(i) + ".pkl", "wb") as f:
                pickle.dump(train_block, f)
            with open(read_dir + train_set_dir + "train_set_y_" + str(i) + ".pkl", "wb") as f:
                pickle.dump(self.train_set_y[i * batch_size: (i+1) * batch_size], f)
            i += 1
        print('last block!')
        train_block = self.urls_to_images_no_check(
            self.train_set_x[i * batch_size: total])
        with open(read_dir + train_set_dir + "train_set_x_" + str(i) + ".pkl", "wb") as f:
            pickle.dump(train_block, f)
        with open(read_dir + train_set_dir + "train_set_y_" + str(i) + ".pkl", "wb") as f:
            pickle.dump(self.train_set_y[i * batch_size: total], f)

        self.create_conf()

    def create_conf(self):
        curpath = os.path.dirname(os.path.realpath(__file__))
        cfgpath = os.path.join(curpath, "cfg.ini")
        print(cfgpath)  # cfg.ini的路径
        # 创建管理对象
        conf = configparser.ConfigParser()
        # 添加一个select
        conf.add_section("parameter")
        # 往select添加key和value
        conf.set("parameter", "batch_index_max", str(self.batch_index_max))
        conf.set("parameter", "batch_size", str(self.batch_size))
        conf.write(open(cfgpath, "w"))  # 删除原文件重新写入   "a"是追加模式

    def read_batch_cfg(self):
        curpath = os.path.dirname(os.path.realpath(__file__))
        cfgpath = os.path.join(curpath, "cfg.ini")
        print(cfgpath)  # cfg.ini的路径
        # 创建管理对象
        conf = configparser.ConfigParser()
        # 读ini文件
        conf.read(cfgpath, encoding="utf-8")  # python3
        self.batch_index_max = int(conf.get("parameter", "batch_index_max"))
        self.batch_size = int(conf.get("parameter", "batch_size"))

    def split_data(self,
                   data_type: str,
                   filedir="AVA_dataset/AVA.txt",
                   save_dir='AVA_data_score/',
                   train_prob=0.8,
                   test_prob=0.1,
                   val_prob=0.1
                   ):
        """
        :param filedir:

        :var image_url: int array
        :var score: float list
        :var cat: 60*2 one-hot array
        :var challenge: int list
        :var batch_index: int
        """

        if data_type == "score":
            with open(filedir, "r") as f:
                lines = f.readlines()
                for line in lines:
                    seg = line.split(" ")
                    seg = list(map(int, seg))
                    self.image_url.append(seg[1])
                    self.score.append(self.cal_score(seg[2:12]))

                # to array
                self.image_url = np.array(self.image_url)
                self.score = np.array(self.score)

                # shuffle
                total = self.image_url.shape[0]
                index = [i for i in range(total)]
                np.random.shuffle(index)
                np.random.shuffle(index)
                np.random.shuffle(index)
                self.image_url = self.image_url[index]
                self.score = self.score[index]

                # split
                print("train set: 0->{end}/{total}".format(end=int(total * train_prob), total=total))
                self.train_set_x = self.image_url[0: int(total * train_prob)]
                self.train_set_y = self.score[0: int(total * train_prob)]

                # url to image
                print("loading test images ... {st}->{ed}".format(st=int(total * train_prob),
                                                                  ed=int(total * (train_prob + test_prob))))
                self.test_set_x, self.test_set_y = self.urls_to_images(
                    self.image_url[int(total * train_prob): int(total * (train_prob + test_prob))],
                    self.score[int(total * train_prob): int(total * (train_prob + test_prob))]
                )
                print("loading validation images ... {st}->{ed}".format(
                    st=int(total * (train_prob + test_prob)), 
                    ed=int(total * (train_prob + test_prob + val_prob))))
                self.val_set_x, self.val_set_y = self.urls_to_images(
                    self.image_url[int(total * (train_prob + test_prob)):
                                   int(total * (train_prob + test_prob + val_prob))],
                    self.score[int(total * (train_prob + test_prob)):
                               int(total * (train_prob + test_prob + val_prob))]
                )
                self.save_data(save_dir=save_dir)
        elif data_type == "score_bi":
            with open(filedir, "r") as f:
                lines = f.readlines()
                for line in lines:
                    seg = line.split(" ")
                    seg = list(map(int, seg))
                    self.image_url.append(seg[1])
                    if self.cal_score(seg[2:12]) >= 5:
                        self.score.append([0, 1])
                    else:
                        self.score.append([1, 0])

                # to array
                self.image_url = np.array(self.image_url)
                self.score = np.array(self.score)

                # shuffle
                total = self.image_url.shape[0]
                index = [i for i in range(total)]
                np.random.shuffle(index)
                np.random.shuffle(index)
                np.random.shuffle(index)
                self.image_url = self.image_url[index]
                self.score = self.score[index]

                # split
                print("train set: 0->{end}/{total}".format(end=int(total * train_prob), total=total))
                self.train_set_x = self.image_url[0: int(total * train_prob)]
                self.train_set_y = self.score[0: int(total * train_prob)]

                # url to image
                print("loading test images ... {st}->{ed}".format(st=int(total * train_prob),
                                                                  ed=int(total * (train_prob + test_prob))))
                self.test_set_x = self.urls_to_images_no_check(
                    self.image_url[int(total * train_prob): int(total * (train_prob + test_prob))],
                )
                self.test_set_y = self.score[int(total * train_prob): int(total * (train_prob + test_prob))]

                print("loading validation images ... {st}->{ed}".format(
                    st=int(total * (train_prob + test_prob)), 
                    ed=int(total * (train_prob + test_prob + val_prob))))
                self.val_set_x= self.urls_to_images_no_check(
                    self.image_url[int(total * (train_prob + test_prob)):
                                   int(total * (train_prob + test_prob + val_prob))],
                )
                self.val_set_y = self.score[int(total * (train_prob + test_prob)):
                                            int(total * (train_prob + test_prob + val_prob))]
                self.save_data(save_dir='AVA_data_score_bi/')
                self.cal_prob()
        elif data_type == "tag":
            with open(filedir, "r") as f:
                lines = f.readlines()
                for line in lines:
                    seg = line.split(" ")
                    seg = list(map(int, seg))
                    if seg[12] != 0 and seg[13] != 0:
                        self.image_url.append(seg[1])
                        p1 = np.eye(66)[seg[12] - 1]
                        p2 = np.eye(66)[seg[13] - 1]
                        self.cat.append(np.hstack((p1, p2)))

                # to array
                self.image_url = np.array(self.image_url)
                self.cat = np.array(self.cat)

                # shuffle
                total = self.image_url.shape[0]
                index = [i for i in range(total)]
                np.random.shuffle(index)
                np.random.shuffle(index)
                np.random.shuffle(index)
                self.image_url = self.image_url[index]
                self.cat = self.cat[index]

                # split
                print("train set: 0->{end}/{total}".format(end=int(total * train_prob), total=total))
                self.train_set_x = self.image_url[0: int(total * train_prob)]
                self.train_set_y = self.cat[0: int(total * train_prob)]

                # url to image
                print("loading test images ... {st}->{ed}".format(st=int(total * train_prob),
                                                                  ed=int(total * (train_prob + test_prob))))
                self.test_set_x, self.test_set_y = self.urls_to_images(
                    self.image_url[int(total * train_prob): int(total * (train_prob + test_prob))],
                    self.cat[int(total * train_prob): int(total * (train_prob + test_prob))]
                )
                print("loading validation images ... {st}->{ed}".format(st=int(total * (train_prob + test_prob)),
                                                                        ed=int(total * (train_prob + test_prob + val_prob))))
                self.val_set_x, self.val_set_y = self.urls_to_images(
                    self.image_url[int(total * (train_prob + test_prob)):
                                   int(total * (train_prob + test_prob + val_prob))],
                    self.cat[int(total * (train_prob + test_prob)):
                             int(total * (train_prob + test_prob + val_prob))]
                )
                self.save_data(save_dir='AVA_data_tag/')

    def save_data(self, save_dir='AVA_data_score/'):
        with open(save_dir + "train_set_x.pkl", "wb") as f:
            pickle.dump(self.train_set_x, f)
        with open(save_dir + "train_set_y.pkl", "wb") as f:
            pickle.dump(self.train_set_y, f)
        with open(save_dir + "test_set_x.pkl", "wb") as f:
            pickle.dump(self.test_set_x, f)
        with open(save_dir + "test_set_y.pkl", "wb") as f:
            pickle.dump(self.test_set_y, f)
        with open(save_dir + "val_set_x.pkl", "wb") as f:
            pickle.dump(self.val_set_x, f)
        with open(save_dir + "val_set_y.pkl", "wb") as f:
            pickle.dump(self.val_set_y, f)

    def read_data(self, read_dir='AVA_data_score/', flag=0):
        """
        read data from dir;
        :param read_dir:
        :param flag: 0 -> do not read training set; 1 -> training set only
        :return:
        """
        if flag == 1:
            with open(read_dir + 'train_set_x.pkl', 'rb') as f:
                self.train_set_x = pickle.load(f)
            with open(read_dir + 'train_set_y.pkl', 'rb') as f:
                self.train_set_y = pickle.load(f)
        else:
            with open(read_dir + 'test_set_x.pkl', 'rb') as f:
                self.test_set_x = pickle.load(f)
            with open(read_dir + 'test_set_y.pkl', 'rb') as f:
                self.test_set_y = pickle.load(f)
            with open(read_dir + 'val_set_x.pkl', 'rb') as f:
                self.val_set_x = pickle.load(f)
            with open(read_dir + 'val_set_y.pkl', 'rb') as f:
                self.val_set_y = pickle.load(f)

    def cal_score(self, score_distribution: list):
        """以均分作为图像的分数"""
        total = score_distribution[0] * 1 \
            + score_distribution[1] * 2 \
            + score_distribution[2] * 3 \
            + score_distribution[3] * 4 \
            + score_distribution[4] * 5 \
            + score_distribution[5] * 6 \
            + score_distribution[6] * 7 \
            + score_distribution[7] * 8 \
            + score_distribution[8] * 9 \
            + score_distribution[9] * 10
        return total / sum(score_distribution)

    def urls_to_images_no_check(self, urls, file_dir='AVA_dataset/images/', flag=1):
        images = []
        i = 0

        if flag == 1:
            progress = ProgressBar(urls.shape[0])
            progress.start()
            for url in urls:
                img = cv2.imread(file_dir + str(url) + ".jpg")
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                images.append(img)
                i += 1
                progress.show_progress(i)
            progress.end()
        else:
            for url in urls:
                img = cv2.imread(file_dir + str(url) + ".jpg")
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                images.append(img)
                i += 1
        return np.array(images) / 225.

    def urls_to_images(self, urls, y_ori, filedir="AVA_dataset/images/", flag=1):
        # print('{name}: {age}'.format(age=24, name='TaoXiao'))  # 通过关键字传递
        # print("before url to images: {url_num}, {y_num}".format(url_num=urls.shape[0], y_num=y_ori_1.shape[0]))
        images = []
        y = []
        i = 0

        if flag == 1:
            progress = ProgressBar(urls.shape[0])
            progress.start()

            for url in urls:
                img = cv2.imread(filedir + str(url) + ".jpg")
                if img is not None:
                    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                    images.append(img)
                    y.append(y_ori[i])
                i += 1
                progress.show_progress(i)
            progress.end()
        else:
            for url in urls:
                img = cv2.imread(filedir + str(url) + ".jpg")
                if img is not None:
                    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                    images.append(img)
                    y.append(y_ori[i])
                i += 1

        # print("after url to images: {url_num}, {y_num}".format(url_num=len(images), y_num=len(y1)))
        return np.array(images) / 225., np.array(y)

    def load_next_batch(self, batch_size: int):
        """
        :param batch_size:
        :return: x_batch, y_batch

        :var x_batch: (batch size, width, height, channels)
        """
        batch_end_flag = 0
        self.batch_index = 0
        self.batch_num = 0
        # print("loading batch images ...")

        if self.batch_num == 0:
            self.batch_num = self.train_set_x.shape[0] // batch_size
        if self.batch_index == self.batch_num:
            print("last batch!")
            x_urls = self.train_set_x[self.batch_index * batch_size: self.train_set_x.shape[0]]
            y = self.train_set_y[self.batch_index * batch_size: self.train_set_x.shape[0]]
            self.batch_index = 0
            batch_end_flag = 1
        else:
            # print("batch {id1}->{id2}/{total}".format(id1=self.batch_index, id2=self.batch_index+1,total=self.batch_num))
            x_urls = self.train_set_x[self.batch_index * batch_size: (self.batch_index + 1) * batch_size]
            y = self.train_set_y[self.batch_index * batch_size: (self.batch_index + 1) * batch_size]
            self.batch_index += 1
        x_batch, y_batch = self.urls_to_images(x_urls, y, flag=0)
        return x_batch, y_batch, batch_end_flag

    def load_next_batch_quicker(self, read_dir='AVA_data_score_bi/'):
        """

        :param max_block: max block index in dir
        :return: 1 -> last batch
        """
        with open(read_dir + 'train_raw/train_set_x_' + str(self.batch_index) + '.pkl', 'rb') as f:
            x = pickle.load(f)
        with open(read_dir + 'train_raw/train_set_y_' + str(self.batch_index) + '.pkl', 'rb') as f:
            y = pickle.load(f)
        if self.batch_index == self.batch_index_max:
            print('last batch!')
            self.batch_index = 0
            flag = 1
        else:
            self.batch_index += 1
            flag = 0
        return x, y, flag




    def cal_prob(self):
        print('high quality images in train set: {num}'.format(
            num=np.sum(np.argmax(self.train_set_y, axis=1)) / self.train_set_y.shape[0]))
        print('high quality images in test set: {num}'.format(
            num=np.sum(np.argmax(self.test_set_y, axis=1)) / self.test_set_y.shape[0]))
        print('high quality images in validation set: {num}'.format(
            num=np.sum(np.argmax(self.val_set_y, axis=1)) / self.val_set_y.shape[0]))