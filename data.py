import cv2
import numpy as np
import pickle
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
        self.batch_num = 0

        # split
        self.train_set_x = 0
        self.train_set_y = 0
        self.test_set_x = 0
        self.test_set_y = 0
        self.val_set_x = 0
        self.val_set_y = 0

    def split_data(self,
                   type: str,
                   filedir="AVA_dataset/AVA.txt",
                   train_prob=0.1,
                   test_prob=0.003,
                   val_prob=0.0001
                   ):
        """
        :param filedir:

        :var image_url: int array
        :var score: float list
        :var cat: 60*2 one-hot array
        :var challenge: int list
        :var batch_index: int
        """

        if type == "score":
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
                print("train set: 0->{end}/{total}".format(end=int(total*train_prob), total=total))
                self.train_set_x = self.image_url[0: int(total * train_prob)]
                self.train_set_y = self.score[0: int(total * train_prob)]

                # url to image
                print("loading test images ... {st}->{ed}".format(st=int(total * train_prob),
                                                                  ed=int(total * (train_prob + test_prob))))
                self.test_set_x, self.test_set_y = self.urls_to_images(
                    self.image_url[int(total * train_prob): int(total * (train_prob + test_prob))],
                    self.score[int(total * train_prob): int(total * (train_prob + test_prob))]
                )
                print("loading validation images ... {st}->{ed}".format(st=int(total * (train_prob + test_prob)),
                                                                        ed=int(total * (train_prob + test_prob + val_prob))))
                self.val_set_x, self.val_set_y = self.urls_to_images(
                    self.image_url[int(total * (train_prob + test_prob)):
                                   int(total * (train_prob + test_prob + val_prob))],
                    self.score[int(total * (train_prob + test_prob)):
                               int(total * (train_prob + test_prob + val_prob))]
                )
                self.save_data(save_dir='AVA_data_score')
        elif type == "tag":
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
                print("train set: 0->{end}/{total}".format(end=int(total*train_prob), total=total))
                self.train_set_x = self.image_url[0: int(total*train_prob)]
                self.train_set_y = self.cat[0: int(total*train_prob)]

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
                self.save_data(save_dir='AVA_data_tag')

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

    def read_data(self, read_dir='AVA_data_score/'):
        with open(read_dir + 'train_set_x.pkl', 'rb') as f:
            self.train_set_x = pickle.load(f)
        with open(read_dir + 'train_set_y.pkl', 'rb') as f:
            self.train_set_y = pickle.load(f)
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
        return np.array(images)/225., np.array(y)

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
            x_urls = self.train_set_x[self.batch_index*batch_size: (self.batch_index+1)*batch_size]
            y = self.train_set_y[self.batch_index * batch_size: (self.batch_index + 1)*batch_size]
            self.batch_index += 1
        x_batch, y_batch = self.urls_to_images(x_urls, y, flag=0)
        return x_batch, y_batch, batch_end_flag


