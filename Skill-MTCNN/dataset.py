import pickle
import configparser
from dataset_utils import split_v2, create_setx_for_Th, create_train_set, select_img_of_same_skill, load_data, split_test_set, split_Th_set

class AVAImages:
    def __init__(self):
        self.batch_index = 0
        self.batch_size = 0
        self.batch_index_max = 0

        self.test_batch_index = 0
        self.test_batch_size = 0
        self.test_batch_index_max = 0

        self.th_batch_index = 0
        self.th_batch_size = 0
        self.th_batch_index_max = 0

        self.train_set_x = 0
        self.train_set_y = 0
        self.val_set_x = 0
        self.val_set_y = 0

    def create_skillmtcnn_dataset(self, root_dir="../../AVA/"):
        split_v2(root_dir=root_dir)
        create_setx_for_Th(root_dir=root_dir)
        create_train_set(root_dir=root_dir, batch_size=32, size=227, if_write=False)

    def create_batch_set(self, root_dir='dataset/', flag="test"):
        if flag == "test":
            split_test_set(root_dir)
        elif flag == "Th":
            split_Th_set(root_dir)

    def load_next_batch_quicker(self, flag="train"):
        """
        :param max_block: max block index in dir
        :return: 1 -> last batch
        """
        if flag == "train":
            with open('dataset/train_raw/train_set_x_' + str(self.batch_index) + '.pkl', 'rb') as f:
                x = pickle.load(f)
            with open('dataset/train_raw/train_set_y_' + str(self.batch_index) + '.pkl', 'rb') as f:
                y = pickle.load(f)
            if self.batch_index == self.batch_index_max:
                # print('last train batch!')
                self.batch_index = 0
                flag = 1
            else:
                self.batch_index += 1
                flag = 0
            return x, y, flag
        elif flag == "test":
            with open('dataset/testbatch/test_set_x_' + str(self.test_batch_index) + '.pkl', 'rb') as f:
                x = pickle.load(f)
            with open('dataset/testbatch/test_set_y_' + str(self.test_batch_index) + '.pkl', 'rb') as f:
                y = pickle.load(f)
            if self.test_batch_index == self.test_batch_index_max:
                # print('last test batch!')
                self.test_batch_index = 0
                flag = 1
            else:
                self.test_batch_index += 1
                flag = 0
            return x, y, flag
        elif flag == "Th":
            with open('dataset/Thbatch/Th_x_' + str(self.th_batch_index) + '.pkl', 'rb') as f:
                x = pickle.load(f)
            with open('dataset/Thbatch/Th_y_' + str(self.th_batch_index) + '.pkl', 'rb') as f:
                y = pickle.load(f)
            if self.th_batch_index == self.th_batch_index_max:
                # print('last Th batch!')
                self.th_batch_index = 0
                flag = 1
            else:
                self.th_batch_index += 1
                flag = 0
            return x, y, flag

    def load_dataset(self):
        # self.train_set_x, self.train_set_y = load_data(flag="train")
        # self.test_set_x, self.test_set_y = load_data(flag="test")
        self.val_set_x, self.val_set_y = load_data(flag="val")
        # self.Th_x, self.Th_y = load_data(flag="Th")

    def read_batch_cfg(self, task="Skill-MTCNN"):
        # 创建管理对象
        conf = configparser.ConfigParser()
        # 读ini文件
        conf.read("config.ini")  # python3
        if task == "TestBatch":
            self.test_batch_index_max = conf.getint(task, "batch_index_max")
            self.test_batch_size = conf.getint(task, "batch_size")
        elif task == "TrainBatch":
            self.batch_index_max = conf.getint(task, "batch_index_max")
            self.batch_size = conf.getint(task, "batch_size")
        elif task == "ThBatch":
            self.th_batch_index_max = conf.getint(task, "batch_index_max")
            self.th_batch_size = conf.getint(task, "batch_size")

    def select_img(self, skill_index=10):
        """
        从test set中，选出使用某特定摄影技巧的所有图片
        :return:
        """
        x, y = select_img_of_same_skill(skill_index)
        with open('x.pkl', 'wb') as f:
            pickle.dump(x, f)
        with open('y.pkl', 'wb') as f:
            pickle.dump(y, f)


#########################调用##################
# data = AVAImages()
# data.create_skillmtcnn_dataset()
# data.create_batch_set(flag="Th")
