import pickle
from dataset_utils import split, split_v2, create_setx_for_Th, create_train_set, select_img_of_same_skill, load_data

class AVAImages:
    def __init__(self):
        self.batch_index = 0
        self.batch_index_max = 0
        self.batch_size = 0

        self.train_set_x = 0
        self.train_set_y = 0
        self.test_set_x = 0
        self.test_set_y = 0
        self.val_set_x = 0
        self.val_set_y = 0
        self.Th_x = 0
        self.Th_y = 0

    def create_skillmtcnn_dataset(self, root_dir="../../AVA/"):
        split_v2(root_dir=root_dir)
        create_setx_for_Th(root_dir=root_dir)
        create_train_set(root_dir=root_dir, batch_size=32, size=227, if_write=False)

    def load_next_batch_quicker(self, read_dir='dataset/'):
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

    def load_dataset(self):
        # self.train_set_x, self.train_set_y = load_data(flag="train")
        self.test_set_x, self.test_set_y = load_data(flag="test")
        self.val_set_x, self.val_set_y = load_data(flag="val")
        self.Th_x, self.Th_y = load_data(flag="Th")

    def read_batch_cfg(self, task="Skill-MTCNN"):
        # 创建管理对象
        conf = configparser.ConfigParser()
        # 读ini文件
        conf.read("config.ini")  # python3
        self.batch_index_max = conf.getint(task, "batch_index_max")
        self.batch_size = conf.getint(task, "batch_size")

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