from flyai.train_helper import upload_data, download, sava_train_model  # 因为要蹭flyai的gpu
from dataset import AVAImages
from resnet import resnet_v2_50
import configparser
import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from operator import itemgetter
import pickle
slim = tf.contrib.slim


def condition_a(time, y_outputs, y, ta):
    return tf.less(time, 10)


def condition_b(time, y_outputs, y, tb):
    return tf.less(time, 10)


def body_a(time, y_outputs, y, ta):
    y1 = tf.reduce_sum(y_outputs[:, 0: time], axis=1)
    y2 = tf.reduce_sum(y[:, 0: time], axis=1)
    ta = ta.write(time - 1, tf.multiply(y1, tf.log(tf.divide(y1, tf.add(0.5 * y1, 0.5 * y2)))))
    return time + 1, y_outputs, y, ta


def body_b(time, y_outputs, y, tb):
    y1 = tf.reduce_sum(y_outputs[:, 0: time], axis=1)
    y2 = tf.reduce_sum(y[:, 0: time], axis=1)
    tb = tb.write(time - 1, tf.multiply(y2, tf.log(tf.divide(y2, tf.add(0.5 * y1, 0.5 * y2)))))
    return time + 1, y_outputs, y, tb


class Network(object):
    def __init__(self, input_size: tuple,
                 output_size: int):
        """
        init the network

        :param input_size: w, h, c
        :param output_size: size
        """
        self.input_size = input_size
        self.output_size = output_size

    def score2style(self, inputs):
        with tf.variable_scope("Cor_Matrix"):
            output = slim.fully_connected(inputs, 14, scope='fc')
        return output

    def MTCNN_v2(self, inputs, training=True):
        with tf.variable_scope("Theta"):
            feature_vec, _ = resnet_v2_50(inputs=inputs, num_classes=4096, is_training=training)
        with tf.variable_scope("W"):
            l7_list = [slim.fully_connected(feature_vec, 1) for i in range(self.output_size)]
        return l7_list

    def MTCNN(self, inputs, training=True):
        with tf.variable_scope("Theta"):
            l1 = slim.conv2d(inputs, 96, [11, 11], stride=2, padding='VALID', normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv1')
            l1_bn = tf.layers.batch_normalization(l1, training=training, name='bn1')
            l1_pool = slim.max_pool2d(l1_bn, [3, 3], padding='VALID', scope='pool1')

            l2 = slim.conv2d(l1_pool, 128, [5, 5], stride=1, padding='SAME', normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv2')
            l2_bn = tf.layers.batch_normalization(l2, training=training, name='bn2')
            l2_pool = slim.max_pool2d(l2_bn, [3, 3], padding='SAME', scope='pool2')

            l3 = slim.conv2d(l2_pool, 128, [3, 3], stride=1, padding='SAME', normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv3')

            l4 = slim.conv2d(l3, 256, [3, 3], stride=1, padding='SAME', normalizer_fn=None, activation_fn=tf.nn.relu, scope='conv4')
            l4_bn = tf.layers.batch_normalization(l4, training=training, name='bn4')
            l4_pool = slim.max_pool2d(l4_bn, [3, 3], scope='pool4')

            l4_pool_flat = tf.reshape(l4_pool, [-1, 13 * 13 * 256], name='flat1')
            l5 = slim.fully_connected(l4_pool_flat, 4096)
            l6 = slim.fully_connected(l5, 4096)

        with tf.variable_scope("W"):
            l7_list = [slim.fully_connected(l6, 1) for i in range(self.output_size)]
        return l7_list

    def eval_binary_acc(self, model_path='./model_MTCNN/'):
        dataset = AVAImages()
        dataset.read_data('dataset/', flag="test")
        y_test_mean = dataset.dis2mean(dataset.test_set_y[:, 0: 10])
        x_test, y_test = dataset.test_set_x, np.int64(y_test_mean >= 5)  # 前提test_set_y.shape=(num,)
        y_test = y_test[:, np.newaxis]
        w, h, c = self.input_size
        x = tf.placeholder(tf.float32, [None, w, h, c])
        y_list = self.MTCNN(x, True)  # y_outputs = (None, 24)
        y_outputs = tf.concat(y_list, axis=1)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                y_outputs = sess.run(y_outputs, feed_dict={x: x_test})
                y_outputs_mean = dataset.dis2mean(y_outputs[:, 0: 10])
                y_pred = np.int64(y_outputs_mean >= 5)[:, np.newaxis]
                correct_prediction = tf.equal(y_pred, y_test)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print("test accuracy - %g" % sess.run(accuracy))
            else:
                print('No checkpoing file found')
                return

    def cal_distribution(self, model_path='./model_MTCNN/', image_path='3.jpg'):
        img = cv2.imread(image_path)
        img = cv2.resize(img * 255, (227, 227), interpolation=cv2.INTER_CUBIC)
        img = img[np.newaxis, :]
        w, h, c = self.input_size
        x = tf.placeholder(tf.float32, [None, w, h, c])
        y_list = self.MTCNN(x, True)  # y_outputs = (None, 24)
        y_outputs = tf.concat(y_list, axis=1)

        dataset = AVAImages()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                y_outputs = sess.run(y_outputs, feed_dict={x: img})
                y_outputs_to_one = y_outputs[:, 0: 10] / np.sum(y_outputs[:, 0: 10])
                y_outputs_mean = dataset.dis2mean(y_outputs_to_one)
                plt.title('predict distribution')
                plt.xlabel('vote')
                plt.ylabel('number')
                plt.plot(np.array(range(1, 11)), y_outputs_to_one[0], color="orange",
                         linewidth=1, linestyle=':', label='predict' + str(y_outputs_mean) , marker='o')
                plt.legend(loc=2)  # 图例展示位置，数字代表第几象限
                plt.show()  # 显示图像
            else:
                print('No checkpoing file found')
                return

    def cal_score(self, model_path='./model_MTCNN/'):
        with open("AVA_dataset/style_image_lists/test.txt", "r") as f:
            lines = f.readlines()

        lines = lines[0: 100]
        img_lists = []
        for line in lines:
            img = cv2.imread("AVA_dataset/images/{index}.jpg".format(index=line[:-1]))
            img = cv2.resize(img * 255, (227, 227), interpolation=cv2.INTER_CUBIC)
            img_lists.append(img)
        img_lists = np.array(img_lists)

        w, h, c = self.input_size
        x = tf.placeholder(tf.float32, [None, w, h, c])
        y_list = self.MTCNN(x, True)  # y_outputs = (None, 24)
        y_outputs = tf.concat(y_list, axis=1)

        dataset = AVAImages()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                y_outputs = sess.run(y_outputs, feed_dict={x: img_lists})
                y_outputs_to_one = y_outputs[:, 0: 10] / np.sum(y_outputs[:, 0: 10])
                y_outputs_mean = dataset.dis2mean(y_outputs_to_one)
                for i in range(y_outputs_mean.shape[0]):
                    if y_outputs_mean[i] < 5:
                        print("index={index}, score={score}".format(index=lines[i][:-1], score=y_outputs_mean[i]))
            else:
                print('No checkpoing file found')
                return

    def view_result(self, model_path='./model_MTCNN/'):
        """
        看看skill-mtcnn预测为正/负样本的图都长啥样
        :param model_path:
        :return:
        """
        skills = {0: 'Complementary_Colors',
                  1: 'Duotones',
                  2: 'HDR',
                  3: 'Image_Grain',
                  4: 'Light_On_White',
                  5: 'Long_Exposure',
                  6: 'Macro',
                  7: 'Motion_Blur',
                  8: 'Negative_Image',
                  9: 'Rule_of_Thirds',
                  10: 'Shallow_DOF',
                  11: 'Silhouettes',
                  12: 'Soft_Focus',
                  13: 'Vanishing_Point'}
        count_skills = np.zeros(15, dtype=int)
        if not os.path.exists("./select"):
            os.makedirs("./select")

        with open("AVA_dataset/style_image_lists/test.txt", "r") as f_x:
            urls = f_x.readlines()
        with open("AVA_dataset/style_image_lists/test_y.txt", "r") as f_y:
            img_skills = f_y.readlines()

        with open("AVA_dataset/AVA_check.txt", "r") as f_ava:
            line_ava = f_ava.readlines()
        index2score_dis = {}
        for line in line_ava:
            seg = line.split(" ")
            seg = list(map(int, seg))
            score_dis = np.array(seg[2: 12]) / sum(seg[2: 12])
            index2score_dis[seg[1]] = score_dis
        dis = np.array([index2score_dis[int(i[0:-1])] for i in urls])
        dataset = AVAImages()
        score = dataset.dis2mean(dis)

        w, h, c = self.input_size
        x = tf.placeholder(tf.float32, [None, w, h, c])
        y_list = self.MTCNN(x, True)  # y_outputs = (None, 24)
        y_outputs = tf.concat(y_list, axis=1)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                i = 0
                for url, img_skill in zip(urls, img_skills):
                    img = cv2.imread("AVA_dataset/images/{index}.jpg".format(index=url[0:-1]))
                    img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)
                    img = img[np.newaxis, :] / 225.
                    y_predict = sess.run(y_outputs, feed_dict={x: img})
                    y_outputs_to_one = y_predict[:, 0: 10] / np.sum(y_predict[:, 0: 10])
                    y_outputs_mean = dataset.dis2mean(y_outputs_to_one)
                    if y_outputs_mean[0] > 5 and score[i] <= 5:
                        seg_img_skill = img_skill[0:-1].split(" ")
                        ski_index = np.nonzero(np.array(list(map(int, seg_img_skill))))[0]
                        count_skills[ski_index] += 1
                        keys = list(ski_index)
                        if len(keys) == 0:
                            count_skills[14] += 1
                            print("index={index}, score={score}, skill={skill}".format(
                                index=urls[i][:-1], score=y_outputs_mean, skill="no skill"))
                        else:
                            print("index={index}, score={score}, skill={skill}".format(
                                index=urls[i][:-1], score=y_outputs_mean, skill=itemgetter(*keys)(skills)))
                        # os.system("cp AVA_dataset/images/{index}.jpg select".format(index=url[0:-1]))
                    i += 1
                print(count_skills)
            else:
                print('No checkpoing file found')
                return

    def view_result_v2(self, model_path='./model_MTCNN'):
        dataset = AVAImages()
        dataset.read_data(flag="test")
        X, Y = dataset.test_set_x, dataset.dis2mean(dataset.test_set_y[:, 0:10])
        Y_sort = np.sort(Y)

        # load weights
        w, h, c = self.input_size
        x = tf.placeholder(tf.float32, [None, w, h, c])
        y_list = self.MTCNN(x, True)  # y_outputs = (None, 24)
        y_outputs = tf.concat(y_list, axis=1)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                y_predict = sess.run(y_outputs, feed_dict={x: X})
                y_outputs_to_one = y_predict[:, 0: 10] / np.sum(y_predict[:, 0: 10])
                y_outputs_mean = dataset.dis2mean(y_outputs_to_one)
            else:
                print('No checkpoing file found')
                return
        y_outputs_sort = np.sort(y_outputs_mean)

        # save
        dir = './select/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir + 'y.pkl', 'wb') as f:
            pickle.dump(Y_sort, f)
        with open(dir + 'y_outputs.pkl', 'wb') as f:
            pickle.dump(y_outputs_sort, f)

    def select_img_of_skill_for_ROC(self, skill_index=10):
        """
        从test set中，选出使用某特定摄影技巧的所有图片
        :return:
        """
        dataset = AVAImages()
        dataset.read_data(flag="test")
        x = []
        y = []
        count = 0
        for i in range(dataset.test_set_y.shape[0]):
            if dataset.test_set_y[i][10 + skill_index] == 1:
                count += 1
                x.append(dataset.test_set_x[i])
                y.append(dataset.test_set_y[i])

        print(count)
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        with open('x.pkl', 'wb') as f:
            pickle.dump(x, f)
        with open('y.pkl', 'wb') as f:
            pickle.dump(y, f)

    def load_img_of_skill_for_ROC(self):
        with open('x.pkl', 'rb') as f:
            x = pickle.load(f)
        with open('y.pkl', 'rb') as f:
            y = pickle.load(f)
        return x, y

    def propagate_ROC(self, output, threshold):
        Y_predict = np.zeros(output.shape)
        for i in range(output.shape[0]):
            if output[i] >= threshold:
                Y_predict[i] = 1
            else:
                Y_predict[i] = 0
        return Y_predict

    def draw_and_save_ROC(self, model_path='./model_MTCNN'):
        # load data
        dataset = AVAImages()
        # dataset.read_data(flag="test")
        # X, Y = dataset.test_set_x, np.int64(dataset.dis2mean(dataset.test_set_y[:, 0:10]) >= 5)
        x, y = self.load_img_of_skill_for_ROC()
        X, Y = x, np.int64(dataset.dis2mean(y[:, 0:10]) >= 5)

        # load weights
        w, h, c = self.input_size
        x = tf.placeholder(tf.float32, [None, w, h, c])
        y_list = self.MTCNN(x, True)  # y_outputs = (None, 24)
        y_outputs = tf.concat(y_list, axis=1)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                y_predict = sess.run(y_outputs, feed_dict={x: X})
                y_outputs_to_one = y_predict[:, 0: 10] / np.sum(y_predict[:, 0: 10])
                y_outputs_mean = dataset.dis2mean(y_outputs_to_one)
            else:
                print('No checkpoing file found')
                return

        threshold = np.sort(y_outputs_mean)
        recall = np.zeros(y_outputs_mean.shape)
        FAR = np.zeros(y_outputs_mean.shape)

        for k in range(threshold.shape[0]):
            Y_predict = self.propagate_ROC(y_outputs_mean, threshold[k])  # (m_test,)
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for i in range(Y.shape[0]):
                if Y_predict[i] - Y[i] == 0 and Y[i] == 1:
                    TP = TP + 1
                elif Y_predict[i] - Y[i] == 0 and Y[i] == 0:
                    TN = TN + 1
                elif Y_predict[i] - Y[i] == 1:
                    FP = FP + 1
                elif Y_predict[i] - Y[i] == -1:
                    FN = FN + 1
                else:
                    pass
            recall[k] = TP / (TP + FN)
            FAR[k] = FP / (FP + TN)

        ######################save ROC
        dir = './ROCcurve/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        cuv = FAR, recall
        with open(dir + 'roc_curve.pkl', 'wb') as f:
            pickle.dump(cuv, f)

    def read_cfg(self):
        # 创建管理对象
        conf = configparser.ConfigParser()
        # 读ini文件
        conf.read("config.ini", encoding="utf-8")  # python3
        return conf.getfloat("parameter", "learning_rate"),\
               conf.getfloat("parameter", "learning_rate_decay"),\
               conf.getint("parameter", "epoch"), \
               conf.getfloat("parameter", "alpha"), \
               conf.getfloat("parameter", "beta"), \
               conf.getfloat("parameter", "gamma"), \
               conf.getfloat("parameter", "theta")

    def fixprob(self, att):
        att = att + 1e-9
        _sum = np.sum(att, axis=1, keepdims=True)
        att = att / _sum
        att = np.clip(att, 1e-9, 1.0)
        return att

    def tf_fixprob(self, att):
        att = att + 1e-9
        _sum = tf.reduce_sum(att, reduction_indices=1, keep_dims=True)
        att = att / _sum
        att = tf.clip_by_value(att, 1e-9, 1.0)
        return att

    def mu(self, x, th):
        one = tf.constant(1, dtype=tf.float32)
        result = tf.cond(pred=tf.less(x, th),
                         true_fn=lambda: tf.divide(tf.log(tf.add(x, 1)), tf.add(tf.log(tf.add(x, 1)), 1)),
                         false_fn=lambda: one)
        return result

    def kus(self, x):
        mean, variance = tf.nn.moments(x, axes=1)
        sub_2 = tf.expand_dims(mean, 0)
        sub_1 = tf.transpose(x)
        sub_op = tf.subtract(sub_1, sub_2)
        return tf.reduce_mean(tf.divide(tf.reduce_sum(tf.pow(tf.transpose(sub_op), 4), axis=1),
                              tf.multiply(tf.square(variance), tf.cast(tf.shape(x)[1], tf.float32))))

    def r_kurtosis(self, y_outputs, th):
        ty = 1. / tf.abs(self.kus(y_outputs) - 3)
        return self.mu(ty, th)

    def JSD(self, y_outputs, y):
        ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        y1 = y_outputs[:, 0]
        y2 = y[:, 0]
        ta = ta.write(0, tf.multiply(y1, tf.log(tf.divide(y1, tf.add(0.5 * y1, 0.5 * y2)))))
        time_a = tf.constant(2)
        result = tf.while_loop(condition_a, body_a, loop_vars=[time_a, y_outputs, y, ta])
        last_time, _, _, last_out = result
        final_out_ta = last_out.stack()
        ta_sum = tf.reduce_sum(final_out_ta, axis=0)

        tb = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        y1 = y_outputs[:, 0]
        y2 = y[:, 0]
        tb = tb.write(0, tf.multiply(y2, tf.log(tf.divide(y2, tf.add(0.5 * y1, 0.5 * y2)))))
        time_b = tf.constant(2)
        result = tf.while_loop(condition_b, body_b, loop_vars=[time_b, y_outputs, y, tb])
        last_time, _, _, last_out = result
        final_out_tb = last_out.stack()
        tb_sum = tf.reduce_sum(final_out_tb, axis=0)
        jsd = 0.5 * tf.add(ta_sum, tb_sum)
        return tf.reduce_mean(jsd)

    def dis_reg(self, y_outputs, fix_marg):
        y_outputs_sum = tf.reduce_sum(y_outputs, axis=1, keep_dims=True)
        multi = tf.tile(y_outputs_sum, multiples=[1, fix_marg])
        return y_outputs / multi

    def distribution_loss(self, y_outputs, y, th):
        # y_outputs = self.dis_reg(y_outputs, fix_marg)
        jsd = self.JSD(y_outputs, y)
        return self.r_kurtosis(y_outputs, th), jsd

    def cross_distribution_loss(self, y_outputs, y):
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_outputs, labels=y)
        return tf.reduce_mean(entropy)

    def style_loss(self, y_outputs, y):
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_outputs, labels=y)
        return tf.reduce_mean(entropy)

    def correlation_tensor(self, w1, w2):
        mean1 = np.mean(w1)
        mean2 = np.mean(w2)
        std1 = np.std(w1, axis=0)
        std2 = np.std(w2, axis=0)
        return np.mean((w1 - mean1) * (w2 - mean2)) / (std1 * std2)

    def min_max_normalization(self, x):
        minn = np.min(x)
        maxx = np.max(x)
        return (x - minn) / (maxx - minn)

    def print_task_correlation(self, W, t1, t2):
        cor_matrix = np.zeros(shape=(t2, t1))
        for i in range(t2):
            for j in range(t1):
                cor_matrix[i][j] = self.correlation_tensor(W[t1+i], W[j])
        # print("correlation between subtasks=", cor_matrix)
        return cor_matrix

    def get_W(self):
        w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='W')
        w = [w[i] for i in range(0, len(w), 2)]
        w = tf.concat(w, axis=1)
        return w

    def ini_omega(self, task_num):
        with tf.variable_scope('Omega'):
            o = tf.Variable(tf.eye(task_num, dtype=tf.float32) / tf.cast(task_num, tf.float32),
                            dtype=tf.float32, name='omega', trainable=True)

    def tr(self, W, o):
        result = tf.linalg.trace(tf.matmul(tf.matmul(W, tf.matrix_inverse(o)), tf.transpose(W)))
        return result

    def update_omega(self, W):
        A = tf.matmul(tf.transpose(W), W)
        eigval, eigvec = tf.self_adjoint_eig(A)
        eigval = tf.matrix_diag(tf.sqrt(eigval))
        A_sqrt = tf.matmul(tf.matmul(tf.matrix_inverse(eigvec), eigval), eigvec)
        return tf.divide(A_sqrt, tf.linalg.trace(A_sqrt))

    def scalar_for_weights(self, grad, var, omega, taskid):
        if var.name[0] == 'W':
            if var.name == 'W/fully_connected/weights:0' or var.name == 'W/fully_connected/biases:0':
                grad = tf.multiply(grad, omega[taskid][0])
            else:
                s = int(var.name[18])
                grad = tf.multiply(grad, omega[taskid][s])
        return grad

    def train_MTCNN(self, data='dataset/', model_save_path='./model_MTCNN/', val=True, task_marg=10, fix_marg=10):
        folder = os.path.exists(model_save_path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(model_save_path)  # makedirs 创建文件时如果路径不存在会创建这个路径

        dataset = AVAImages()
        if val:
            dataset.read_data(read_dir=data, flag="val")
            dataset.val_set_y[:, 0: fix_marg] = self.fixprob(dataset.val_set_y[:, 0: fix_marg])
            # y_val = dataset.dis2mean(dataset.val_set_y[:, 0: 10])
            # y_val = np.int64(y_val >= 5)  # 前提test_set_y.shape=(num,)
        dataset.read_data(read_dir=data, flag="test")
        y_test = dataset.dis2mean(dataset.test_set_y[:, 0: 10])
        y_test = np.int64(y_test >= 5)  # 前提test_set_y.shape=(num,)
        dataset.read_data(read_dir=data, flag="Th")
        dataset.Th_y[:, 0: fix_marg] = self.fixprob(dataset.Th_y[:, 0: fix_marg])
        dataset.read_batch_cfg()
        learning_rate, learning_rate_decay, epoch, alpha, beta, gamma, theta = self.read_cfg()
        w, h, c = self.input_size
        with tf.name_scope("Inputs"):
            x = tf.placeholder(tf.float32, [None, w, h, c])
            y = tf.placeholder(tf.float32, [None, self.output_size])
            th = tf.placeholder(tf.float32)
            task_id = tf.placeholder(tf.int32)
        y_list = self.MTCNN(x, True)  # y_outputs = (None, 24)
        y_outputs = tf.concat(y_list, axis=1)
        y_outputs_to_one_ori = y_outputs[:, 0: task_marg] / tf.reduce_sum(y_outputs[:, 0: task_marg], keep_dims=True)
        y_outputs_to_one = self.tf_fixprob(y_outputs_to_one_ori)
        y_mv = self.score2style(y_outputs_to_one)
        global_step = tf.Variable(0, trainable=False)
        upgrade_global_step = tf.assign(global_step, tf.add(global_step, 1))

        with tf.name_scope("Loss"):
            cross_val_loss = self.JSD(y_outputs_to_one, y[:, 0: task_marg])
            W = self.get_W()
            self.ini_omega(self.output_size)
            omegaaa = tf.get_default_graph().get_tensor_by_name('Loss/Omega/omega:0')
            tr_W_omega_WT = self.tr(W, omegaaa)
            r_kus = self.r_kurtosis(y_outputs_to_one, th)
            dis_loss = self.JSD(y_outputs_to_one, y[:, 0: task_marg])
            loss = r_kus * (dis_loss +
                            gamma * self.style_loss(y_outputs[:, task_marg:], y[:, task_marg:]) +
                            tf.contrib.layers.apply_regularization(
                               regularizer=tf.contrib.layers.l2_regularizer(alpha, scope=None),
                               weights_list=tf.trainable_variables()) +
                            theta * tr_W_omega_WT + 
                            beta * self.style_loss(y_mv, y[:, 10:])
                            )
        with tf.name_scope("Train"):
            # get variables
            train_theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Theta')
            WW = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='W')

            # lr weight decay
            learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step, decay_steps=10, decay_rate=learning_rate_decay, staircase=False)
           
            # optimize
            opt = tf.train.AdamOptimizer(learning_rate)
            gradient_var_all = opt.compute_gradients(loss, var_list=train_theta+WW)
            capped_gvs = [(self.scalar_for_weights(grad, var, omegaaa, task_id), var)
                          for grad, var in gradient_var_all]
            train_op = opt.apply_gradients(capped_gvs)
            train_op_all = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step,
                                                                          var_list=train_theta+WW)
            train_op_omega = tf.assign(omegaaa, self.update_omega(W))

        saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=2)
        cross_val_loss_transfer = 0
        train_theta_and_W_first = 20
        best_val_loss = 1000
        improvement_threshold = 0.999
        last_cor_dis = 0.0
        best_test_acc = 0.0
        best_test_acc_epoch = 0
        best_test_acc_batch = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                while True:
                    # 遍历所有batch
                    x_b, y_b, end = dataset.load_next_batch_quicker(read_dir=data)
                    y_b[:, 0: fix_marg] = self.fixprob(y_b[:, 0: fix_marg])
                    step = sess.run(global_step)
                    if step < train_theta_and_W_first:
                        cross_val_loss_transfer = sess.run(cross_val_loss, feed_dict={x: dataset.Th_x, y: dataset.Th_y})
                        train_op_, loss_ = sess.run([train_op_all, loss],
                                                    feed_dict={x: x_b, y: y_b, th: cross_val_loss_transfer})
                    elif np.random.rand() < 0.5:
                        train_op_ = sess.run(train_op_omega)
                        sess.run(upgrade_global_step)
                    else:
                        cross_val_loss_transfer = sess.run(cross_val_loss, feed_dict={x: dataset.Th_x, y: dataset.Th_y})
                        for taskid in range(self.output_size):
                            train_op_, loss_ = sess.run([train_op, loss],
                                                        feed_dict={x: x_b, y: y_b,
                                                                   th: cross_val_loss_transfer, task_id: taskid})
                        sess.run(upgrade_global_step)

                    if val:
                        # y_outputs_ = sess.run(y_outputs, feed_dict={x: dataset.val_set_x})
                        # y_outputs_ = dataset.dis2mean(y_outputs_[:, 0: 10])
                        # y_pred = np.int64(y_outputs_ >= 5)
                        # val_acc = sum((y_pred-y_val)==0) / dataset.val_set_x.shape[0]
                        val_loss = sess.run(loss, feed_dict={x: dataset.val_set_x, y: dataset.val_set_y,
                                                             th: cross_val_loss_transfer})
                        print("epoch {3} batch {4}/{0} loss {1}, validation loss {2}".
                              format(dataset.batch_index_max, loss_, val_loss, i+1, dataset.batch_index))
                        
                        if val_loss < best_val_loss * improvement_threshold:
                            # if improvement_threshold < 1:
                            #     improvement_threshold += 0.001
                            best_val_loss = val_loss
                            ### test acc
                            y_outputs_to_zero_one = y_outputs[:, 0: task_marg] / \
                                                    tf.reduce_sum(y_outputs[:, 0: task_marg], keep_dims=True)
                            y_outputs_ = sess.run(y_outputs_to_zero_one, feed_dict={x: dataset.test_set_x})

                            y_outputs_ = dataset.dis2mean(y_outputs_[:, 0: 10])
                            y_pred = np.int64(y_outputs_ >= 5)
                            test_acc = sum((y_pred-y_test) == 0) / dataset.test_set_x.shape[0]
                            print("    test acc {acc} with best acc {best} in epoch{e}/batch{b}".format(acc=test_acc, best=best_test_acc, e=best_test_acc_epoch, b=best_test_acc_batch))
                            if test_acc > best_test_acc:
                                best_test_acc = test_acc
                                best_test_acc_epoch = i
                                best_test_acc_batch = dataset.batch_index

                            ### correlation matrix
                            Wa_and_Ws = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='W'))
                            W = np.zeros(shape=(self.output_size, 4096))
                            for ii in range(W.shape[0]):
                                W[ii] = np.array(np.squeeze(Wa_and_Ws[ii*2]))
                            cor_matrix1 = self.print_task_correlation(W, task_marg, self.output_size-task_marg)
                            cor_matrix1 = self.min_max_normalization(cor_matrix1)
                            cor_matrix2 = sess.run(tf.transpose(
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Cor_Matrix')[0])
                            )
                            cor_matrix2 = self.min_max_normalization(cor_matrix2)
                            cor_dis = np.sum(np.square(cor_matrix1-cor_matrix2))
                            print("    distance {0}, add distance {1}.".format(cor_dis, cor_dis - last_cor_dis))
                            last_cor_dis = cor_dis
                    else:
                        print("training step {0}, loss {1}".format(step, loss_))

                    if end == 1:
                        break

                # ### save
            cv2.imwrite(model_save_path + "cor_matrix1.png",
                        cv2.resize(cor_matrix1 * 255, (300, 420), interpolation=cv2.INTER_CUBIC))
            cv2.imwrite(model_save_path + "cor_matrix2.png",
                        cv2.resize(cor_matrix2 * 255, (300, 420), interpolation=cv2.INTER_CUBIC))
            saver.save(sess, model_save_path + 'my_model')
            os.system('zip -r myfile.zip ./' + model_save_path)
            # sava_train_model(model_file="myfile.zip", dir_name="./file", overwrite=True)
            # upload_data("myfile.zip", overwrite=True)

    def train_MTCNN_continue(self, data='dateset/', model_read_path='./model_MTCNN/', model_save_path='./model_MTCNN_continue/', op_freq=10, val=True, task_marg=10, fix_marg=10):
        folder = os.path.exists(model_save_path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(model_save_path)  # makedirs 创建文件时如果路径不存在会创建这个路径

        dataset = AVAImages()
        if val:
            dataset.read_data(read_dir=data, flag="val")
            dataset.val_set_y[:, 0: fix_marg] = self.fixprob(dataset.val_set_y[:, 0: fix_marg])
            # y_val = dataset.dis2mean(dataset.val_set_y[:, 0: 10])
            # y_val = np.int64(y_val >= 5)  # 前提test_set_y.shape=(num,)
        dataset.read_data(read_dir=data, flag="test")
        y_test = dataset.dis2mean(dataset.test_set_y[:, 0: 10])
        y_test = np.int64(y_test >= 5)  # 前提test_set_y.shape=(num,)
        dataset.read_data(read_dir=data, flag="Th")
        dataset.Th_y[:, 0: fix_marg] = self.fixprob(dataset.Th_y[:, 0: fix_marg])
        dataset.read_batch_cfg()
        learning_rate, learning_rate_decay, epoch, alpha, beta, gamma, theta = self.read_cfg()
        w, h, c = self.input_size
        with tf.name_scope("Inputs"):
            x = tf.placeholder(tf.float32, [None, w, h, c])
            y = tf.placeholder(tf.float32, [None, self.output_size])
            th = tf.placeholder(tf.float32)
            task_id = tf.placeholder(tf.int32)
        y_list = self.MTCNN(x, True)  # y_outputs = (None, 24)
        y_outputs = tf.concat(y_list, axis=1)
        y_mv = self.score2style(y_outputs[:, 0: 10])
        global_step = tf.Variable(0, trainable=False)
        upgrade_global_step = tf.assign(global_step, tf.add(global_step, 1))

        with tf.name_scope("Loss"):
            cross_val_loss = self.JSD(y_outputs[:, 0: task_marg], y[:, 0: task_marg])
            W = self.get_W()
            self.ini_omega(self.output_size)
            omegaaa = tf.get_default_graph().get_tensor_by_name('Loss/Omega/omega:0')
            tr_W_omega_WT = self.tr(W, omegaaa)
            loss = self.distribution_loss(y_outputs[:, 0: task_marg], y[:, 0: task_marg], th, fix_marg) + \
                   gamma * self.style_loss(y_outputs[:, task_marg:], y[:, task_marg:]) + \
                   tf.contrib.layers.apply_regularization(
                       regularizer=tf.contrib.layers.l2_regularizer(alpha, scope=None),
                       weights_list=tf.trainable_variables()) + \
                   theta * tr_W_omega_WT + beta * self.style_loss(y_mv, y[:, 10:])
        with tf.name_scope("Train"):
            # get variables
            train_theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Theta')
            WW = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='W')

            opt = tf.train.AdamOptimizer(learning_rate)
            gradient_var_all = opt.compute_gradients(loss, var_list=train_theta + WW)
            capped_gvs = [(self.scalar_for_weights(grad, var, omegaaa, task_id), var)
                          for grad, var in gradient_var_all]
            train_op = opt.apply_gradients(capped_gvs)
            train_op_all = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step,
                                                                          var_list=train_theta + WW)
            train_op_omega = tf.assign(omegaaa, self.update_omega(W))

        saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=2)
        re_saver = tf.train.Saver()
        train_theta_and_W_first = 20
        best_val_loss = 1000
        improvement_threshold = 0.98
        last_cor_dis = 0.0
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_read_path)
            if ckpt and ckpt.model_checkpoint_path:
                re_saver.restore(sess, ckpt.model_checkpoint_path)
            for i in range(epoch):
                while True:
                    # 遍历所有batch
                    x_b, y_b, end = dataset.load_next_batch_quicker(read_dir=data)
                    y_b[:, 0: fix_marg] = self.fixprob(y_b[:, 0: fix_marg])
                    step = sess.run(global_step)

                    if step < train_theta_and_W_first:
                        cross_val_loss_transfer = sess.run(cross_val_loss, feed_dict={x: dataset.Th_x, y: dataset.Th_y})
                        train_op_, loss_ = sess.run([train_op_all, loss],
                                                    feed_dict={x: x_b, y: y_b, th: cross_val_loss_transfer})
                    elif np.random.rand() < 0.5:
                        train_op_ = sess.run(train_op_omega)
                        sess.run(upgrade_global_step)
                    else:
                        cross_val_loss_transfer = sess.run(cross_val_loss, feed_dict={x: dataset.Th_x, y: dataset.Th_y})
                        for taskid in range(self.output_size):
                            train_op_, loss_ = sess.run([train_op, loss],
                                                        feed_dict={x: x_b, y: y_b,
                                                                   th: cross_val_loss_transfer, task_id: taskid})
                        sess.run(upgrade_global_step)

                        if val:
                            val_loss = sess.run(loss, feed_dict={x: dataset.val_set_x, y: dataset.val_set_y,
                                                                 th: cross_val_loss_transfer})
                            print("epoch {3} batch {4}/{0} loss {1}, validation loss {2}".
                                  format(dataset.batch_index_max, loss_, val_loss, i + 1, dataset.batch_index))

                            if val_loss < best_val_loss * improvement_threshold:
                                if improvement_threshold < 1:
                                    improvement_threshold *= 1.001
                                best_val_loss = val_loss
                                ### test acc
                                y_outputs_ = sess.run(y_outputs, feed_dict={x: dataset.test_set_x})
                                y_outputs_ = dataset.dis2mean(y_outputs_[:, 0: 10])
                                y_pred = np.int64(y_outputs_ >= 5)
                                test_acc = sum((y_pred - y_test) == 0) / dataset.test_set_x.shape[0]
                                print("    test acc {0}.".format(test_acc))

                                ### correlation matrix
                                Wa_and_Ws = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='W'))
                                W = np.zeros(shape=(self.output_size, 4096))
                                for ii in range(W.shape[0]):
                                    W[ii] = np.array(np.squeeze(Wa_and_Ws[ii * 2]))
                                cor_matrix1 = self.print_task_correlation(W, task_marg, self.output_size - task_marg)
                                cor_matrix1 = self.min_max_normalization(cor_matrix1)
                                cor_matrix2 = sess.run(tf.transpose(
                                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Cor_Matrix')[0])
                                )
                                cor_matrix2 = self.min_max_normalization(cor_matrix2)
                                cor_dis = np.sum(np.square(cor_matrix1 - cor_matrix2))
                                print("    distance {0}, add distance {1}.".format(cor_dis, cor_dis - last_cor_dis))
                                last_cor_dis = cor_dis

                        else:
                            print("training step {0}, loss {1}".format(step, loss_))

                    if end == 1:
                        break

                ### save
                cv2.imwrite(model_save_path + "cor_matrix1.png",
                            cv2.resize(cor_matrix1 * 255, (300, 420), interpolation=cv2.INTER_CUBIC))
                cv2.imwrite(model_save_path + "cor_matrix2.png",
                            cv2.resize(cor_matrix2 * 255, (300, 420), interpolation=cv2.INTER_CUBIC))
                saver.save(sess, model_save_path + 'my_model')
                os.system('zip -r myfile.zip ./' + model_save_path)
                # sava_train_model(model_file="myfile.zip", dir_name="./file", overwrite=True)
                # upload_data("myfile.zip", overwrite=True)

    def train_score_CNN(self, data='dataset/', model_save_path='./model_score_CNN/', val=True, task_marg=10, fix_marg=10):
        """
        训练单纯的分数分布预测模型
        :param data:
        :param model_save_path:
        :param val:
        :param task_marg:
        :param fix_marg:
        :return:
        """
        folder = os.path.exists(model_save_path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(model_save_path)  # makedirs 创建文件时如果路径不存在会创建这个路径

        dataset = AVAImages()
        if val:
            dataset.read_data(read_dir=data, flag="val")
            dataset.val_set_y[:, 0: fix_marg] = self.fixprob(dataset.val_set_y[:, 0: fix_marg])
        dataset.read_data(read_dir=data, flag="test")
        y_test = dataset.dis2mean(dataset.test_set_y[:, 0: 10])
        y_test = np.int64(y_test >= 5)  # 前提test_set_y.shape=(num,)
        dataset.read_data(read_dir=data, flag="Th")
        dataset.Th_y[:, 0: fix_marg] = self.fixprob(dataset.Th_y[:, 0: fix_marg])
        dataset.read_batch_cfg()
        learning_rate, learning_rate_decay, epoch, alpha, beta, gamma, theta = self.read_cfg()
        w, h, c = self.input_size
        with tf.name_scope("Inputs"):
            x = tf.placeholder(tf.float32, [None, w, h, c])
            y = tf.placeholder(tf.float32, [None, self.output_size])
            th = tf.placeholder(tf.float32)
        y_list = self.MTCNN(x, True)  # y_outputs = (None, 24)
        y_outputs = tf.concat(y_list, axis=1)
        y_outputs_to_one_ori = y_outputs[:, 0: task_marg] / tf.reduce_sum(y_outputs[:, 0: task_marg],
                                                                          keep_dims=True)
        y_outputs_to_one = self.tf_fixprob(y_outputs_to_one_ori)
        global_step = tf.Variable(0, trainable=False)
        upgrade_global_step = tf.assign(global_step, tf.add(global_step, 1))

        with tf.name_scope("Loss"):
            cross_val_loss = self.JSD(y_outputs_to_one, y[:, 0: task_marg])
            r_kus, dis_loss = self.distribution_loss(y_outputs_to_one, y[:, 0: task_marg], th, fix_marg)
            loss = r_kus * dis_loss
        with tf.name_scope("Train"):
            # get variables
            train_theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Theta')
            WW = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='W')

            # lr weight decay
            learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                                                       decay_steps=10, decay_rate=learning_rate_decay, staircase=False)

            # optimize
            train_op_all = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step,
                                                                          var_list=train_theta + WW)

        saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=2)
        best_val_loss = 1000
        improvement_threshold = 0.999
        best_test_acc = 0.0
        best_test_acc_epoch = 0
        best_test_acc_batch = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                while True:
                    # 遍历所有batch
                    x_b, y_b, end = dataset.load_next_batch_quicker(read_dir=data)
                    y_b[:, 0: fix_marg] = self.fixprob(y_b[:, 0: fix_marg])
                    step = sess.run(global_step)
                    cross_val_loss_transfer = sess.run(cross_val_loss, feed_dict={x: dataset.Th_x, y: dataset.Th_y})
                    train_op_, loss_ = sess.run([train_op_all, loss],
                                                feed_dict={x: x_b, y: y_b, th: cross_val_loss_transfer})

                    if val:
                        val_loss = sess.run(loss, feed_dict={x: dataset.val_set_x, y: dataset.val_set_y,
                                                             th: cross_val_loss_transfer})
                        print("epoch {3} batch {4}/{0} loss {1}, validation loss {2}".
                              format(dataset.batch_index_max, loss_, val_loss, i + 1, dataset.batch_index))

                        if val_loss < best_val_loss * improvement_threshold:
                            best_val_loss = val_loss
                            ### test acc
                            y_outputs_to_zero_one = y_outputs[:, 0: task_marg] / tf.reduce_sum(y_outputs[:, 0: task_marg], keep_dims=True)
                            y_outputs_ = sess.run(y_outputs_to_zero_one, feed_dict={x: dataset.test_set_x})
                            y_outputs_ = dataset.dis2mean(y_outputs_[:, 0: 10])
                            y_pred = np.int64(y_outputs_ >= 5)
                            test_acc = sum((y_pred - y_test) == 0) / dataset.test_set_x.shape[0]
                            print("    test acc {acc} with best acc {best} in epoch{e}/batch{b}".
                                  format(acc=test_acc, best=best_test_acc, e=best_test_acc_epoch, b=best_test_acc_batch))
                            if test_acc > best_test_acc:
                                best_test_acc = test_acc
                                best_test_acc_epoch = i
                                best_test_acc_batch = dataset.batch_index
                    else:
                        print("training step {0}, loss {1}".format(step, loss_))

                    if end == 1:
                        break

            #### save
            saver.save(sess, model_save_path + 'my_model')
            os.system('zip -r myfile.zip ./' + model_save_path)
            sava_train_model(model_file="myfile.zip", dir_name="./file", overwrite=True)
            upload_data("myfile.zip", overwrite=True)

    def train_MTCNN_v2(self, data='dataset/', model_save_path='./model_MTCNN_v2/', val=True, task_marg=10, fix_marg=10):
        folder = os.path.exists(model_save_path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(model_save_path)  # makedirs 创建文件时如果路径不存在会创建这个路径

        # load data
        dataset = AVAImages()
        if val:
            dataset.read_data(read_dir=data, flag="val")
            dataset.val_set_y[:, 0: fix_marg] = self.fixprob(dataset.val_set_y[:, 0: fix_marg])
        dataset.read_data(read_dir=data, flag="test")
        y_test = dataset.dis2mean(dataset.test_set_y[:, 0: 10])
        y_test = np.int64(y_test >= 5)  # 前提test_set_y.shape=(num,)
        dataset.read_data(read_dir=data, flag="Th")
        dataset.Th_y[:, 0: fix_marg] = self.fixprob(dataset.Th_y[:, 0: fix_marg])

        # load parameters
        dataset.read_batch_cfg()
        learning_rate, learning_rate_decay, epoch, alpha, beta, gamma, theta = self.read_cfg()

        # placeholders
        w, h, c = self.input_size
        with tf.name_scope("Inputs"):
            x = tf.placeholder(tf.float32, [None, w, h, c])
            y = tf.placeholder(tf.float32, [None, self.output_size])
            th = tf.placeholder(tf.float32)
            task_id = tf.placeholder(tf.int32)
        y_list = self.MTCNN_v2(x, True)  # y_outputs = (None, 24)
        y_outputs = tf.concat(y_list, axis=1)
        y_outputs_to_one_ori = y_outputs[:, 0: task_marg] / tf.reduce_sum(y_outputs[:, 0: task_marg], keep_dims=True)
        y_outputs_to_one = self.tf_fixprob(y_outputs_to_one_ori)

        # other parameters
        global_step = tf.Variable(0, trainable=False)
        upgrade_global_step = tf.assign(global_step, tf.add(global_step, 1))

        with tf.name_scope("Loss"):
            cross_val_loss = self.JSD(y_outputs_to_one, y[:, 0: task_marg])
            W = self.get_W()
            self.ini_omega(self.output_size)
            omegaaa = tf.get_default_graph().get_tensor_by_name('Loss/Omega/omega:0')
            tr_W_omega_WT = self.tr(W, omegaaa)
            r_kus = self.r_kurtosis(y_outputs_to_one, th)
            dis_loss = self.JSD(y_outputs_to_one, y[:, 0: task_marg])
            loss = r_kus * (dis_loss +
                            gamma * self.style_loss(y_outputs[:, task_marg:], y[:, task_marg:]) +
                            tf.contrib.layers.apply_regularization(
                                regularizer=tf.contrib.layers.l2_regularizer(alpha, scope=None),
                                weights_list=tf.trainable_variables()) +
                            theta * tr_W_omega_WT
                            )
        with tf.name_scope("Train"):
            # get variables
            train_theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Theta')
            WW = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='W')

            # lr weight decay
            learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                                                       decay_steps=10, decay_rate=learning_rate_decay, staircase=False)

            # optimize
            opt = tf.train.AdamOptimizer(learning_rate)
            gradient_var_all = opt.compute_gradients(loss, var_list=train_theta + WW)
            capped_gvs = [(self.scalar_for_weights(grad, var, omegaaa, task_id), var)
                          for grad, var in gradient_var_all]
            train_op = opt.apply_gradients(capped_gvs)
            train_op_all = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, var_list=train_theta + WW)
            train_op_omega = tf.assign(omegaaa, self.update_omega(W))

        saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=2)
        cross_val_loss_transfer = 0
        train_theta_and_W_first = 20
        best_val_loss = 1000
        improvement_threshold = 0.999
        best_test_acc = 0.0
        best_test_acc_epoch = 0
        best_test_acc_batch = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                while True:
                    # 遍历所有batch
                    x_b, y_b, end = dataset.load_next_batch_quicker(read_dir=data)
                    y_b[:, 0: fix_marg] = self.fixprob(y_b[:, 0: fix_marg])
                    step = sess.run(global_step)
                    if step < train_theta_and_W_first:
                        cross_val_loss_transfer = sess.run(cross_val_loss, feed_dict={x: dataset.Th_x, y: dataset.Th_y})
                        train_op_, loss_ = sess.run([train_op_all, loss], feed_dict={x: x_b, y: y_b, th: cross_val_loss_transfer})
                    elif np.random.rand() < 0.5:
                        train_op_ = sess.run(train_op_omega)
                        sess.run(upgrade_global_step)
                    else:
                        cross_val_loss_transfer = sess.run(cross_val_loss, feed_dict={x: dataset.Th_x, y: dataset.Th_y})
                        for taskid in range(self.output_size):
                            train_op_, loss_ = sess.run([train_op, loss], feed_dict={x: x_b, y: y_b, th: cross_val_loss_transfer, task_id: taskid})
                        sess.run(upgrade_global_step)

                    if val:
                        val_loss = sess.run(loss, feed_dict={x: dataset.val_set_x, y: dataset.val_set_y,
                                                             th: cross_val_loss_transfer})
                        print("epoch {3} batch {4}/{0} loss {1}, validation loss {2}".
                              format(dataset.batch_index_max, loss_, val_loss, i + 1, dataset.batch_index))

                        if val_loss < best_val_loss * improvement_threshold:
                            best_val_loss = val_loss
                            # test acc
                            y_outputs_ = y_outputs[:, 0: task_marg] / tf.reduce_sum(y_outputs[:, 0: task_marg], keep_dims=True)
                            y_outputs_to_one_test = sess.run(y_outputs_, feed_dict={x: dataset.test_set_x})
                            y_outputs_mean = dataset.dis2mean(y_outputs_to_one_test[:, 0: 10])
                            y_pred = np.int64(y_outputs_mean >= 5)
                            test_acc = sum((y_pred - y_test) == 0) / dataset.test_set_x.shape[0]
                            print("    test acc {acc} with best acc {best} in epoch{e}/batch{b}".format(acc=test_acc,
                                                                                                        best=best_test_acc,
                                                                                                        e=best_test_acc_epoch,
                                                                                                        b=best_test_acc_batch))
                            if test_acc > best_test_acc:
                                best_test_acc = test_acc
                                best_test_acc_epoch = i
                                best_test_acc_batch = dataset.batch_index

                            # correlation matrix
                            # cor1
                            Wa_and_Ws = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='W'))
                            W = np.zeros(shape=(self.output_size, 4096))
                            for ii in range(W.shape[0]):
                                W[ii] = np.array(np.squeeze(Wa_and_Ws[ii * 2]))
                            cor_matrix1 = self.print_task_correlation(W, task_marg, self.output_size - task_marg)
                            cor_matrix1 = self.min_max_normalization(cor_matrix1)
                    else:
                        print("training step {0}, loss {1}".format(step, loss_))

                    if end == 1:
                        break

            # ### save
            cv2.imwrite(model_save_path + "cor_matrix1.png",
                        cv2.resize(cor_matrix1 * 255, (300, 420), interpolation=cv2.INTER_CUBIC))
            saver.save(sess, model_save_path + 'my_model')
            os.system('zip -r myfile.zip ./' + model_save_path)
            sava_train_model(model_file="myfile.zip", dir_name="./file", overwrite=True)
            upload_data("myfile.zip", overwrite=True)

    def train_cor_matrix_label(self, data='dataset/', model_save_path='./model_cor_matrix2/', val=True):
        folder = os.path.exists(model_save_path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(model_save_path)  # makedirs 创建文件时如果路径不存在会创建这个路径

        # load data
        dataset = AVAImages()
        if val:
            dataset.read_data(read_dir=data, flag="val")
            dataset.val_set_y[:, 10:] = self.fixprob(dataset.val_set_y[:, 10:])

        # load parameters
        dataset.read_batch_cfg()
        learning_rate, learning_rate_decay, epoch, alpha, beta, gamma, theta = self.read_cfg()

        # placeholders
        with tf.name_scope("Inputs"):
            x = tf.placeholder(tf.float32, [None, 10])
            y = tf.placeholder(tf.float32, [None, 14])

        # cor_fc_layer
        y_outputs = self.score2style(x)
        y_outputs = self.tf_fixprob(y_outputs)

        # other parameters
        global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("Loss"):
            loss_c = self.JSD(y_outputs, y)
        with tf.name_scope("Train"):
            # get variables
            Wc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Cor_Matrix')

            # lr weight decay
            learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                                                       decay_steps=10, decay_rate=learning_rate_decay, staircase=False)

            # optimize
            train_op_wc = tf.train.AdamOptimizer(learning_rate).minimize(loss_c, global_step=global_step, var_list=Wc)

        saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=2)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # train wc
            print("training of Wc ...")
            best_val_loss = 1000
            improvement_threshold = 0.999
            patience = 4
            i = 0
            while i <= patience:
                while True:
                    _, y_b, end = dataset.load_next_batch_quicker(read_dir=data)
                    y_b[:, 10:] = self.fixprob(y_b[:, 10:])
                    sess.run(global_step)
                    if end == 1:
                        break

                    train_op_, loss_ = sess.run([train_op_wc, loss_c], feed_dict={x: y_b[:, 0: 10], y: y_b[:, 10:]})
                    if val:
                        val_loss = sess.run(loss_c, feed_dict={x: dataset.val_set_y[:, 0: 10], y: dataset.val_set_y[:, 10:]})
                        print("epoch {3} batch {4}/{0} loss {1}, validation loss {2}".
                              format(dataset.batch_index_max, loss_, val_loss, i + 1, dataset.batch_index))

                        if val_loss < best_val_loss * improvement_threshold:
                            patience *= 2
                            best_val_loss = val_loss
                i += 1

            # cor2
            cor_matrix2 = sess.run(tf.transpose(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Cor_Matrix')[0])
            )
            cor_matrix2 = self.min_max_normalization(cor_matrix2)

            # ### save
            cv2.imwrite(model_save_path + "cor_matrix2.png",
                        cv2.resize(cor_matrix2 * 255, (300, 420), interpolation=cv2.INTER_CUBIC))
            saver.save(sess, model_save_path + 'my_model')

    def get_uninitialized_variables(self, sess):
        global_vars = tf.global_variables()
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        print([str(i.name) for i in not_initialized_vars])
        return not_initialized_vars

    def train_cor_matrix_predict(self, data='dataset/', model_read_path='./model_MTCNN_v2', model_save_path='./model_cor_matrix2/'):
        folder = os.path.exists(model_save_path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(model_save_path)  # makedirs 创建文件时如果路径不存在会创建这个路径

        # load data
        dataset = AVAImages()

        # load parameters
        dataset.read_batch_cfg()
        learning_rate, learning_rate_decay, epoch, alpha, beta, gamma, theta = self.read_cfg()

        # placeholders
        w, h, c = self.input_size
        with tf.name_scope("Inputs"):
            x_train = tf.placeholder(tf.float32, [None, 10])
            y_train = tf.placeholder(tf.float32, [None, 14])
            x = tf.placeholder(tf.float32, [None, w, h, c])
        y_list = self.MTCNN_v2(x, True)  # y_outputs = (None, 24)
        y_outputs_mtcnn = tf.concat(y_list, axis=1)
        # y_outputs_to_one_ori = y_outputs[:, 0: 10] / tf.reduce_sum(y_outputs[:, 0: 10], keep_dims=True)
        # y_outputs_to_one = self.tf_fixprob(y_outputs_to_one_ori)

        # cor_fc_layer
        y_outputs = self.score2style(x_train)
        y_outputs = self.tf_fixprob(y_outputs)

        # other parameters
        global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("Loss"):
            loss_c = self.JSD(y_outputs, y_train)
        with tf.name_scope("Train"):
            # get variables
            Wc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Cor_Matrix')

            # lr weight decay
            learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
                                                       decay_steps=10, decay_rate=learning_rate_decay, staircase=False)

            # optimize
            train_op_wc = tf.train.AdamOptimizer(learning_rate).minimize(loss_c, global_step=global_step, var_list=Wc)

        variables_to_restore = slim.get_variables_to_restore(include=['Theta', 'W'])  # 单引号指只恢复一个层。双引号会恢复含该内容的所有层。
        re_saver = tf.train.Saver(variables_to_restore)  # 如果这里不指定特定的参数，sess会把目前graph中所有都恢复
        tf_vars = tf.trainable_variables(scope="Cor_Matrix")
        saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=2, var_list=tf_vars)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_read_path)
            if ckpt and ckpt.model_checkpoint_path:
                re_saver.restore(sess, ckpt.model_checkpoint_path)
            un_init = tf.variables_initializer(self.get_uninitialized_variables(sess))  # 获取没有初始化(通过已有model加载)的变量
            sess.run(un_init)  # 对没有初始化的变量进行初始化并训练.

            # train wc
            print("training of Wc ...")
            patience = 4
            i = 0
            while i <= patience:
                while True:
                    x_b, y_b, end = dataset.load_next_batch_quicker(read_dir=data)
                    sess.run(global_step)
                    if end == 1:
                        break

                    y_predict = sess.run(y_outputs_mtcnn, feed_dict={x: x_b})
                    y_predict[:, 10:] = self.fixprob(y_predict[:, 10:])
                    train_op_, loss_ = sess.run([train_op_wc, loss_c], feed_dict={x_train: y_predict[:, 0: 10],
                                                                                  y_train: y_predict[:, 10:]})

                    print("epoch {e} batch {b_index}/{b} loss {loss}".
                          format(e=i+1, b_index=dataset.batch_index, b=dataset.batch_index_max, loss=loss_))
                i += 1

            # cor2
            cor_matrix2 = sess.run(tf.transpose(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Cor_Matrix')[0])
            )
            cor_matrix2 = self.min_max_normalization(cor_matrix2)

            # ### save
            cv2.imwrite(model_save_path + "cor_matrix2.png",
                        cv2.resize(cor_matrix2 * 255, (300, 420), interpolation=cv2.INTER_CUBIC))
            saver.save(sess, model_save_path + 'my_model')
            os.system('zip -r myfile.zip ./' + model_save_path)
            sava_train_model(model_file="myfile.zip", dir_name="./file", overwrite=True)
            upload_data("myfile.zip", overwrite=True)
