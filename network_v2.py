from resnet import resnet_v2_4x4
from resnet import resnet_v2_4x4_shallow
from data import AVAImages
import configparser
import os
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim


class Network(object):
    def __init__(self, input_size: tuple,
                 output_size: int,
                 net: str):
        """
        init the network

        :param input_size: w, h, c
        :param output_size: size
        :param net: "predict_tags" or "comparator"
        :param model_save_path: model_save_path = './model/'
        """
        self.net = net
        self.input_size = input_size
        self.output_size = output_size

    def score2style(self, inputs, is_training):
        output = slim.stack(inputs, slim.fully_connected, [32, 64, 14], scope='fc', is_training=is_training)
        return output

    def multi_task(self, inputs):
        _, softmax, _ = resnet_v2_4x4(inputs, num_classes=self.output_size)  # num_classess=2+14
        return softmax

    def propagate(self, inputs):
        pass

    def validation_acc(self, sess, y_outputs, ph, real_data):
        pass

    def read_cfg(self):
        # 创建管理对象
        conf = configparser.ConfigParser()
        # 读ini文件
        conf.read("cfg.ini", encoding="utf-8")  # python3
        return conf.getfloat("parameter", "learning_rate"),\
               conf.getfloat("parameter", "learning_rate_decay"),\
               conf.getint("parameter", "epoch"), \
               conf.getfloat("parameter", "a"), \
               conf.getfloat("parameter", "b"), \
               conf.getfloat("parameter", "c"), \
               conf.getfloat("parameter", "d")

    def train_score2style(self, model_save_path='./model_score2style/', op_freq=10, val=True):
        dataset = AVAImages()
        if val:
            dataset.read_data(read_dir='AVA_score_style/', flag=0)
        dataset.read_batch_cfg()
        learning_rate, learning_rate_decay, epoch, a, b, c, d = self.read_cfg()
        w, h, c = self.input_size
        with tf.name_scope("Inputs"):
            x = tf.placeholder(tf.float32, [None, w, h, c])
            y = tf.placeholder(tf.float32, [None, self.output_size])
            ph = x, y
        y_outputs = self.score2style(x, is_training=True)  # y_outputs = (None, 2)
        global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("Loss"):
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=y_outputs, labels=tf.argmax(y, 1))
            loss = tf.reduce_mean(entropy)
        with tf.name_scope("Train"):
            rate = tf.train.exponential_decay(learning_rate, global_step, 200, learning_rate_decay)  # 指数衰减学习率
            train_op = tf.train.AdamOptimizer(rate).minimize(loss, global_step=global_step)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                while True:
                    # 遍历所有batch
                    x_b, y_b, end = dataset.load_next_batch_quicker()
                    train_op_, loss_, step = sess.run([train_op, loss, global_step], feed_dict={x: x_b, y: y_b})
                    if step % op_freq == 0:
                        if val:
                            print("training step {0}, loss {1}, validation acc {2}"
                                  .format(step, loss_, self.validation_acc(sess, y_outputs, ph, dataset)))
                        else:
                            print("training step {0}, loss {1}".format(step, loss_))
                        saver.save(sess, model_save_path + 'my_model', global_step=global_step)
                    if end == 1:
                        break
            writer = tf.summary.FileWriter("logs/", tf.get_default_graph())
            writer.close()

    def train_multi_task(self, score2style_model='./model_score2style/', model_save_path='./model_ulti/', op_freq=10, val=True):
        dataset = AVAImages()
        if val:
            dataset.read_data(read_dir='AVA_score_style/', flag=0)
        dataset.read_batch_cfg()
        learning_rate, learning_rate_decay, epoch, a, b, c, d = self.read_cfg()
        w, h, c = self.input_size
        with tf.name_scope("Inputs"):
            x = tf.placeholder(tf.float32, [None, w, h, c])
            y = tf.placeholder(tf.float32, [None, self.output_size])
            x_mv = tf.placeholder(tf.float32, [None, 2])
            y_mv = tf.placeholder(tf.float32, [None, self.output_size])
            ph = x, y
        y_outputs = self.multi_task(x)  # y_outputs = (None, 2)
        y_mv = self.score2style(x_mv, is_training=False)
        global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("Loss"):
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=y_outputs[2: 15], labels=tf.argmax(y[2: 15], 1))
            style_loss = tf.reduce_mean(entropy)
            mse0 = tf.losses.mean_squared_error(y[0], y_outputs[0])
            mse1 = tf.losses.mean_squared_error(y[1], y_outputs[1])
            loss = a * style_loss + b * mse0 + c * mse1 + d * y_mv
        with tf.name_scope("Train"):
            rate = tf.train.exponential_decay(learning_rate, global_step, 200, learning_rate_decay)  # 指数衰减学习率
            train_op = tf.train.AdamOptimizer(rate).minimize(loss, global_step=global_step)

        saver = tf.train.Saver()
        re_saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            re_saver.restore(sess=sess, save_path=score2style_model + 'my_model-1')
            for i in range(epoch):
                while True:
                    # 遍历所有batch
                    x_b, y_b, end = dataset.load_next_batch_quicker()
                    train_op_, loss_, step = sess.run([train_op, loss, global_step], feed_dict={x: x_b, y: y_b})
                    if step % op_freq == 0:
                        if val:
                            print("training step {0}, loss {1}, validation acc {2}"
                                  .format(step, loss_, self.validation_acc(sess, y_outputs, ph, dataset)))
                        else:
                            print("training step {0}, loss {1}".format(step, loss_))
                        saver.save(sess, model_save_path + 'my_model', global_step=global_step)
                    if end == 1:
                        break
            writer = tf.summary.FileWriter("logs/", tf.get_default_graph())
            writer.close()
