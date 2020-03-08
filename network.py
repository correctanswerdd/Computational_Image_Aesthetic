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

    def score_net(self, inputs):
        _, softmax, _ = resnet_v2_4x4(inputs, num_classes=self.output_size)
        return softmax

    def feature_extractor(self, input1, input2):
        concat1 = resnet_v2_4x4_shallow(input1)
        concat2 = resnet_v2_4x4_shallow(input2, reuse=True)
        fc1_concat = tf.concat([concat1, concat2], 3, name='fc1_concat')
        bn_training = True
        with tf.variable_scope('fc1_concat_bn'):
            fc1_concat = tf.layers.batch_normalization(fc1_concat, center=False, scale=False, training=bn_training)
        fc1_concat = tf.nn.relu(fc1_concat)
        return fc1_concat

    def propagate(self, inputs):
        if self.net == "predict_bi_class":
            x = inputs
            return self.score_net(x)
        elif self.net == "comparator":
            x1, x2 = inputs
            fc1_concat = self.feature_extractor(x1, x2)
            bn_training = True
            fc3 = slim.conv2d(fc1_concat, 1024, [1, 1], scope='fc3', activation_fn=None)
            with tf.variable_scope('fc3_bn'):
                fc3 = tf.layers.batch_normalization(fc3, center=False, scale=False, training=bn_training)
            fc3 = tf.nn.relu(fc3)
            fc4 = slim.conv2d(fc3, 1024, [1, 1], scope='fc4', activation_fn=None)
            with tf.variable_scope('fc4_bn'):
                fc4 = tf.layers.batch_normalization(fc4, center=False, scale=False, training=bn_training)
            fc4 = tf.nn.relu(fc4)
            fc5 = slim.conv2d(fc4, self.output_size, [1, 1], scope='fc5', activation_fn=None)
            logit = tf.squeeze(fc5, [1, 2], name='SpatialSqueeze_2')
            softmax = slim.softmax(logit, scope='predictions')
            return softmax

    def validation_acc(self, sess, y_outputs, ph, real_data):
        x_val, y_val = real_data.val_set_x, real_data.val_set_y
        x, y = ph
        y_outputs = sess.run(y_outputs, feed_dict={x: x_val, y: y_val})
        correct_prediction = tf.equal(tf.argmax(y_outputs, axis=1), tf.argmax(y_val, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return sess.run(accuracy)

    def validation_acc_2inputs(self, sess, y_outputs, ph, real_data):
        x_val, y_val = real_data.val_set_x, real_data.val_set_y
        x_b1, x_b2, y_b = self.create_cmp_batch(x_val, y_val)
        x1, x2, y = ph
        y_outputs = sess.run(y_outputs, feed_dict={x1: x_b1, x2: x_b2, y: y_b})
        correct_prediction = tf.equal(tf.argmax(y_outputs, axis=1), tf.argmax(y_val, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return sess.run(accuracy)

    def read_cfg(self):
        # 创建管理对象
        conf = configparser.ConfigParser()
        # 读ini文件
        conf.read("cfg.ini", encoding="utf-8")  # python3
        return conf.getfloat("parameter", "learning_rate"),\
               conf.getfloat("parameter", "learning_rate_decay"),\
               conf.getint("parameter", "epoch")

    def train_baseline_net(self, model_save_path='./model_baseline/', op_freq=10):
        dataset = AVAImages()
        dataset.read_data(read_dir='AVA_data_score_bi/', flag=0)
        dataset.read_batch_cfg()
        learning_rate, learning_rate_decay, epoch = self.read_cfg()
        w, h, c = self.input_size
        with tf.name_scope("Inputs"):
            x = tf.placeholder(tf.float32, [None, w, h, c])
            y = tf.placeholder(tf.float32, [None, self.output_size])
            ph = x, y
        y_outputs = self.propagate(x)  # y_outputs = (None, 2)
        global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("Loss"):
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=y_outputs, labels=tf.argmax(y, 1))
            loss = tf.reduce_mean(entropy)
        with tf.name_scope("Train"):
            rate = tf.train.exponential_decay(learning_rate, global_step, 200, learning_rate_decay)  # 指数衰减学习率
            train_op = tf.train.AdamOptimizer(rate).minimize(loss, global_step=global_step)

        # with saver&sess
        # tf_vars = tf.trainable_variables(scope="resnet_v2_50")
        # saver = tf.train.Saver(var_list=tf_vars)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                while True:
                    # 遍历所有batch
                    x_b, y_b, end = dataset.load_next_batch_quicker()
                    train_op_, loss_, step = sess.run([train_op, loss, global_step], feed_dict={x: x_b, y: y_b})
                    if step % op_freq == 0:
                        print("training step {0}, loss {1}, validation acc {2}"
                              .format(step, loss_, self.validation_acc(sess, y_outputs, ph, dataset)))
                        saver.save(sess, model_save_path + 'my_model', global_step=global_step)
                    if end == 1:
                        break
            writer = tf.summary.FileWriter("logs/", tf.get_default_graph())
            writer.close()

    def get_uninitialized_variables(self, sess):
        global_vars = tf.global_variables()
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        print([str(i.name) for i in not_initialized_vars])
        return not_initialized_vars

    def create_cmp_batch(self, x_b, y_b, th1=1.15, th2=0.85):
        y = np.zeros((y_b.shape[0], self.output_size))

        index = [i for i in range(x_b.shape[0])]
        index.reverse()
        x_b_re = x_b[index]
        y_b_re = y_b[index]
        y_ori = y_b / y_b_re

        eye = np.eye(self.output_size)
        for i in range(y_ori.shape[0]):
            if y_ori[i] > th1:
                y[i] = eye[2]
            elif y_ori[i] > th2:
                y[i] = eye[1]
            else:
                y[i] = eye[0]

        return x_b, x_b_re, y

    def train_comparator(self, baseline_model='./model_baseline/', cmp_model='./model_cmp/', op_freq=10):
        # data set
        dataset = AVAImages()
        dataset.read_data(read_dir='AVA_data_score/', flag=0)
        dataset.read_batch_cfg()

        # graph
        tf.reset_default_graph()

        # parameter
        learning_rate, learning_rate_decay, epoch = self.read_cfg()

        # input
        w, h, c = self.input_size
        with tf.name_scope("Inputs"):
            x1 = tf.placeholder(tf.float32, [None, w, h, c])
            x2 = tf.placeholder(tf.float32, [None, w, h, c])
            y = tf.placeholder(tf.float32, [None, self.output_size])
            ph = x1, x2, y

        # network
        inputs = x1, x2
        output = self.propagate(inputs)
        global_step = tf.Variable(0, trainable=False)

        # loss
        with tf.name_scope("Loss"):
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=output, labels=tf.argmax(y, 1))
            loss = tf.reduce_mean(entropy)

        # train
        t_vars = tf.trainable_variables()  # 获取所有的变量
        with tf.name_scope("Train"):
            rate = tf.train.exponential_decay(learning_rate, global_step, 200, learning_rate_decay)  # 指数衰减学习率
            train_op = tf.train.AdamOptimizer(rate).minimize(loss, var_list=t_vars, global_step=global_step)

        # saver&re_saver
        variables_to_restore = slim.get_variables_to_restore(include=['resnet_v2_aes'])
        re_saver = tf.train.Saver(variables_to_restore)  # 建立一个saver 从已有的模型中恢复部分参数到网络中.
        saver = tf.train.Saver()  # 建立一个saver，训练的时候保存整个模型的ckpt

        # with saver&sess
        with tf.Session() as sess:
            # model_path = './model.ckpt'  # 后缀名称仅需要写ckpt即可,后面的00001-00000不必添加
            re_saver.restore(sess=sess, save_path=baseline_model + 'my_model-1')  # 恢复模型的参数到新的模型
            un_init = tf.variables_initializer(self.get_uninitialized_variables(sess))  # 获取没有初始化(通过已有model加载)的变量
            sess.run(un_init)  # 对没有初始化的变量进行初始化并训练.
            for i in range(epoch):
                while True:
                    pass
                    # 遍历所有batch
                    x_b, y_b, end = dataset.load_next_batch_quicker(read_dir='AVA_data_score/')
                    x_b1, x_b2, y_b = self.create_cmp_batch(x_b, y_b)
                    train_op_, loss_, step = sess.run([train_op, loss, global_step],
                                                      feed_dict={x1: x_b1, x2: x_b2, y: y_b})
                    if step % op_freq == 0:
                        print("training step {0}, loss {1}, validation acc {2}"
                              .format(step, loss_, self.validation_acc_2inputs(sess, output, ph, dataset)))
                        saver.save(sess, cmp_model + 'my_model', global_step=global_step)
                    if end == 1:
                        break
            writer = tf.summary.FileWriter("logs/", tf.get_default_graph())
            writer.close()
