from resnet import resnet_v2_4x4
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

    def eval_binary_acc(self, model_path='./model2'):
        dataset = AVAImages()
        dataset.read_data('AVA_data_score')
        x_test, y_test = dataset.test_set_x, np.int64(dataset.test_set_y >= 5)  # 前提test_set_y.shape=(num,)
        y_test = y_test[:, np.newaxis]
        w, h, c = self.input_size
        x = tf.placeholder(tf.float32, [None, w, h, c])
        y = tf.placeholder(tf.float32, [None, self.output_size])  # y.shape=(num, 1)
        y_outputs = self.propagate(x)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                y_outputs = sess.run(y_outputs, feed_dict={x: x_test, y: y_test})
                # np.squeeze(y_outputs)
                y_outputs[y_outputs < 5] = 0
                y_outputs[y_outputs >= 5] = 1
                correct_prediction = tf.equal(y_outputs, y_test)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print("test accuracy - %g" % sess.run(accuracy))
            else:
                print('No checkpoing file found')
                return

    def propagate(self, inputs):
        if self.net == "predict_bi_class":
            x = inputs
            return self.score_net(x)

    def validation_acc(self, sess, y_outputs, ph, real_data):
        x_val, y_val = real_data.val_set_x, real_data.val_set_y
        x, y = ph
        y_outputs = sess.run(y_outputs, feed_dict={x: x_val, y: y_val})
        correct_prediction = tf.equal(y_outputs, y_val)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return sess.run(accuracy)

    def read_cfg(self):
        curpath = os.path.dirname(os.path.realpath(__file__))
        cfgpath = os.path.join(curpath, "cfg.ini")
        print(cfgpath)  # cfg.ini的路径
        # 创建管理对象
        conf = configparser.ConfigParser()
        # 读ini文件
        conf.read(cfgpath, encoding="utf-8")  # python3
        return int(conf.get("parameter", "batch_size")),\
               float(conf.get("parameter", "learning_rate")),\
               float(conf.get("parameter", "learning_rate_decay")),\
               int(conf.get("parameter", "epoch"))

    def train_baseline_net(self, model_save_path='./model_baseline/'):
        dataset = AVAImages()
        dataset.read_data(read_dir='AVA_data_score_bi/', flag=0)
        dataset.read_batch_cfg()
        batch_size, learning_rate, learning_rate_decay, epoch = self.read_cfg()
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
                    if step % 5 == 0:
                        print("training step {0}, loss {1}, validation loss {2}"
                              .format(step, loss_, self.validation_acc(sess, y_outputs, ph, dataset)))
                        saver.save(sess, model_save_path + 'my_model', global_step=global_step)
                    if end == 1:
                        break
            writer = tf.summary.FileWriter("logs/", tf.get_default_graph())
            writer.close()

