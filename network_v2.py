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

    def score2style(self, inputs):
        # output = slim.stack(inputs, slim.fully_connected, [32, 14], scope='fc')
        output = slim.fully_connected(inputs, 14, scope='fc')
        return output

    def multi_task(self, inputs):
        _, softmax, _ = resnet_v2_4x4(inputs, num_classes=self.output_size)  # num_classess=2+14
        return softmax

    def MTCNN(self, inputs, training=True):
        with tf.variable_scope("theta"):
            l1 = slim.conv2d(inputs, 96, [11, 11], stride=2, padding='VALID', normalizer_fn=None, activation_fn=None, scope='conv1')
            l1_bn = tf.layers.batch_normalization(l1, training=training, name='bn1')
            l1_pool = slim.max_pool2d(l1_bn, [3, 3], padding='VALID', scope='pool1')

            l2 = slim.conv2d(l1_pool, 128, [5, 5], stride=1, padding='SAME', normalizer_fn=None, activation_fn=None, scope='conv2')
            l2_bn = tf.layers.batch_normalization(l2, training=training, name='bn2')
            l2_pool = slim.max_pool2d(l2_bn, [3, 3], padding='SAME', scope='pool2')

            l3 = slim.conv2d(l2_pool, 128, [3, 3], stride=1, padding='SAME', normalizer_fn=None, activation_fn=None, scope='conv3')

            l4 = slim.conv2d(l3, 256, [3, 3], stride=1, padding='SAME', normalizer_fn=None, activation_fn=None, scope='conv4')
            l4_bn = tf.layers.batch_normalization(l4, training=training, name='bn4')
            l4_pool = slim.max_pool2d(l4_bn, [3, 3], scope='pool4')

            l4_pool_flat = tf.reshape(l4_pool, [-1, 13 * 13 * 256], name='flat1')
            l5 = slim.fully_connected(l4_pool_flat, 4096)
            l6 = slim.fully_connected(l5, 4096)

        with tf.variable_scope("W"):
            l7_score = slim.fully_connected(l6, 10)
            l7_style = slim.fully_connected(l6, 14)
        return tf.concat([l7_score, l7_style], axis=1, name='concat')

    def eval_binary_acc(self, model_path='./model_ulti/'):
        dataset = AVAImages()
        dataset.read_data('AVA_data_score_mean_var_style/', flag=2)
        x_test, y_test = dataset.test_set_x, np.int64(dataset.test_set_y[:, 0] >= 5)  # 前提test_set_y.shape=(num,)
        y_test = y_test[:, np.newaxis]
        w, h, c = self.input_size
        x = tf.placeholder(tf.float32, [None, w, h, c])
        y = tf.placeholder(tf.float32, [None, self.output_size])  # y.shape=(num, 1)
        y_outputs = self.multi_task(x)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                y_outputs = sess.run(y_outputs, feed_dict={x: x_test})
                # np.squeeze(y_outputs)
                y_pred = np.int64(y_outputs[:, 0] >= 0.5)[:, np.newaxis]
                correct_prediction = tf.equal(y_pred, y_test)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print("test accuracy - %g" % sess.run(accuracy))
            else:
                print('No checkpoing file found')
                return

    def validation_acc(self, sess, y_outputs, ph, real_data):
        x_val, y_val = real_data.val_set_x, np.int64(real_data.val_set_y[:, 0] >= 5)
        y_val = y_val[:, np.newaxis]
        x, y = ph
        y_outputs = sess.run(y_outputs, feed_dict={x: x_val})
        y_pred = np.int64(y_outputs[:, 0] >= 0.5)[:, np.newaxis]
        correct_prediction = tf.equal(y_pred, y_val)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return sess.run(accuracy)

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
        y_outputs = self.score2style(x)  # y_outputs = (None, 2)
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

    def train_multi_task(self, model_save_path='./model_ulti/', op_freq=10, val=True):
        folder = os.path.exists(model_save_path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(model_save_path)  # makedirs 创建文件时如果路径不存在会创建这个路径

        dataset = AVAImages()
        if val:
            dataset.read_data(read_dir='AVA_data_score_mean_var_style/', flag=0)
        dataset.read_batch_cfg()
        learning_rate, learning_rate_decay, epoch, a, b, c, d = self.read_cfg()
        w, h, c = self.input_size
        with tf.name_scope("Inputs"):
            x = tf.placeholder(tf.float32, [None, w, h, c])
            y = tf.placeholder(tf.float32, [None, self.output_size])
            # x_mv = tf.placeholder(tf.float32, [None, 2])
            ph = x, y
        y_outputs = self.multi_task(x)  # y_outputs = (None, 2)
        y_mv = self.score2style(y_outputs[:, 0: 2])
        global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("Loss"):
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=y_outputs[:, 2: 15], labels=tf.argmax(y[:, 2: 15], 1))
            style_loss = tf.reduce_mean(entropy)
            mse0 = tf.losses.mean_squared_error(y[:, 0], y_outputs[:, 0])
            mse1 = tf.losses.mean_squared_error(y[:, 1], y_outputs[:, 1])
            entropy_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=y_mv, labels=tf.argmax(y[:, 2: 15], 1))
            style_loss_2 = tf.reduce_mean(entropy_2)
            loss = a * style_loss + b * mse0 + c * mse1 + d * style_loss_2
        with tf.name_scope("Train"):
            rate = tf.train.exponential_decay(learning_rate, global_step, 200, learning_rate_decay)  # 指数衰减学习率
            train_op = tf.train.AdamOptimizer(rate).minimize(loss, global_step=global_step)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                while True:
                    # 遍历所有batch
                    x_b, y_b, end = dataset.load_next_batch_quicker(read_dir='AVA_data_score_mean_var_style/')
                    train_op_, loss_, step = sess.run([train_op, loss, global_step],
                                                      feed_dict={x: x_b, y: y_b})
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

    def KLD(self, x, y):
        x_add = tf.add(x, 1e-10)
        y_add = tf.add(y, 1e-10)
        X = tf.distributions.Categorical(probs=x_add)
        Y = tf.distributions.Categorical(probs=y_add)
        return tf.distributions.kl_divergence(X, Y)

    def mu(self, x):
        one = tf.ones([1, 1])
        result = tf.cond(pred=tf.less(x, 0.1),
                         true_fn=lambda: tf.divide(tf.log(tf.add(x, 1)), tf.add(tf.log(tf.add(x, 1)), 1)),
                         false_fn=lambda: one)
        return result

    def kurtosis(self, x):
        mean, variance = tf.nn.moments(x, axes=1)
        return tf.reduce_mean(tf.subtract(tf.divide(tf.reduce_sum(tf.pow(tf.subtract(x, mean), 4), axis=1),
                                     tf.multiply(tf.square(variance), tf.cast(tf.shape(x)[1], tf.float32))), 3))

    def r_kurtosis(self, y_outputs):
        ty = tf.divide(1, tf.abs(tf.subtract(self.kurtosis(y_outputs), 3)))
        return self.mu(ty)

    def distribution_loss(self, y_outputs, y):
        ym = tf.multiply(0.5, tf.add(y_outputs, y))
        jsd = tf.add(tf.multiply(0.5, self.KLD(y_outputs, ym)),
                     tf.multiply(0.5, self.KLD(y, ym)))
        return tf.multiply(self.r_kurtosis(y_outputs), jsd)

    def style_loss(self, y_outputs, y):
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_outputs, labels=y)
        return tf.reduce_mean(entropy)

    def sum_theta_and_W(self):
        var = tf.trainable_variables()
        s = tf.Variable([0], dtype=tf.float32, name='theta_and_W')
        conv_index = -1
        fc_index = -1
        filter_num = [96, 128, 128, 256]
        filter_size = [11, 5, 3, 3]
        channel_num = [3, 96, 128, 128]
        fc_insize = [43264, 4096, 4096, 4096]
        for v in var:
            if "conv" in v.name:
                if "weights" in v.name:
                    conv_index += 1
                    for i in range(filter_num[conv_index]):
                        for c in range(channel_num[conv_index]):
                            w = tf.reshape(v[:, :, c, i], [-1, filter_size[conv_index] * filter_size[conv_index]])
                            s = tf.add(s, tf.reduce_sum(tf.square(w)))
                elif "biases" in v.name:
                    s = tf.add(s, tf.reduce_sum(tf.square(v)))
            if "bn" in v.name:
                s = tf.add(s, tf.reduce_sum(tf.square(v)))
            if "fully" in v.name:
                fc_index += 1
                for i in range(fc_insize[fc_index]):
                    s = tf.add(s, tf.reduce_sum(tf.square(v[i])))
        return s

    def tr_W(self):
        var = tf.trainable_variables()
        wa = tf.get_default_graph().get_tensor_by_name('W/fully_connected/weights:0')
        ws = tf.get_default_graph().get_tensor_by_name('W/fully_connected_1/weights:0')
        W = tf.concat([wa, ws], axis=1)
        omega = tf.divide(tf.sqrt(tf.matmul(tf.transpose(W), W)), tf.linalg.trace(tf.sqrt(tf.matmul(tf.transpose(W), W))))
        return tf.linalg.trace(tf.matmul(tf.matmul(W, omega), tf.transpose(W)))

    def train_MTCNN(self, data='AVA_data_score_dis_style/', model_save_path='./model_MTCNN/', op_freq=10, val=True):
        folder = os.path.exists(model_save_path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(model_save_path)  # makedirs 创建文件时如果路径不存在会创建这个路径

        dataset = AVAImages()
        if val:
            dataset.read_data(read_dir=data, flag=0)
        dataset.read_batch_cfg()
        learning_rate, learning_rate_decay, epoch, a, b, c, d = self.read_cfg()
        w, h, c = self.input_size
        with tf.name_scope("Inputs"):
            x = tf.placeholder(tf.float32, [None, w, h, c])
            y = tf.placeholder(tf.float32, [None, self.output_size])
            ph = x, y
        y_outputs = self.MTCNN(x, True)  # y_outputs = (None, 2)
        y_mv = self.score2style(y_outputs[:, 0: 10])
        global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("Loss"):

            loss = tf.add_n([self.distribution_loss(y_outputs[:, 0: 10], y[:, 0: 10]),
                             tf.multiply(1 / 14, self.style_loss(y_outputs[:, 10:], y[:, 10:])),
                             tf.contrib.layers.apply_regularization(
                                 regularizer=tf.contrib.layers.l2_regularizer(a, scope=None),
                                 weights_list=tf.trainable_variables()),
                             self.tr_W(),
                             tf.multiply(b, self.style_loss(y_mv, y[:, 10:]))
                             ])
        with tf.name_scope("Train"):
            rate = tf.train.exponential_decay(learning_rate, global_step, 200, learning_rate_decay)  # 指数衰减学习率
            train_op = tf.train.AdamOptimizer(rate).minimize(loss, global_step=global_step)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                while True:
                    # 遍历所有batch
                    x_b, y_b, end = dataset.load_next_batch_quicker(read_dir=data)
                    train_op_, loss_, step = sess.run([train_op, loss, global_step],
                                                      feed_dict={x: x_b, y: y_b})
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


