from resnet import resnet_v2_4x4
from resnet import resnet_v2_4x4_shallow
from data import AVAImages
import configparser
import os
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim


def condition_wa_ws(time, l6, l7):
    return tf.less(time, 24)


def body_wa_ws(time, l6, l7):
    l7 = l7.write(time, slim.fully_connected(l6, 1, activation_fn=tf.nn.sigmoid))
    return time + 1, l6, l7


def condition_w_list(time, w_b, w_list):
    return tf.less(time, 24)


def body_w_list(time, w_b, w_list):
    w_list = w_list.write(time, w_b[time: time + 2])
    return time + 1, w_b, w_list


def condition_top_list(time, train_theta, w_list, lr_list, train_op_list, loss, global_step):
    return tf.less(time, 24)


def body_top_list(time, train_theta, w_list, lr_list, train_op_list, loss, global_step):
    train_op_list = train_op_list.write(time,
                                        tf.train.AdamOptimizer(lr_list.read(time)).
                                        minimize(loss, global_step=global_step, var_list=train_theta + w_list.read(time)))
    return time + 1, train_theta, w_list, lr_list, train_op_list, loss, global_step


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


def validation_acc(sess, y_outputs, ph, real_data):
    x_val, y_val = real_data.val_set_x, np.int64(real_data.val_set_y[:, 0] >= 5)
    y_val = y_val[:, np.newaxis]
    x, y = ph
    y_outputs = sess.run(y_outputs, feed_dict={x: x_val})
    y_pred = np.int64(y_outputs[:, 0] >= 0.5)[:, np.newaxis]
    correct_prediction = tf.equal(y_pred, y_val)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return sess.run(accuracy)


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
            # last layer do not need activation function
            # l7 = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            # time = tf.constant(self.output_size)
            # result = tf.while_loop(condition_wa_ws, body_wa_ws, loop_vars=[time, l6, l7])
            # last_time, _, last_out = result
            # final_out_l7 = last_out.stack()
            l7_list = [slim.fully_connected(l6, 1, activation_fn=tf.nn.sigmoid) for i in range(self.output_size)]
        return l7_list

    def eval_binary_acc(self, model_path='./model_MTCNN/'):
        dataset = AVAImages()
        dataset.read_data('AVA_data_score_dis_style/', flag="test")
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

    def read_cfg(self):
        # 创建管理对象
        conf = configparser.ConfigParser()
        # 读ini文件
        conf.read("cfg.ini", encoding="utf-8")  # python3
        return conf.getfloat("parameter", "learning_rate"),\
               conf.getfloat("parameter", "learning_rate_decay"),\
               conf.getint("parameter", "epoch"), \
               conf.getfloat("parameter", "alpha"), \
               conf.getfloat("parameter", "beta"), \
               conf.getfloat("parameter", "gamma"), \
               conf.getfloat("parameter", "theta"), \
               conf.getint("parameter", "switch")

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
                                  .format(step, loss_, validation_acc(sess, y_outputs, ph, dataset)))
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
                                  .format(step, loss_, validation_acc(sess, y_outputs, ph, dataset)))
                        else:
                            print("training step {0}, loss {1}".format(step, loss_))
                        saver.save(sess, model_save_path + 'my_model', global_step=global_step)
                    if end == 1:
                        break
            writer = tf.summary.FileWriter("logs/", tf.get_default_graph())
            writer.close()

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

    def KLD(self, x, y):
        x_add = self.fixprob(x)
        y_add = self.fixprob(y)
        X = tf.distributions.Categorical(probs=x_add)
        Y = tf.distributions.Categorical(probs=y_add)
        return tf.distributions.kl_divergence(X, Y)

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

    # def JSD(self, y_outputs, y):
    #     tb = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    #     y1 = y_outputs[:, 0]
    #     y2 = y[:, 0]
    #     add_op = tf.add(0.5 * y1, 0.5 * y2)
    #     divide_op = tf.divide(y2, add_op)
    #     tb = tb.write(0, tf.multiply(y2, tf.log(divide_op)))
    #     y1 = tf.reduce_sum(y_outputs[:, 0: 2], axis=1)
    #     y2 = tf.reduce_sum(y[:, 0: 2], axis=1)
    #     tb = tb.write(1, tf.multiply(y2, tf.log(tf.divide(y2, tf.add(0.5 * y1, 0.5 * y2)))))
    #     y1 = tf.reduce_sum(y_outputs[:, 0: 3], axis=1)
    #     y2 = tf.reduce_sum(y[:, 0: 3], axis=1)
    #     tb = tb.write(2, tf.multiply(y2, tf.log(tf.divide(y2, tf.add(0.5 * y1, 0.5 * y2)))))
    #     y1 = tf.reduce_sum(y_outputs[:, 0: 4], axis=1)
    #     y2 = tf.reduce_sum(y[:, 0: 4], axis=1)
    #     tb = tb.write(3, tf.multiply(y2, tf.log(tf.divide(y2, tf.add(0.5 * y1, 0.5 * y2)))))
    #     y1 = tf.reduce_sum(y_outputs[:, 0: 5], axis=1)
    #     y2 = tf.reduce_sum(y[:, 0: 5], axis=1)
    #     tb = tb.write(4, tf.multiply(y2, tf.log(tf.divide(y2, tf.add(0.5 * y1, 0.5 * y2)))))
    #     y1 = tf.reduce_sum(y_outputs[:, 0: 6], axis=1)
    #     y2 = tf.reduce_sum(y[:, 0: 6], axis=1)
    #     tb = tb.write(5, tf.multiply(y2, tf.log(tf.divide(y2, tf.add(0.5 * y1, 0.5 * y2)))))
    #     y1 = tf.reduce_sum(y_outputs[:, 0: 7], axis=1)
    #     y2 = tf.reduce_sum(y[:, 0: 7], axis=1)
    #     tb = tb.write(6, tf.multiply(y2, tf.log(tf.divide(y2, tf.add(0.5 * y1, 0.5 * y2)))))
    #     y1 = tf.reduce_sum(y_outputs[:, 0: 8], axis=1)
    #     y2 = tf.reduce_sum(y[:, 0: 8], axis=1)
    #     tb = tb.write(7, tf.multiply(y2, tf.log(tf.divide(y2, tf.add(0.5 * y1, 0.5 * y2)))))
    #     y1 = tf.reduce_sum(y_outputs[:, 0: 9], axis=1)
    #     y2 = tf.reduce_sum(y[:, 0: 9], axis=1)
    #     tb = tb.write(8, tf.multiply(y2, tf.log(tf.divide(y2, tf.add(0.5 * y1, 0.5 * y2)))))
    #     y1 = tf.reduce_sum(y_outputs[:, 0: 10], axis=1)
    #     y2 = tf.reduce_sum(y[:, 0: 10], axis=1)
    #     tb = tb.write(9, tf.multiply(y2, tf.log(tf.divide(y2, tf.add(0.5 * y1, 0.5 * y2)))))
    #     tb = tb.stack()
    #     return tb

    def dis_reg(self, y_outputs, fix_marg):
        y_outputs_sum = tf.reduce_sum(y_outputs, axis=1, keep_dims=True)
        multi = tf.tile(y_outputs_sum, multiples=[1, fix_marg])
        return y_outputs / multi

    def distribution_loss(self, y_outputs, y, th, fix_marg):
        # ym = tf.multiply(0.5, tf.add(y_outputs, y))
        # jsd = tf.reduce_mean(tf.add(tf.multiply(0.5, self.KLD(y_outputs, ym)), tf.multiply(0.5, self.KLD(y, ym))))
        # jsd = tf.reduce_mean(tf.add(tf.multiply(0.5, tf.keras.losses.kullback_leibler_divergence(y_outputs, ym)),
        #                             tf.multiply(0.5, tf.keras.losses.kullback_leibler_divergence(y, ym))))
        y_outputs = self.dis_reg(y_outputs, fix_marg)
        jsd = self.JSD(y_outputs, y)
        return tf.multiply(self.r_kurtosis(y_outputs, th), jsd)

    def cross_distribution_loss(self, y_outputs, y):
        # ym = tf.multiply(0.5, tf.add(y_outputs, y))
        # jsd = tf.reduce_mean(tf.add(tf.multiply(0.5, self.KLD(y_outputs, ym)), tf.multiply(0.5, self.KLD(y, ym))))
        # return jsd
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_outputs, labels=y)
        return tf.reduce_mean(entropy)

    def style_loss(self, y_outputs, y):
        # entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_outputs, labels=y)
        # return tf.reduce_mean(entropy)
        entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y_outputs, labels=tf.argmax(y, 1))
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

    def cov_matrix(self, W):
        mean = tf.reduce_mean(W, axis=1, keep_dims=True)
        # mean = tf.expand_dims(mean, 1)
        sub = tf.subtract(W, mean)
        cov = tf.divide(tf.matmul(tf.transpose(sub), sub), tf.cast(tf.shape(W)[0], tf.float32))
        # return cov
        return tf.shape(cov)

    def get_W(self):
        w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='W')
        w = [w[i] for i in range(0, len(w), 2)]
        w = tf.concat(w, axis=1)
        return w

    def ini_omega(self, task_num):
        with tf.variable_scope('Omega'):
            o = tf.Variable(tf.eye(task_num, dtype=tf.float32) / tf.cast(task_num, tf.float32),
                            dtype=tf.float32, name='omega', trainable=True)
        return o

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
        # 'W/fully_connected_1/weights:0'
        # w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='W')
        if var.name[0] == 'W':
            if var.name == 'W/fully_connected/weights:0' or var.name == 'W/fully_connected/biases:0':
                grad = tf.multiply(grad, omega[taskid][0])
            else:
                s = int(var.name[18])
                grad = tf.multiply(grad, omega[taskid][s])
        return grad

    def train_MTCNN(self, data='AVA_data_score_dis_style/', model_save_path='./model_MTCNN/', op_freq=10, val=True, task_marg=10, fix_marg=10):
        folder = os.path.exists(model_save_path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(model_save_path)  # makedirs 创建文件时如果路径不存在会创建这个路径

        dataset = AVAImages()
        if val:
            dataset.read_data(read_dir=data, flag="val")
        dataset.read_data(read_dir=data, flag="Th")
        dataset.Th_y[:, 0: fix_marg] = self.fixprob(dataset.Th_y[:, 0: fix_marg])
        dataset.read_batch_cfg()
        learning_rate, learning_rate_decay, epoch, alpha, beta, gamma, theta, switch = self.read_cfg()
        w, h, c = self.input_size
        with tf.name_scope("Inputs"):
            x = tf.placeholder(tf.float32, [None, w, h, c])
            y = tf.placeholder(tf.float32, [None, self.output_size])
            th = tf.placeholder(tf.float32)
            task_id = tf.placeholder(tf.int32)
            ph = x, y
        y_list = self.MTCNN(x, True)  # y_outputs = (None, 24)
        y_outputs = tf.concat(y_list, axis=1)
        y_mv = self.score2style(y_outputs[:, 0: 10])
        global_step = tf.Variable(0, trainable=False)
        upgrade_global_step = tf.assign(global_step, tf.add(global_step, 1))

        with tf.name_scope("Loss"):
            cross_val_loss = self.JSD(y_outputs[:, 0: task_marg], y[:, 0: task_marg])
            W = self.get_W()
            omega = self.ini_omega(self.output_size)
            tr_W_omega_WT = self.tr(W, omega)
            loss = self.distribution_loss(y_outputs[:, 0: task_marg], y[:, 0: task_marg], th, fix_marg) + \
                   gamma * self.style_loss(y_outputs[:, task_marg:], y[:, task_marg:]) + \
                   tf.contrib.layers.apply_regularization(
                       regularizer=tf.contrib.layers.l2_regularizer(alpha, scope=None),
                       weights_list=tf.trainable_variables()) + \
                   theta * tr_W_omega_WT + beta * self.style_loss(y_mv, y[:, 10:])
            # loss = tf.add_n([self.distribution_loss(y_outputs[:, 0: 10], y[:, 0: 10], th),
            #                  tf.multiply(gamma, self.style_loss(y_outputs[:, 10:], y[:, 10:])),
            #                  tf.contrib.layers.apply_regularization(
            #                      regularizer=tf.contrib.layers.l2_regularizer(alpha, scope=None),
            #                      weights_list=tf.trainable_variables()),
            #                  tf.multiply(theta, self.tr_W()),
            #                  tf.multiply(beta, self.style_loss(y_mv, y[:, 10:]))
            #                  ])
        with tf.name_scope("Train"):
            # get variables
            train_theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Theta')
            WW = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='W')
            # w = [w[i] for i in range(0, len(WW), 2)]
            omegaaa = tf.get_default_graph().get_tensor_by_name('Loss/Omega/omega:0')

            opt = tf.train.AdamOptimizer(learning_rate)
            gradient_var_all = opt.compute_gradients(loss, var_list=train_theta+WW)
            capped_gvs = [(self.scalar_for_weights(grad, var, omegaaa, task_id), var)
                          for grad, var in gradient_var_all]
            train_op = opt.apply_gradients(capped_gvs)
            train_op_all = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step,
                                                                          var_list=train_theta+WW)
            train_op_omega = tf.assign(omega, self.update_omega(W))

        saver = tf.train.Saver()
        cross_val_loss_transfer = 0
        train_theta_and_W_first = 10
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                while True:
                    # 遍历所有batch
                    x_b, y_b, end = dataset.load_next_batch_quicker(read_dir=data)
                    y_b[:, 0: fix_marg] = self.fixprob(y_b[:, 0: fix_marg])
                    step = sess.run(global_step)
                    if step < train_theta_and_W_first:
                        cross_val_loss_transfer, y_outputs_, tr = sess.run(
                            [cross_val_loss, y_outputs, tr_W_omega_WT], feed_dict={x: dataset.Th_x, y: dataset.Th_y})
                        train_op_, loss_ = sess.run([train_op_all, loss],
                                                    feed_dict={x: x_b, y: y_b, th: cross_val_loss_transfer})

                    elif np.random.rand() < 0.5:
                        train_op_ = sess.run(train_op_omega)
                        sess.run(upgrade_global_step)
                    else:
                        cross_val_loss_transfer, y_outputs_, tr = sess.run(
                            [cross_val_loss, y_outputs, tr_W_omega_WT], feed_dict={x: dataset.Th_x, y: dataset.Th_y})
                        for taskid in range(self.output_size):
                            train_op_, loss_ = sess.run([train_op, loss],
                                                        feed_dict={x: x_b, y: y_b,
                                                                   th: cross_val_loss_transfer, task_id: taskid})
                        sess.run(upgrade_global_step)

                    if step % op_freq == 0:
                        if val:
                            print("training step {0}, loss {1}, validation acc {2}"
                                  .format(step, loss_, validation_acc(sess, y_outputs, ph, dataset)))
                        else:
                            print("training step {0}, loss {1}".format(step, loss_))
                        saver.save(sess, model_save_path + 'my_model', global_step=global_step)
                    if end == 1:
                        break
            writer = tf.summary.FileWriter("logs/", tf.get_default_graph())
            writer.close()

    def train_MTCNN_continue(self, data='AVA_data_score_dis_style/', model_read_path='./model_MTCNN/', model_save_path='./model_MTCNN_continue/', op_freq=10, val=True, task_marg=10, fix_marg=10):
        folder = os.path.exists(model_save_path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(model_save_path)  # makedirs 创建文件时如果路径不存在会创建这个路径

        dataset = AVAImages()
        if val:
            dataset.read_data(read_dir=data, flag="val")
        dataset.read_data(read_dir=data, flag="Th")
        dataset.Th_y[:, 0: fix_marg] = self.fixprob(dataset.Th_y[:, 0: fix_marg])
        dataset.read_batch_cfg()
        learning_rate, learning_rate_decay, epoch, alpha, beta, gamma, theta, switch = self.read_cfg()
        w, h, c = self.input_size
        with tf.name_scope("Inputs"):
            x = tf.placeholder(tf.float32, [None, w, h, c])
            y = tf.placeholder(tf.float32, [None, self.output_size])
            th = tf.placeholder(tf.float32)
            task_id = tf.placeholder(tf.int32)
            ph = x, y
        y_list = self.MTCNN(x, True)  # y_outputs = (None, 24)
        y_outputs = tf.concat(y_list, axis=1)
        y_mv = self.score2style(y_outputs[:, 0: 10])
        global_step = tf.Variable(0, trainable=False)
        upgrade_global_step = tf.assign(global_step, tf.add(global_step, 1))

        with tf.name_scope("Loss"):
            cross_val_loss = self.JSD(y_outputs[:, 0: task_marg], y[:, 0: task_marg])
            W = self.get_W()
            omega = self.ini_omega(self.output_size)
            tr_W_omega_WT = self.tr(W, omega)
            loss = self.distribution_loss(y_outputs[:, 0: task_marg], y[:, 0: task_marg], th, fix_marg) + \
                   gamma * self.style_loss(y_outputs[:, task_marg:], y[:, task_marg:]) + \
                   tf.contrib.layers.apply_regularization(
                       regularizer=tf.contrib.layers.l2_regularizer(alpha, scope=None),
                       weights_list=tf.trainable_variables()) + \
                   theta * tr_W_omega_WT + beta * self.style_loss(y_mv, y[:, 10:])
            # loss = self.distribution_loss(y_outputs[:, 0: task_marg], y[:, 0: task_marg], th) + \
            #        gamma * self.style_loss(y_outputs[:, task_marg:], y[:, task_marg:]) + \
            #        tf.contrib.layers.apply_regularization(
            #            regularizer=tf.contrib.layers.l2_regularizer(alpha, scope=None),
            #            weights_list=tf.trainable_variables()) + \
            #        theta * tr_W_omega_WT + beta * self.style_loss(y_mv, y[:, 10:])
        with tf.name_scope("Train"):
            # get variables
            train_theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Theta')
            WW = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='W')
            omegaaa = tf.get_default_graph().get_tensor_by_name('Loss/Omega/omega:0')

            opt = tf.train.AdamOptimizer(learning_rate)
            gradient_var_all = opt.compute_gradients(loss, var_list=train_theta+WW)
            capped_gvs = [(self.scalar_for_weights(grad, var, omegaaa, task_id), var)
                          for grad, var in gradient_var_all]
            train_op = opt.apply_gradients(capped_gvs)
            train_op_all = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step,
                                                                          var_list=train_theta+WW)
            train_op_omega = tf.assign(omega, self.update_omega(W))

        saver = tf.train.Saver()
        re_saver = tf.train.Saver()
        train_theta_and_W_first = 10
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
                        cross_val_loss_transfer, y_outputs_, tr = sess.run(
                            [cross_val_loss, y_outputs, tr_W_omega_WT], feed_dict={x: dataset.Th_x, y: dataset.Th_y})
                        train_op_, loss_ = sess.run([train_op_all, loss],
                                                    feed_dict={x: x_b, y: y_b, th: cross_val_loss_transfer})

                    elif np.random.rand() < 0.5:
                        train_op_ = sess.run(train_op_omega)
                        sess.run(upgrade_global_step)
                    else:
                        cross_val_loss_transfer, y_outputs_, tr = sess.run(
                            [cross_val_loss, y_outputs, tr_W_omega_WT], feed_dict={x: dataset.Th_x, y: dataset.Th_y})
                        for taskid in range(self.output_size):
                            train_op_, loss_ = sess.run([train_op, loss],
                                                        feed_dict={x: x_b, y: y_b,
                                                                   th: cross_val_loss_transfer, task_id: taskid})
                        sess.run(upgrade_global_step)

                    if step % op_freq == 0:
                        if val:
                            print("training step {0}, loss {1}, validation acc {2}"
                                  .format(step, loss_, validation_acc(sess, y_outputs, ph, dataset)))
                        else:
                            print("training step {0}, loss {1}".format(step, loss_))
                        saver.save(sess, model_save_path + 'my_model', global_step=global_step)
                    if end == 1:
                        break
            writer = tf.summary.FileWriter("logs/", tf.get_default_graph())
            writer.close()

