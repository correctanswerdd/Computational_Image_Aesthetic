import tensorflow as tf
import numpy as np
import configparser
from resnet import resnet_v2_50
from AlexNet import inference
from dataset import AVAImages
slim = tf.contrib.slim


# ###############################network architecture###########################################
def score2style(inputs):
    with tf.variable_scope("Cor_Matrix"):
        output = slim.fully_connected(inputs, 14, scope='fc')
    return output


def MTCNN_v3(inputs, outputs, training=True):
    with tf.variable_scope("Theta"):
        feature_vec, _ = inference(images=inputs)

    with tf.variable_scope("W"):
        l7_list1 = [slim.fully_connected(feature_vec, 1) for i in range(10)]
        l7_concat1 = tf.concat(l7_list1, axis=1)
        l7_concat1 = l7_concat1 / tf.reduce_sum(l7_concat1, axis=1, keep_dims=True)

        l7_list2 = [slim.fully_connected(feature_vec, 1) for i in range(outputs-10)]
        l7_concat2 = tf.concat(l7_list2, axis=1)

    return tf.concat([l7_concat1, l7_concat2], axis=1, name='concat')


def MTCNN_v2(inputs, outputs, training=True):
    with tf.variable_scope("Theta"):
        l1, _ = resnet_v2_50(inputs=inputs, num_classes=4096, is_training=training)
        feature_vec = slim.fully_connected(l1, 4096)
    with tf.variable_scope("W"):
        l7_list1 = [slim.fully_connected(feature_vec, 1) for i in range(10)]
        l7_concat1 = tf.concat(l7_list1, axis=1)
        l7_concat1 = l7_concat1 / tf.reduce_sum(l7_concat1, axis=1, keep_dims=True)

        l7_list2 = [slim.fully_connected(feature_vec, 1) for i in range(outputs-10)]
        l7_concat2 = tf.concat(l7_list2, axis=1)

        # for i in range(10):
        #     x = slim.fully_connected(feature_vec, 1)
        #     l7_concat1 = x if i == 0 else tf.concat([l7_concat1, x], axis=1)
        # for i in range(outputs-10):
        #     x = slim.fully_connected(feature_vec, 1)
        #     l7_concat2 = x if i == 0 else tf.concat([l7_concat2, x], axis=1)

    return tf.concat([l7_concat1, l7_concat2], axis=1, name='concat')


def MTCNN(inputs, outputs, training=True):
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
        l7_list1 = [slim.fully_connected(l6, 1) for i in range(10)]
        l7_concat1 = tf.concat(l7_list1, axis=1)
        l7_concat1 = l7_concat1 / tf.reduce_sum(l7_concat1, axis=1, keep_dims=True)

        l7_list2 = [slim.fully_connected(l6, 1) for i in range(outputs - 10)]
        l7_concat2 = tf.concat(l7_list2, axis=1)
    return tf.concat([l7_concat1, l7_concat2], axis=1, name='concat')


# #####################################################################################
def read_cfg(task="Skill-MTCNN"):
    # 创建管理对象
    conf = configparser.ConfigParser()
    # 读ini文件
    conf.read("config.ini", encoding="utf-8")  # python3
    if task == "Skill-MTCNN":
        return conf.getfloat(task, "learning_rate"), \
               conf.getfloat(task, "learning_rate_decay"), \
               conf.getint(task, "epoch"), \
               conf.getfloat(task, "alpha"), \
               conf.getfloat(task, "gamma"), \
               conf.getfloat(task, "theta")
    elif task == "Cor_Matrix":
        return conf.getfloat(task, "learning_rate"), \
               conf.getfloat(task, "learning_rate_decay"), \
               conf.getint(task, "epoch")


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


def fixprob(att):
    assert att.shape[1] == 10
    att = att + 1e-9
    _sum = np.sum(att, axis=1, keepdims=True)
    att = att / _sum
    att = np.clip(att, 1e-9, 1.0)
    return att


def tf_fixprob(att):
    att = att + 1e-9
    _sum = tf.reduce_sum(att, reduction_indices=1, keep_dims=True)
    att = att / _sum
    att = tf.clip_by_value(att, 1e-9, 1.0)
    return att


def mu(x, th):
    one = tf.constant(1, dtype=tf.float32)
    result = tf.cond(pred=tf.less(x, th),
                     true_fn=lambda: tf.divide(tf.log(tf.add(x, 1)), tf.add(tf.log(tf.add(x, 1)), 1)),
                     false_fn=lambda: one)
    return result


def kus(x):
    mean, variance = tf.nn.moments(x, axes=1)
    sub_2 = tf.expand_dims(mean, 0)
    sub_1 = tf.transpose(x)
    sub_op = tf.subtract(sub_1, sub_2)
    return tf.reduce_mean(tf.divide(tf.reduce_sum(tf.pow(tf.transpose(sub_op), 4), axis=1),
                          tf.multiply(tf.square(variance), tf.cast(tf.shape(x)[1], tf.float32))))


def r_kurtosis(y_outputs, th):
    ty = 1. / tf.abs(kus(y_outputs) - 3)
    return mu(ty, th)


def JSD(y_outputs, y):
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


def dis_reg(y_outputs, fix_marg):
    y_outputs_sum = tf.reduce_sum(y_outputs, axis=1, keep_dims=True)
    multi = tf.tile(y_outputs_sum, multiples=[1, fix_marg])
    return y_outputs / multi


def distribution_loss(y_outputs, y, th):
    # y_outputs = self.dis_reg(y_outputs, fix_marg)
    jsd = JSD(y_outputs, y)
    return r_kurtosis(y_outputs, th), jsd


def cross_distribution_loss(y_outputs, y):
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_outputs, labels=y)
    return tf.reduce_mean(entropy)


def style_loss(y_outputs, y):
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_outputs, labels=y)
    return tf.reduce_mean(entropy)


def correlation_tensor(w1, w2):
    mean1 = np.mean(w1)
    mean2 = np.mean(w2)
    std1 = np.std(w1, axis=0)
    std2 = np.std(w2, axis=0)
    return np.mean((w1 - mean1) * (w2 - mean2)) / (std1 * std2)


def min_max_normalization(x):
    minn = np.min(x)
    maxx = np.max(x)
    return (x - minn) / (maxx - minn)


def print_task_correlation(W, t1, t2):
    cor_matrix = np.zeros(shape=(t2, t1))
    for i in range(t2):
        for j in range(t1):
            cor_matrix[i][j] = correlation_tensor(W[t1+i], W[j])
    # print("correlation between subtasks=", cor_matrix)
    return cor_matrix


def get_W():
    w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='W')
    w = [w[i] for i in range(0, len(w), 2)]
    w = tf.concat(w, axis=1)
    return w


def ini_omega(task_num):
    with tf.variable_scope('Omega'):
        o = tf.Variable(tf.eye(task_num, dtype=tf.float32) / tf.cast(task_num, tf.float32),
                        dtype=tf.float32, name='omega', trainable=True)


def tr(W, o):
    result = tf.linalg.trace(tf.matmul(tf.matmul(W, tf.matrix_inverse(o)), tf.transpose(W)))
    return result


def update_omega(W):
    A = tf.matmul(tf.transpose(W), W)
    eigval, eigvec = tf.self_adjoint_eig(A)
    eigval = tf.matrix_diag(tf.sqrt(eigval))
    A_sqrt = tf.matmul(tf.matmul(tf.matrix_inverse(eigvec), eigval), eigvec)
    return tf.divide(A_sqrt, tf.linalg.trace(A_sqrt))


def scalar_for_weights(grad, var, omega, taskid):
    if var.name[0] == 'W':
        if var.name == 'W/fully_connected/weights:0' or var.name == 'W/fully_connected/biases:0':
            grad = tf.multiply(grad, omega[taskid][0])
        else:
            s = int(var.name[18])
            grad = tf.multiply(grad, omega[taskid][s])
    return grad


def propagate_ROC(output, threshold):
    Y_predict = np.zeros(output.shape)
    for i in range(output.shape[0]):
        if output[i] >= threshold:
            Y_predict[i] = 1
        else:
            Y_predict[i] = 0
    return Y_predict


def get_uninitialized_variables(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    print([str(i.name) for i in not_initialized_vars])
    return not_initialized_vars


def get_cross_val_loss_transfer(sess, dataset, dis_loss, x, y):
    th_end = 0
    cross_val_loss_transfer = 0.0
    while th_end == 0:
        th_x_b, th_y_b, th_end = dataset.load_next_batch_quicker("Th")
        th_y_b[:, 0: 10] = fixprob(th_y_b[:, 0: 10])
        cross_val_loss_ = sess.run(dis_loss, feed_dict={x: th_x_b, y: th_y_b})
        cross_val_loss_transfer += cross_val_loss_
    cross_val_loss_transfer /= dataset.th_batch_index_max
    return cross_val_loss_transfer


def get_all_train_accuracy(sess, y_outputs, x):
    dataset = AVAImages()
    dataset.read_batch_cfg(task="TrainBatch")
    end = 0
    correct_count = 0.0
    train_size = 0
    while end == 0:
        x_b, y_b, end = dataset.load_next_batch_quicker("train")
        y_outputs_ = sess.run(y_outputs, feed_dict={x: x_b})
        y_pred_ = np.argmax(y_outputs_[:, 0: 10], axis=1)
        y_pred_ = np.int64(y_pred_ >= 5)
        y_b_ = np.int64(np.argmax(y_b[:, 0: 10], axis=1) >= 5)
        correct_count += sum((y_pred_ - y_b_) == 0)
        train_size += x_b.shape[0]
    return correct_count / train_size


def get_all_test_accuracy(sess, y_outputs, dataset, x):
    end = 0
    correct_count = 0.0
    test_size = 0
    while end == 0:
        x_b, y_b, end = dataset.load_next_batch_quicker("test")
        y_outputs_ = sess.run(y_outputs, feed_dict={x: x_b})
        y_pred_ = np.argmax(y_outputs_[:, 0: 10], axis=1)
        y_pred_ = np.int64(y_pred_ >= 5)
        y_b_ = np.int64(np.argmax(y_b[:, 0: 10], axis=1) >= 5)
        correct_count += sum((y_pred_ - y_b_) == 0)
        test_size += x_b.shape[0]
    return correct_count / test_size