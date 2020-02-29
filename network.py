from resnet import resnet_v2_50
from data import AVAImages
from progressbar import ProgressBar
import tensorflow as tf
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

    def comparator(self, inputs_1, inputs_2):
        block1 = resnet_v2_50(inputs=inputs_1, num_classes=50, reuse=True, scope='res-50')
        block2 = resnet_v2_50(inputs=inputs_2, num_classes=50, reuse=True, scope='res-50')
        concat = tf.concat([block1, block2], axis=1)
        fc1 = slim.fully_connected(concat, 20)
        fc2 = slim.fully_connected(fc1, 1)
        return fc2

    def tags_net(self, inputs):
        block1, _ = resnet_v2_50(inputs=inputs, num_classes=16)
        fc1 = slim.fully_connected(block1, 8, activation_fn=tf.nn.relu)
        output = slim.fully_connected(fc1, 132, activation_fn=tf.nn.softmax)
        return output

    def score_net(self, inputs):
        block1, _ = resnet_v2_50(inputs=inputs, num_classes=128)
        with tf.variable_scope('trainable'):
            fc1 = slim.fully_connected(block1, 64, activation_fn=tf.nn.relu)
            output = slim.fully_connected(fc1, 1)
        return output

    def propagate(self, inputs):
        if self.net == "predict_tags":
            x = inputs
            return self.tags_net(x)
        elif self.net == "predict_score":
            x = inputs
            return self.score_net(x)
        elif self.net == "comparator":
            x1, x2 = inputs
            return self.comparator(x1, x2)

    def validation_loss(self, sess, y_outputs, x, y, dataset):
        if self.net == "predict_tags":
            x_val, y_val = dataset.val_set_x, dataset.val_set_y_tag
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_outputs, labels=tf.argmax(y, 1))
            loss = tf.reduce_mean(entropy)
        elif self.net == "predict_score":
            x_val, y_val = dataset.val_set_x, dataset.val_set_y_score
            loss = tf.losses.mean_squared_error(y, y_outputs)
        else:
            x_val, y_val = 0, 0
            loss = 0
        return sess.run(loss, feed_dict={x: x_val, y: y_val})

    def train_AB(self, parameter_list: tuple, dataset=AVAImages("tag"), model_save_path='./model/'):
        dataset.load_data()
        batch_size, learning_rate, learning_rate_decay, epoch = parameter_list
        w, h, c = self.input_size
        x = tf.placeholder(tf.float32, [None, w, h, c])
        y = tf.placeholder(tf.float32, [None, self.output_size])
        y_outputs = self.propagate(x)
        global_step = tf.Variable(0, trainable=False)

        entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_outputs, labels=tf.argmax(y, 1))
        loss = tf.reduce_mean(entropy)
        rate = tf.train.exponential_decay(learning_rate, global_step, 200, learning_rate_decay)  # 指数衰减学习率
        train_op = tf.train.AdamOptimizer(rate).minimize(loss, global_step=global_step)

        # with saver&sess
        # tf_vars = tf.trainable_variables(scope="resnet_v2_50")
        # saver = tf.train.Saver(var_list=tf_vars)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                # progress = ProgressBar(dataset.train_set_x.shape[0] // batch_size)
                # progress.start()
                # bn = 0
                while True:
                    # 遍历所有batch
                    x_b, y_b, end = dataset.load_next_batch(batch_size)
                    train_op_, loss_, step = sess.run([train_op, loss, global_step], feed_dict={x: x_b, y: y_b})
                    if step % 1 == 0:
                        print("training step {0}, loss {1}, validation loss {2}"
                              .format(step, loss_, self.validation_loss(sess, y_outputs, x, y, dataset)))
                        saver.save(sess, model_save_path + 'my_model', global_step=global_step)
                    # bn += 1
                    # progress.show_progress(bn)
                    if end == 1:
                        break
                # progress.end()

    def get_uninitialized_variables(self, sess):
        global_vars = tf.global_variables()
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        print([str(i.name) for i in not_initialized_vars])
        return not_initialized_vars

    def train_UA_C(self, parameter_list: tuple, dataset=AVAImages("score"),
                   model_read_path='./model/', model_save_path='./model2/'):
        dataset.load_data()
        # 重置默认图 防止出现意外错误
        tf.reset_default_graph()  # 重置默认图。
        # parameter list
        batch_size, learning_rate, learning_rate_decay, epoch = parameter_list

        # network
        w, h, c = self.input_size
        x = tf.placeholder(tf.float32, [None, w, h, c])
        y = tf.placeholder(tf.float32, [None, self.output_size])
        y_outputs = self.propagate(x)
        global_step = tf.Variable(0, trainable=False)
        # Tensorflow中集成的函数
        mse = tf.losses.mean_squared_error(y, y_outputs)
        # 利用Tensorflow基础函数手工实现
        # mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # get variables
        t_vars = tf.trainable_variables()  # 获取所有的变量
        g_vars = [var for var in t_vars if 'trainable' in var.name]  # 附加的finetune网络层（需要训练的层）
        var_list = [var for var in t_vars if 'resnet' in var.name]  # 不需要改变的网络层
        rate = tf.train.exponential_decay(learning_rate, global_step, 200, learning_rate_decay)  # 指数衰减学习率
        train_op = tf.train.AdamOptimizer(rate).minimize(mse, var_list=g_vars, global_step=global_step)

        # saver&re_saver
        variables_to_restore = slim.get_variables_to_restore(include=['resnet_v2_50'])
        # 单引号指只恢复一个层。双引号会恢复含该内容的所有层。
        re_saver = tf.train.Saver(variables_to_restore)  # 建立一个saver 从已有的模型中恢复res50系列参数到网络中.
        saver = tf.train.Saver()  # 建立一个模型，训练的时候保存整个模型的ckpt

        # with saver&sess
        with tf.Session() as sess:
            # model_path = './model.ckpt'  # 后缀名称仅需要写ckpt即可,后面的00001-00000不必添加
            re_saver.restore(sess=sess, save_path=model_read_path+'my_model-1')  # 恢复模型的参数到新的模型
            un_init = tf.variables_initializer(self.get_uninitialized_variables(sess))  # 获取没有初始化(通过已有model加载)的变量
            # print info of uninitialized variables
            sess.run(un_init)  # 对没有初始化的变量进行初始化并训练.
            for i in range(epoch):
                while True:
                    # 遍历所有batch
                    x_b, y_b, end = dataset.load_next_batch(batch_size)
                    train_op_, loss_, step = sess.run([train_op, mse, global_step], feed_dict={x: x_b, y: y_b})
                    if dataset.batch_index % 5 == 0:
                        print("training step {0}, loss {1}, validation loss {2}"
                              .format(step, loss_, self.validation_loss(sess, y_outputs, x, y, dataset)))
                        saver.save(sess, model_save_path + 'my_model', global_step=global_step)
                    if end == 1:
                        break

    def restore_net(self):
        """
        不需要提前定义好网络结构。因为.meta文件已经存好以前的网络结构了。
        :return:
        """
        # this will create the graph/network for you
        # but we still need to load the value of the parameters that we had trained on this graph.
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('model/my_model-1.meta')
            # saver.restore(sess, tf.train.latest_checkpoint('model'))
            saver.restore(sess, 'model/my_model-1')
            # print(sess.run('fully_connected_1/Softmax:0'))  # error

