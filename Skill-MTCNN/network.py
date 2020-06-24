from flyai.train_helper import upload_data, download, sava_train_model  # 因为要蹭flyai的gpu
from dataset import AVAImages
from dataset_utils import dis2mean, index2score_dis, load_data
from network_utils import MTCNN_v2, JSD, distribution_loss, propagate_ROC, fixprob, tf_fixprob, read_cfg, get_W, ini_omega, \
    tr, r_kurtosis, style_loss, scalar_for_weights, update_omega, print_task_correlation, min_max_normalization
import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from operator import itemgetter
import pickle
slim = tf.contrib.slim


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

    # ############################################PREDICTION######################################
    def predict_distribution_1img_draw(self, model_path='./model_MTCNN/', image_path='3.jpg'):
        img = cv2.imread(image_path)
        img = cv2.resize(img * 255, (227, 227), interpolation=cv2.INTER_CUBIC)
        img = img[np.newaxis, :]
        w, h, c = self.input_size
        x = tf.placeholder(tf.float32, [None, w, h, c])
        y_outputs = MTCNN_v2(intputs=x, outputs=self.output_size, training=True)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                y_outputs = sess.run(y_outputs, feed_dict={x: img})
                y_outputs_mean = dis2mean(y_outputs[:, 0: 10])
                plt.title('predict distribution')
                plt.xlabel('vote')
                plt.ylabel('number')
                plt.plot(np.array(range(1, 11)), y_outputs[0], color="orange",
                         linewidth=1, linestyle=':', label='predict' + str(y_outputs_mean) , marker='o')
                plt.legend(loc=2)  # 图例展示位置，数字代表第几象限
                plt.show()  # 显示图像
            else:
                print('No checkpoing file found')
                return

    def predict_score_Nimg(self, model_path='./model_MTCNN/'):
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
        y_outputs = MTCNN_v2(intputs=x, outputs=self.output_size, training=True)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                y_outputs = sess.run(y_outputs, feed_dict={x: img_lists})
                y_outputs_mean = dis2mean(y_outputs[:, 0: 10])
                for i in range(y_outputs_mean.shape[0]):
                    if y_outputs_mean[i] < 5:
                        print("index={index}, score={score}".format(index=lines[i][:-1], score=y_outputs_mean[i]))
            else:
                print('No checkpoing file found')
                return

    def predict_TPTNFPFN_show_imgs(self, model_path='./model_MTCNN/', flag="TP", th=5.5):
        """
        统计skill-mtcnn预测为TPTNFPFN的 skill使用情况
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

        test_set_x, test_set_y = load_data(flag="test")
        score_label = dis2mean(test_set_y[:, 0: 10])
        score_label = np.int64(score_label >= th)

        w, h, c = self.input_size
        x = tf.placeholder(tf.float32, [None, w, h, c])
        y_outputs = MTCNN_v2(intputs=x, outputs=self.output_size, training=True)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                y_predict = sess.run(y_outputs, feed_dict={x: test_set_x})
                score_predict = dis2mean(y_predict[:, 0: 10])
                score_predict = np.int64(score_predict >= th)

                for i in range(test_set_x.shape[0]):
                    if flag == "TP":
                        if score_predict[i] - score_label[i] == 0 and score_label[i] == 1:
                            ski_index = np.nonzero(np.array(list(test_set_y[i, 10:])))[0]
                            count_skills[ski_index] += 1
                            keys = list(ski_index)
                            if sum(test_set_y[i, 10:]) == 0:
                                count_skills[14] += 1
                                print("score={score}, skill={skill}".format(score=score_predict[i], skill="no skill"))
                            else:
                                print("score={score}, skill={skill}".format(score=score_predict[i], skill=itemgetter(*keys)(skills)))
                            # os.system("cp AVA_dataset/images/{index}.jpg select".format(index=url[0:-1]))
                    i += 1
                print(count_skills)
            else:
                print('No checkpoing file found')
                return

    def predict_score_ALLTESTimg(self, model_path='./model_MTCNN'):
        test_set_x, test_set_y = load_data(flag="test")
        X, Y = test_set_x, dis2mean(test_set_y[:, 0:10])
        Y_sort = np.sort(Y)

        # load weights
        w, h, c = self.input_size
        x = tf.placeholder(tf.float32, [None, w, h, c])
        y_outputs = MTCNN_v2(intputs=x, outputs=self.output_size, training=True)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                y_predict = sess.run(y_outputs, feed_dict={x: X})
                y_outputs_mean = dis2mean(y_predict[:, 0: 10])
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

    def draw_and_save_ROC(self, model_path='./model_MTCNN'):
        # load data
        dataset = AVAImages()
        # dataset.read_data(flag="test")
        # X, Y = dataset.test_set_x, np.int64(dataset.dis2mean(dataset.test_set_y[:, 0:10]) >= 5)
        with open('x.pkl', 'rb') as f:
            x = pickle.load(f)
        with open('y.pkl', 'rb') as f:
            y = pickle.load(f)
        X, Y = x, np.int64(dataset.dis2mean(y[:, 0:10]) >= 5)

        # load weights
        w, h, c = self.input_size
        x = tf.placeholder(tf.float32, [None, w, h, c])
        y_outputs = MTCNN_v2(intputs=x, outputs=self.output_size, training=True)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                y_predict = sess.run(y_outputs, feed_dict={x: X})
                y_outputs_mean = dis2mean(y_predict[:, 0: 10])
            else:
                print('No checkpoing file found')
                return

        threshold = np.sort(y_outputs_mean)
        recall = np.zeros(y_outputs_mean.shape)
        FAR = np.zeros(y_outputs_mean.shape)

        for k in range(threshold.shape[0]):
            Y_predict = propagate_ROC(y_outputs_mean, threshold[k])  # (m_test,)
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

    # #######################################################TRAIN######################################################
    def train_score_CNN(self, data='dataset/', model_save_path='./model_score_CNN/', th_score=5.5):
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
        dataset.load_data()
        dataset.val_set_y[:, 0: 10] = fixprob(dataset.val_set_y[:, 0: 10])
        y_test = dataset.dis2mean(dataset.test_set_y[:, 0: 10])
        y_test = np.int64(y_test >= th_score)  # 前提test_set_y.shape=(num,)
        dataset.Th_y[:, 0: 10] = fixprob(dataset.Th_y[:, 0: 10])
        dataset.read_batch_cfg()
        learning_rate, learning_rate_decay, epoch, alpha, beta, gamma, theta = read_cfg()

        w, h, c = self.input_size
        with tf.name_scope("Inputs"):
            x = tf.placeholder(tf.float32, [None, w, h, c])
            y = tf.placeholder(tf.float32, [None, self.output_size])
            th = tf.placeholder(tf.float32)
        y_outputs = MTCNN_v2(intputs=x, outputs=self.output_size, training=True)
        y_outputs = tf_fixprob(y_outputs[:, 0: 10])
        global_step = tf.Variable(0, trainable=False)
        upgrade_global_step = tf.assign(global_step, tf.add(global_step, 1))

        with tf.name_scope("Loss"):
            r_kus = r_kurtosis(y_outputs[:, 0: 10], th)
            dis_loss = JSD(y_outputs[:, 0: 10], y[:, 0: 10])
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
                    y_b[:, 0: 10] = fixprob(y_b[:, 0: 10])
                    step = sess.run(global_step)
                    cross_val_loss_transfer = sess.run(dis_loss, feed_dict={x: dataset.Th_x, y: dataset.Th_y})
                    train_op_, loss_ = sess.run([train_op_all, loss], feed_dict={x: x_b, y: y_b, th: cross_val_loss_transfer})

                    val_loss = sess.run(loss, feed_dict={x: dataset.val_set_x, y: dataset.val_set_y,
                                                         th: cross_val_loss_transfer})
                    print("epoch {3} batch {4}/{0} loss {1}, validation loss {2}".
                          format(dataset.batch_index_max, loss_, val_loss, i + 1, dataset.batch_index))

                    if val_loss < best_val_loss * improvement_threshold:
                        best_val_loss = val_loss
                        saver.save(sess, model_save_path + 'my_model')

                        ### test acc
                        y_outputs_ = sess.run(y_outputs, feed_dict={x: dataset.test_set_x})
                        y_outputs_ = dis2mean(y_outputs_[:, 0: 10])
                        y_pred = np.int64(y_outputs_ >= th_score)
                        test_acc = sum((y_pred - y_test) == 0) / dataset.test_set_x.shape[0]
                        print("    test acc {acc} with best acc {best} in epoch{e}/batch{b}".
                              format(acc=test_acc, best=best_test_acc, e=best_test_acc_epoch, b=best_test_acc_batch))
                        if test_acc > best_test_acc:
                            best_test_acc = test_acc
                            best_test_acc_epoch = i
                            best_test_acc_batch = dataset.batch_index

                    if end == 1:
                        break

            #### save
            os.system('zip -r myfile.zip ./' + model_save_path)
            sava_train_model(model_file="myfile.zip", dir_name="./file", overwrite=True)
            upload_data("myfile.zip", overwrite=True)

    def train_MTCNN_v2(self, data='dataset/', model_save_path='./model_MTCNN_v2/', th_score=5.5):
        folder = os.path.exists(model_save_path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(model_save_path)  # makedirs 创建文件时如果路径不存在会创建这个路径

        # load data
        dataset = AVAImages()
        dataset.load_data()
        dataset.val_set_y[:, 0: 10] = fixprob(dataset.val_set_y[:, 0: 10])
        y_test = dataset.dis2mean(dataset.test_set_y[:, 0: 10])
        y_test = np.int64(y_test >= th_score)  # 前提test_set_y.shape=(num,)
        dataset.Th_y[:, 0: 10] = fixprob(dataset.Th_y[:, 0: 10])

        # load parameters
        dataset.read_batch_cfg(task="Skill-MTCNN")
        learning_rate, learning_rate_decay, epoch, alpha, beta, gamma, theta = read_cfg(task="Skill-MTCNN")

        # placeholders
        w, h, c = self.input_size
        with tf.name_scope("Inputs"):
            x = tf.placeholder(tf.float32, [None, w, h, c])
            y = tf.placeholder(tf.float32, [None, self.output_size])
            th = tf.placeholder(tf.float32)
            task_id = tf.placeholder(tf.int32)
        y_outputs = MTCNN_v2(intputs=x, outputs=self.output_size, training=True)
        y_outputs = tf_fixprob(y_outputs[:, 0: 10])

        # other parameters
        global_step = tf.Variable(0, trainable=False)
        upgrade_global_step = tf.assign(global_step, tf.add(global_step, 1))

        with tf.name_scope("Loss"):
            W = get_W()
            ini_omega(self.output_size)
            omegaaa = tf.get_default_graph().get_tensor_by_name('Loss/Omega/omega:0')
            tr_W_omega_WT = tr(W, omegaaa)
            r_kus = r_kurtosis(y_outputs[:, 0: 10], th)
            dis_loss = JSD(y_outputs[:, 0: 10], y[:, 0: 10])
            loss = r_kus * (dis_loss +
                            gamma * style_loss(y_outputs[:, 10:], y[:, 10:]) +
                            tf.contrib.layers.apply_regularization(
                                regularizer=tf.contrib.layers.l2_regularizer(alpha, scope=None),
                                weights_list=tf.trainable_variables()) +
                            theta * tr_W_omega_WT)
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
            capped_gvs = [(scalar_for_weights(grad, var, omegaaa, task_id), var)
                          for grad, var in gradient_var_all]
            train_op = opt.apply_gradients(capped_gvs)
            train_op_all = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, var_list=train_theta + WW)
            train_op_omega = tf.assign(omegaaa, update_omega(W))

        saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=2)
        cross_val_loss_transfer = 0
        train_theta_and_W_first = 20
        best_val_loss = 1000
        best_val = 0
        patience = 60
        stop_flag = False
        improvement_threshold = 0.999
        best_test_acc = 0.0
        best_test_acc_epoch = 0
        best_test_acc_batch = 0
        i = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            while i <= epoch and not stop_flag:
                while True:
                    # 遍历所有batch
                    x_b, y_b, end = dataset.load_next_batch_quicker(read_dir=data)
                    y_b[:, 0: 10] = fixprob(y_b[:, 0: 10])
                    step = sess.run(global_step)
                    if step < train_theta_and_W_first:
                        cross_val_loss_transfer = sess.run(dis_loss, feed_dict={x: dataset.Th_x, y: dataset.Th_y})
                        train_op_, loss_ = sess.run([train_op_all, loss], feed_dict={x: x_b, y: y_b, th: cross_val_loss_transfer})
                    elif np.random.rand() < 0.5:
                        train_op_ = sess.run(train_op_omega)
                        sess.run(upgrade_global_step)
                    else:
                        cross_val_loss_transfer = sess.run(dis_loss, feed_dict={x: dataset.Th_x, y: dataset.Th_y})
                        for taskid in range(self.output_size):
                            train_op_, loss_ = sess.run([train_op, loss], feed_dict={x: x_b, y: y_b, th: cross_val_loss_transfer, task_id: taskid})
                        sess.run(upgrade_global_step)

                    val_loss = sess.run(loss, feed_dict={x: dataset.val_set_x, y: dataset.val_set_y,
                                                         th: cross_val_loss_transfer})
                    print("epoch {3} batch {4}/{0} loss {1}, validation loss {2}".
                          format(dataset.batch_index_max, loss_, val_loss, i + 1, dataset.batch_index))

                    if val_loss < best_val_loss * improvement_threshold:
                        best_val_loss = val_loss
                        best_val = step  # 记录最小val所对应的batch index

                        # save data
                        saver.save(sess, model_save_path + 'my_model')
                        # correlation matrix
                        # cor1
                        Wa_and_Ws = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='W'))
                        W = np.zeros(shape=(self.output_size, 4096))
                        for ii in range(W.shape[0]):
                            W[ii] = np.array(np.squeeze(Wa_and_Ws[ii * 2]))
                        cor_matrix1 = print_task_correlation(W, 10, self.output_size - 10)
                        cor_matrix1 = min_max_normalization(cor_matrix1)

                        # test acc
                        y_outputs_ = sess.run(y_outputs, feed_dict={x: dataset.test_set_x})
                        y_outputs_mean = dis2mean(y_outputs_[:, 0: 10])
                        y_pred = np.int64(y_outputs_mean >= th_score)
                        test_acc = sum((y_pred - y_test) == 0) / dataset.test_set_x.shape[0]
                        print("    test acc {acc} with best acc {best} in epoch{e}/batch{b}".format(acc=test_acc,
                                                                                                    best=best_test_acc,
                                                                                                    e=best_test_acc_epoch,
                                                                                                    b=best_test_acc_batch))
                        if test_acc > best_test_acc:
                            best_test_acc = test_acc
                            best_test_acc_epoch = i
                            best_test_acc_batch = dataset.batch_index

                    # 如果连着几个batch的val loss都没下降，则停止训练
                    if step - best_val > patience:
                        stop_flag = True
                        break
                    else:
                        print("training step {0}, loss {1}".format(step, loss_))

                    if end == 1:  # 一个epoch结束
                        break
                i += 1




            # ### save
            cv2.imwrite(model_save_path + "cor_matrix1.png", cv2.resize(cor_matrix1 * 255, (300, 420), interpolation=cv2.INTER_CUBIC))
            os.system('zip -r mtcnn_v2.zip ./' + model_save_path)
            sava_train_model(model_file="mtcnn_v2.zip", dir_name="./file", overwrite=True)
            upload_data("mtcnn_v2.zip", overwrite=True)

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

    def train_cor_matrix_predict(self, data='dataset/', model_read_path='./model_MTCNN_v2', model_save_path='./model_cor_matrix2/'):
        folder = os.path.exists(model_save_path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(model_save_path)  # makedirs 创建文件时如果路径不存在会创建这个路径

        # load data
        dataset = AVAImages()

        # load parameters
        dataset.read_batch_cfg(task="Cor_Matrix")
        learning_rate, learning_rate_decay, epoch= self.read_cfg(task="Cor_Matrix")

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
            best_loss = 1000
            patience = 4
            i = 0
            while i <= patience or i <= epoch:
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
                if loss_ <= best_loss:
                    best_loss = loss_
                    patience *= 2
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
