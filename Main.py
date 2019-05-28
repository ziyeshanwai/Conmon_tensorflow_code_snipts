# -*- coding: UTF-8 -*-
# Author:liyouwang
# Date: 2018.7.9

import numpy as np
import glob
import tensorflow as tf
from random import shuffle
import Model
from skimage import exposure
import tensorflow.contrib.slim as slim


def shuffle_data(data, batch_size=128):
    mean_Value = 113.48202195680676
    data_path = data
    shuffle(data_path)
    next_train_data = data_path[:batch_size]
    img_X = []
    cor_Y = []
    for pt in next_train_data:
        img = np.load(pt).item().get('X')
        cor_y = np.load(pt).item().get('Y').reshape(-1)*1000
        # for k in range(4):
        #     img_aug_com = iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        #                              rotate=(-15, 15), scale=(0.8, 1.2), mode='edge')
        #     img_aug = img_aug_com.augment_image(img)
        #     img_aug = img_aug[:, :, np.newaxis]
        #     img_X.append((img_aug.astype(np.float32)-mean_Value)/255)
        #     cor_Y.append(cor_y)
        #     img_aug_com = iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, mode='constant')
        #     img_aug = img_aug_com.augment_image(img)
        #     img_aug = img_aug[:, :, np.newaxis]
        #     img_X.append((img_aug.astype(np.float32) - mean_Value) / 255)
        #     cor_Y.append(cor_y)
        #     img_aug_com = iaa.Affine(rotate=(-15, 15), mode='constant')
        #     img_aug = img_aug_com.augment_image(img)
        #     img_aug = img_aug[:, :, np.newaxis]
        #     img_X.append((img_aug.astype(np.float32) - mean_Value) / 255)
        #     cor_Y.append(cor_y)
        #     img_aug_com = iaa.Affine(scale=(0.8, 1.2), mode='constant')
        #     img_aug = img_aug_com.augment_image(img)
        #     img_aug = img_aug[:, :, np.newaxis]
        #     img_X.append((img_aug.astype(np.float32) - mean_Value) / 255)
        #     cor_Y.append(cor_y)
        # img_norm = cv2.equalizeHist(img)
        # img_norm = img_norm[:, :, np.newaxis]
        # img_X.append((img.astype(np.float32)-mean_Value)/255)
        # cor_Y.append(cor_y)
        # img_constrast = image_contrast(img, np.random.randint(low=20, high=150, size=(1,))[0]/100,
        #                                np.random.randint(low=-20, high=20, size=(1,))[0])
        # img_constrast = img_constrast[:, :, np.newaxis]
        # img_X.append((img_constrast.astype(np.float32) - mean_Value) / 255)
        # cor_Y.append(cor_y)
        # for con_c in range(6):
        #     img_log_con = exposure.adjust_gamma(img, gamma=np.random.randint(low=10, high=300, size=(1,))[0]/100)
        #     img_log_con = img_log_con[:, :, np.newaxis]
        #     img_X.append((img_log_con.astype(np.float32) - mean_Value) / 255)
        #     cor_Y.append(cor_y)
        img = img[:, :, np.newaxis]
        img_X.append((img.astype(np.float32) - mean_Value) / 255)
        cor_Y.append(cor_y)
    c = list(zip(img_X, cor_Y))
    shuffle(c)
    x, y = zip(*c)

    data_x = np.array(x)
    data_y = np.array(y)

    return data_x, data_y


def add_variable_summary(variable):

    return tf.summary.histogram('train/'+variable.name, variable)


if __name__ == '__main__':
    train_data_path = glob.glob('\\\\192.168.20.63\\ai\\face_data\\20190419\\Data_DeepLearning\\train\\*.npy')
    # test_data_path = glob.glob('../Face_train_data/person1/test/*.npy')
    X = tf.placeholder(shape=[None, 258, 386, 1], dtype=tf.float32, name='input_x')
    Y = tf.placeholder(shape=[None, 10090*3], dtype=tf.float32, name='input_y')
    # w_ini = np.load('./vers/pca_transform_weights.npy')
    # b_ini = np.load('./vers/pca_mean_bias.npy')
    Model_path = "\\\\192.168.20.63\\ai\\face_data\\20190419\\Data_DeepLearning\\net_model\\"
    tensor_board_path = "\\\\192.168.20.63\\ai\\face_data\\20190419\\Data_DeepLearning\\tensorboard"
    model = Model.Face_net(X)
    output = model.face_net()
    regu_loss = tf.reduce_mean(slim.losses.get_regularization_losses())
    mse_loss = tf.reduce_mean(tf.squared_difference(output, Y))
    loss = mse_loss + regu_loss
    tf.summary.scalar('train/loss', loss)
    lr = tf.Variable(0.001, dtype=tf.float32, name='lr', trainable=False)
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    ite = 10000
    for var in tf.trainable_variables():
        add_variable_summary(var)
    merged = tf.summary.merge_all()
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(tensor_board_path, sess.graph)
        saver = tf.train.Saver()
        saver.restore(sess, Model_path+"net_model-20000")
        while ite <= 20000:
            if ite == 10000:
                sess.run(tf.assign(lr, 0.0001))
            x, y = shuffle_data(train_data_path, batch_size=128)
            sess.run(train_step, feed_dict={X: x, Y: y})
            if ite % 100 == 0:
                print("iteration:{}, loss:{}".format(ite, sess.run(loss, feed_dict={X: x, Y: y})))
                print('regulization loss is {}'.format(sess.run(regu_loss, feed_dict={X: x, Y: y})))
                print('MSE loss is {}'.format(sess.run(mse_loss, feed_dict={X: x, Y: y})))
                rs = sess.run(merged, feed_dict={X: x, Y: y})
                writer.add_summary(rs, float(ite))
                # test_x, test_y = shuffle_data(test_data_path, batch_size=32)
                # print('test mse_loss is {}'.format(sess.run(mse_loss, feed_dict={X: test_x, Y: test_y})))
                # result = sess.run(output, feed_dict={X: test_x})
                # print('the different predict output is {}'.format(np.mean(result[0, :] - result[1, :])))
                # grads_and_var = tf.train.AdamOptimizer().compute_gradients(loss)
                # for gv in grads_and_var:
                #     print('{}:{}'.format(gv[1].name, sess.run(gv[0], feed_dict={X: x, Y: y})))
            ite += 1
            if ite % 1000 == 0:
                saver.save(sess, Model_path+"net_model", global_step=ite)



