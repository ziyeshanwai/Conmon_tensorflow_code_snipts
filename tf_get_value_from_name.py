import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class Face_net(object):
    def __init__(self, inputs):
        self.inputs = inputs

    def face_net(self):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            ):
            net = slim.conv2d(self.inputs, num_outputs=32, kernel_size=[3, 3], stride=[2, 2],
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              padding='SAME', activation_fn=tf.nn.relu, scope='conv1a')  # 193 * 129
            net = slim.conv2d(net, num_outputs=32, kernel_size=[3, 3], stride=[1, 1],
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              padding='SAME', activation_fn=tf.nn.relu, scope='conv1b')   # 193 * 129
            net = slim.conv2d(net, num_outputs=48, kernel_size=[3, 3], stride=[2, 2],
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              padding='SAME', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.001),
                              scope='conv2a')
            net = slim.conv2d(net, num_outputs=48, kernel_size=[3, 3], stride=[1, 1],
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              padding='SAME', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.001),
                              scope='conv2b')  # 97 * 65
            net = slim.conv2d(net, num_outputs=72, kernel_size=[3, 3], stride=[2, 2],
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              padding='SAME', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.001),
                              scope='conv3a')
            net = slim.conv2d(net, num_outputs=72, kernel_size=[3, 3], stride=[1, 1],
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              padding='SAME', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.001),
                              scope='conv3b')  # 49 * 33
            net = slim.conv2d(net, num_outputs=108, kernel_size=[3, 3], stride=[2, 2],
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              padding='SAME', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.001),
                              scope='conv4a')
            net = slim.conv2d(net, num_outputs=108, kernel_size=[3, 3], stride=[1, 1],
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              padding='SAME', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.001),
                              scope='conv4b')  # 25 * 17
            # net = tf.py_func(save_net_to_npy, [net], tf.float32)
            # net.set_shape([None, 7, 7, 108])
            net = slim.conv2d(net, num_outputs=162, kernel_size=[3, 3], stride=[2, 2],
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              padding='SAME', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.001),
                              scope='conv5a')
            net = slim.conv2d(net, num_outputs=162, kernel_size=[3, 3], stride=[1, 1],
                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                              padding='SAME', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.001),
                              scope='conv5b')  # 13 * 9
            net = tf.reshape(net, [-1, 13 * 9 * 162])
            net = slim.fully_connected(net, 256, weights_initializer=tf.contrib.layers.xavier_initializer(),
                                       activation_fn=tf.nn.relu, scope='fc1')
            net = slim.fully_connected(net, 10090*3, weights_initializer=tf.contrib.layers.xavier_initializer(),
                                       activation_fn=None, scope='fc2')
            return net


if __name__ == "__main__":
    model_input = tf.placeholder(name="input", shape=[None, 258, 386, 1], dtype=tf.float32)
    model_ouput = Face_net(model_input).face_net()
    print("start....")
    # for var in tf.trainable_variables():
    #     print(var)  # 这里可以得到所有参与梯度下降的变量

    for var in tf.trainable_variables():
        print(var)  # 这里可以得到所有参与梯度下降的变量
        # print(var.name)  # conv4b/biases:0
        # print(var.op.name)  # conv4b/biases
        # print(var.shape)
    print("___________" * 6)
    varibale_name_to_fix = "conv4b/weights:0"  # (3, 3, 108, 108)
    va = np.random.rand(3, 3, 108, 108)
    print_va = None
    """给指定变量赋值"""
    for var in tf.trainable_variables():
        if var.name == varibale_name_to_fix:
            print(var.name)
            print_va = var
            assign_op = tf.assign(var, va)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())  # 必须先初始化
                print(sess.run(var))
                sess.run(assign_op)  # 这是值已经在内存中了
                print("___________" * 6)
                print(sess.run(var))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # 必须先初始化 新开一个sess值就不是上面的了 一个图只能有一个sess,会话关闭内存就释放了
        print("经过修改后的变量值是{}".format(sess.run(print_va)))

    """method 2 """
    print("method 2")
    graph = tf.get_default_graph()
    con4b = graph.get_tensor_by_name(varibale_name_to_fix)
    print(con4b)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # 必须先初始化
        print(sess.run(con4b))
        print("___________" * 6)
        asign = tf.assign(con4b, va)
        sess.run(asign)
        print(sess.run(con4b))

