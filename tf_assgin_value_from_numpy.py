import tensorflow as tf
import numpy as np


if __name__ == "__main__":
    """method 1
       使用numpy的数组的值作为初始化
       这种方法需要先已知numpy数组值后定义tf变量
    """
    print("method1")
    a = np.random.rand(4, 5, 6)
    print(a)
    with tf.variable_scope("test"):
        tf_a = tf.get_variable(name="a", initializer=a)
        print("tf_a:{}".format(tf_a))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(tf_a))

    """method2
       通过tf.assign 这种方法对变量定义顺序么有要求，可以先定义tf变量
    """
    print("method2")
    b = np.random.rand(4, 5, 6)
    print(b)
    with tf.variable_scope("test"):
        tf_b = tf.get_variable(name="b", shape=[4, 5, 6], initializer=tf.truncated_normal_initializer)
    assing_b = tf.assign(tf_b, b)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(tf_b))
        sess.run(assing_b)
        print("--" * 10)
        print(sess.run(tf_b))

    """
    method3 tf.convert_to_tensor() 但是这种方法是常量
    """
    print("method3")
    c = np.random.rand(4, 5, 6)
    print(c)
    tf_c = tf.convert_to_tensor(c)
    print(tf_c)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(tf_c))
