import tensorflow as tf
from Util.util import *
import numpy as np

"""
使用偏移量模型
"""
if __name__ == "__main__":
    number_blendshapes = 160
    blendshape_path = "./Blendshapes"
    mesh_path = "\\\\192.168.20.63\\ai\\face_data\\20190419\\Data_DeepLearning\\test_output"
    out_path = "./output"
    v0, f0 = loadObj(os.path.join(blendshape_path, "0.obj"))
    V = np.array(v0, dtype=np.float32).flatten()
    for i in range(1, number_blendshapes):
        v, f = loadObj(os.path.join(blendshape_path, "{}.obj".format(i)))
        v = np.array(v, dtype=np.float32).flatten()
        V = np.vstack((V, v))
    mean_mesh = np.mean(V, axis=0)
    # mean_mesh = mean_mesh.reshape(-1, 3)
    # writeObj(os.path.join(blendshape_path, "base_mesh.obj"), mean_mesh, f0)
    delta_V = V - mean_mesh
    mean_mesh_tf = tf.constant(value=mean_mesh, name="mean_mesh")

    delta_v_tf = tf.constant(value=delta_V, name="blend_shapes")
    print(delta_v_tf)
    ini_coes = np.random.rand(1, number_blendshapes).astype(np.float32)
    coes = tf.get_variable(initializer=ini_coes, name="coes", trainable=True)
    print(coes)
    pre_mesh = tf.matmul(coes, delta_v_tf) + mean_mesh_tf
    true_mesh, _ = loadObj(os.path.join(mesh_path, "6470.obj"))
    true_mesh = np.array(true_mesh, dtype=np.float32).reshape(1, -1)
    loss = tf.reduce_mean(tf.squared_difference(1000*pre_mesh, 1000*true_mesh))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        last_loss = 100
        count = 0
        while True:
            sess.run(optimizer)
            # print(sess.run(coes))
            diff = sess.run(loss)
            if np.abs(diff - last_loss) < 1e-5:
                print("ite is over in {} times".format(count))
                coe = sess.run(coes)
                print("coe is {}".format(coe))
                calculated_mesh = coe.dot(delta_V) + mean_mesh
                calculated_mesh = calculated_mesh.reshape(-1, 3)
                writeObj(os.path.join(out_path, "5.obj"), calculated_mesh, f0)
                # pre_mesh = sess.run()
                break
            else:
                last_loss = diff
                count += 1
                print("loss:{}".format(diff))



