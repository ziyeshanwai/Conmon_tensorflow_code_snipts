import numpy as np
import cv2
from random import shuffle
from Util.util import loadObj, writeObj
import os


if __name__ == '__main__':

    image_path = '\\\\192.168.20.63\\ai\\face_data\\20190419\\image\\048170110027'
    obj_path = "\\\\192.168.20.63\\ai\\face_data\\20190419\\smooth"
    train_data_path = "\\\\192.168.20.63\\ai\\face_data\\20190419\\Data_DeepLearning\\train"
    test_data_path = "\\\\192.168.20.63\\ai\\face_data\\20190419\\Data_DeepLearning\\test"
    train_data = {}
    mean_value = 0
    count = 0
    for i in range(0, 1000):
        if os.path.exists(os.path.join(obj_path, "smooth-{}.obj".format(i))):
            ver, face = loadObj(os.path.join(obj_path, "smooth-{}.obj".format(i)))
            ver = np.array(ver, dtype=np.float32)
            img_file = os.path.join(image_path, "{}.jpg".format(i))
            if count % 200 == 0:
                print("process {} data..".format(i))
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.T
            img = cv2.resize(img, dsize=(258, 386))
            # cv2.namedWindow('test')
            # cv2.imshow('test', img)
            # cv2.waitKey(0)
            train_data['X'] = img
            train_data['Y'] = ver
            mean_value += np.mean(img)
            count += 1
            np.save(os.path.join(train_data_path, "{}.npy".format(i)), train_data)

    print('the train data mean value is {}'.format(mean_value/count))
    np.save(os.path.join(train_data_path, "mean_value.npy"), mean_value/count)