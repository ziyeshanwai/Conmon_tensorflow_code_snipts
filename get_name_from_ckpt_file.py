import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python import pywrap_tensorflow
import os


if __name__ == "__main__":

    """method 1"""
    chkp_path = "\\\\192.168.20.63\\ai\\face_data\\20190419\\Data_DeepLearning\\net_model\\net_model-5000"
    #chkp.print_tensors_in_checkpoint_file(chkp_path, tensor_name='', all_tensors=True)
    chkp.print_tensors_in_checkpoint_file(chkp_path, tensor_name='fc1/weights', all_tensors=False)

    """method 2"""
    reader = pywrap_tensorflow.NewCheckpointReader(chkp_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        print(reader.get_tensor(key))  # Remove this is you want to print only variable names
        print(reader.get_tensor(key).shape)

