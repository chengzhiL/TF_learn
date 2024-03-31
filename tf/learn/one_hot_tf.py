import tensorflow as tf
import numpy as np

def ont_hot_test():
    labels = tf.constant([0,1,2,3],dtype=tf.int32)
    depth = 3
    one_hot = tf.one_hot(labels, depth)
    print(one_hot)

if __name__ == '__main__':
    ont_hot_test()