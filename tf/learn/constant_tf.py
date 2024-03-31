import tensorflow as tf
import numpy as np


def constant01():
    a = tf.constant([1,2,3], dtype=tf.int64)
    print(a)
    print(a.dtype)
    print(a.shape)    #result = shape=(2,)  张量是几维的，就看逗号隔开了几个数字，隔开了一个数字，就是一维。




def constant02():
    a = np.arange(0,5)
    b = tf.convert_to_tensor(a,dtype=tf.int64)
    print(a)
    print(b)

    c = np.arange(24).reshape(2,3,4)
    d = tf.convert_to_tensor(c,dtype=tf.int64)
    print(d)


def creatTen01():
    a = tf.zeros([2,3]) #一维直接写个数，二维写【i，j]，多维用[n,j,k]
    b = tf.ones(4)
    c = tf.fill([2,2],9)
    print(a)
    print("\n")
    print(b)
    print("\n")

    print(c)

def creatTen02():
    # 生成正态分布的随机数，默认均值为0，标准差为1
    a = tf.random.normal([2,2],mean=0.5,stddev=1)
    print(a)
    print("\n")
    # 生成截断式正态分布的随机数,生成随机数在[u-2𝜎,u+2𝜎])
    b = tf.random.truncated_normal([2,3],mean=0.5,stddev=1)
    print(b)
    print("\n")
    # uniform(维度，最小值，最大值)
    c = tf.random.uniform([2,2],minval=0,maxval=40,dtype=tf.int64)
    print(c)

def creatTen03():
    # 利用
    x1 = tf.constant([1,2,3], dtype=tf.int64)
    print(x1)
    # 强制转换数据类型 cast 投
    x2 = tf.cast(x1,tf.int32)
    print(x2)

    print(tf.reduce_min(x2), tf.reduce_max(x2))



if __name__ == '__main__':
    ## test02()#方法一定要定义在main函数之前。这应该和加载机制有关
    #constant02()
    creatTen02()





