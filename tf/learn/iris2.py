# 导入所需要的模块
import tensorflow as tf
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

#导入数据，并随机打乱
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 随机打乱顺序
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 将打乱后的顺序分给x_train, x_test
x_train = x_data[:-30]  # from 0 to last 3o
x_test = x_data[-30:]   # from -30 to last 保证x_train 和 x_test 永远不能重叠
y_train = y_data[:-30]
y_test = y_data[-30:]

# 转换数据类型，否则报错
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
y_train = tf.cast(y_train, tf.float32)
y_test = tf.cast(y_test, tf.float32)

# 输入特征和标签值对应
 #from_tensor_slices 函数使输入特征和标签值一一对应。（把数据集分批次，每个批次batch组数据）
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32) # 分批次，每批次32个
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)


#搭建网络
w1 = tf.Variable(tf.random.truncated_normal([4,3], stddev = 0.1, seed = 1))

b1 = tf.Variable(tf.random.truncated_normal([3], stddev = 0.1, seed = 1))

lr = 0.1 # 学习率
train_loss_results = []
test_acc = []
epoch = 1
loss_all = 0

for epoch in range(epoch): #数据集级别的循环，每个epoch 循环一次数据集
    for step, (x_train,y_train) in enumerate(train_db): # batch 级别的循环，每个step循环一个batch
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train,w1) + b1
            y = tf.nn.softmax() # 获取y三个值的概率分布
            y_ = tf.one_hot(y_, depth = 3)
            loss = tf.reduce_mean(tf.square(y - y_))
            loss_all += loss.numpy()
            grads = tape.gradient(loss, [w1,b1])
            w1.assign_sub(lr * grads[0])
            b1.assign_sub(lr * grads[1])

            # 每个epoch，打印loss信息
        print("Epoch {}, loss {}".format(epoch, loss_all / 4))
        train_loss_results.append(loss_all / 4)
        loss_al = 0

#迭代优化