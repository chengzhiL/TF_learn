# 导入所需模块
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

# 导入数据，分别为输入特征和标签
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 随机打乱数据（ 因为原始数据是顺序的，顺序不打乱会影响准确率）
# seed: 随机数种子，是一个整数，当设置后，每次生成的随机数都一样
np.random.seed(116)
np.random.shuffle(x_data)  # 随机打乱数据
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 将打乱后的数据集分割为训练集和测试集。训练集为前120行
x_train = x_data[:-30]
x_test = x_data[-30:]
y_train = y_data[:-30]
y_test = y_data[-30:]

# 转换x的数据类型，否则后后面矩阵相乘可能报错
x_train = tf.cast(x_train,tf.float32)
x_test = tf.cast(x_test,tf.float32)

# from_tensor_slices 函数使输入特征和标签值一一对应。（把数据集分批次，每个批次batch组数据）
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32) # 分批次，每批次32个
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

# 生成神经网络的参数，4个输入特征，输入层为4个输入节点；因为3分类，故输出层为3个神经元
# 用tf.Variable()标记参数可训练
# 使用seed使每次生成的随机数相同
w1 = tf.Variable(tf.random.truncated_normal([4,3], stddev = 0.1, seed = 1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev = 0.1, seed = 1))

lr = 0.1 # 学习率 = 0.1
train_loss_results = [] # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
test_acc = [] # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
epoch = 500 # 循环500轮
loss_all = 0 # 每轮分为4个step, loss_all记录四个step生成的4个loss的和

# 训练部分
for epoch in range(epoch): #数据集级别的循环，每个epoch 循环一次数据集
    for step, (x_train,y_train) in enumerate(train_db): # batch 级别的循环，每个step循环一个batch
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train,w1)+b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth = 3)
            loss = tf.reduce_mean(tf.square(y-y_))
            loss_all += loss.numpy()
        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss, [w1,b1])

        # 实现梯度更新
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])

    # 每个epoch，打印loss信息
    print("Epoch {}, loss {}".format(epoch, loss_all/4))
    train_loss_results.append(loss_all / 4)
    loss_al = 0


    # 测试部分
    # total_correct 为预测对的样本个数，total_number为测试的总样本数，将这两个变量都初始化为0
    total_correct, total_number = 0,0
    for x_test,y_test in test_db:
        y = tf.matmul(x_test,w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis = 1)
        pred = tf.cast(pred, dtype = y_test.dtype)
        # 若分类正确，则correct = 1,否则为0，将bool 类型转换为int类型
        correct = tf.cast(tf.equal(pred, y_test), dtype = y_test.dtype)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc: ", acc)
    print("---------------")

# 绘制 loss 曲线
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Loss')  # y轴变量名称
plt.plot(train_loss_results, label="$Loss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()  # 画出曲线图标
plt.show()  # 画出图像

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()
