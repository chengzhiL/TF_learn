import numpy as np
from tensorflow.keras.layers import LSTM , GlobalAveragePooling1D
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling1D,Dense, Dropout,Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

np.random.seed(42)   #随机数种子
from matplotlib import pyplot as plt
import os

import numpy as np
import pandas as pd
import tensorflow
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # RNN、激活函数、全连接层模块
from tensorflow.keras.losses import SparseCategoricalCrossentropy  # loss模块
from tensorflow.keras.models import Sequential

data = pd.read_csv(r'C:\LM paper\application\model2数字\model2number\model2number.csv',encoding='gbk',index_col=0)
# 划分特征与标签
x = data.iloc[:, data.columns != "label"]
y = data.iloc[:, data.columns == "label"]

print(type(x))



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
print(type(x_train))
x_test.to_csv(r'C:\LM paper\application\model2数字\model2number\LSTM\x_test_split.csv')
y_test.to_csv(r'C:\LM paper\application\model2数字\model2number\LSTM\y_test_split.csv')

x_train =np.array(x_train.values)
x_test =np.array(x_test.values)
y_train= np.array(y_train.values)
y_test=np.array(y_test.values)
print(type(x_train))


x_train = x_train.reshape(x_train.shape[0],10,10)
print("x.shape", x_train.shape)
x_test = x_test.reshape(x_test.shape[0],10,10)
print("x.shape", x_test.shape)

#parameters for LSTM
nb_lstm_outputs = 200  #神经元个数
nb_time_steps = 10  #时间序列长度
nb_input_vector = 10 #输入序列

#lstm
model = Sequential()
model.add(LSTM(units= nb_lstm_outputs, input_shape=(nb_time_steps, nb_input_vector),return_sequences=True))
model.add(Dropout(0.95))
# model.add(BatchNormalization())
#model.add(TimeDistributed(Dense(26, activation='softmax')))


#fc
model.add(GlobalAveragePooling1D())

#output
model.add(Dense(10, activation='softmax'))
print(model.summary())
model.summary()

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#               metrics=['sparse_categorical_accuracy'])
learning_rate = 0.004

# 编译模型，使用Adam优化器，并设置学习率
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss=SparseCategoricalCrossentropy(from_logits=False),
              optimizer=optimizer,
              metrics=["sparse_categorical_accuracy"])
"""
# Compile model
model.compile( loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
               optimizer="adam",
               metrics=["sparse_categorical_accuracy"]
             )
"""
# checkpoint
filepath="./LSTM/LSTMMODEL1.1-{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_sparse_categorical_accuracy',verbose=1,save_best_only=True,mode='max')
callbacks_list = [checkpoint]

#Fit the model
history=model.fit(x_train, y_train,
                  validation_data=(x_test, y_test),
                  epochs= 500,
                  batch_size=32,
                  callbacks=callbacks_list,
                  verbose=1)


model.save(r"./LSTM/modelLSTM1-1.hdf5")


acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

# 倒入数据至excel部分
# 需要这两个包：pip install openpyxl pip install openpyxl
# history.history.keys()
# dict_keys(['loss', 'sparse_categorical_accuracy', 'val_loss', 'val_sparse_categorical_accuracy'])
a = history.history['loss']
b = history.history['sparse_categorical_accuracy']
c = history.history['val_loss']
d = history.history['val_sparse_categorical_accuracy']
count = len(a)
# 先弄dataframe
df = pd.DataFrame(columns=['loss', 'sparse_categorical_accuracy', 'val_loss', 'val_sparse_categorical_accuracy'])
for i in range(count):
    df.loc[i] = {'loss': a[i], 'sparse_categorical_accuracy': b[i], 'val_loss': c[i],
                 'val_sparse_categorical_accuracy': d[i]}

writer = pd.ExcelWriter("./LSTM/LSTMhistory1.xlsx")  # 初始化一个writer
df.to_excel(writer, float_format='%.6f')  # table输出为excel, 传入writer
writer.save()  # 保存
