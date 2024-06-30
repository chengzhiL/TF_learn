import tensorflow as tf
import os
import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import GlobalAveragePooling2D, LSTM, GlobalAveragePooling1D
from matplotlib import pyplot as plt
from tensorflow.keras import Model

from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization,MaxPool2D
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'../model2number.csv',encoding='gbk',index_col=0)
# 划分特征与标签
x = data.iloc[:, data.columns != "label"]
y = data.iloc[:, data.columns == "label"]

print(type(x))

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
print(type(x_train))
x_test.to_csv(r'./x_test_split.csv')
y_test.to_csv(r'./y_test_split.csv')

x_train =np.array(x_train.values)
x_test =np.array(x_test.values)
y_train= np.array(y_train.values)
y_test=np.array(y_test.values)
print(type(x_train))


x_train = x_train.reshape(x_train.shape[0],10,10,1)
print("x.shape", x_train.shape)
x_test = x_test.reshape(x_test.shape[0],10,10,1)
print("x.shape", x_test.shape)



#parameters for LSTM
nb_lstm_outputs = 200  #神经元个数
nb_time_steps = 10  #时间序列长度
nb_input_vector = 10 #输入序列

model = Sequential()
model.add(LSTM(units= nb_lstm_outputs, input_shape=(nb_time_steps, nb_input_vector),return_sequences=True))
model.add(Dropout(0.95))


model.add(GlobalAveragePooling1D())

#output
model.add(Dense(10, activation='softmax'))
print(model.summary())
model.summary()




#
# class Lstm(Model):
#     def __init__(self):
#         super(Lstm,self).__init__()
#         self.model = tf.keras.Sequential([
#             LSTM(units= nb_lstm_outputs, input_shape=(nb_time_steps, nb_input_vector),return_sequences=True),
#             Dense(0.95),
#             GlobalAveragePooling1D(),
#             Dense(10, activation='softmax')
#         ])
#
#     def call(self,x):
#         x = self.model(x)
#         return x
#
# model = Lstm()

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#               metrics=['sparse_categorical_accuracy'])
learning_rate = 0.004

# 编译模型，使用Adam优化器，并设置学习率
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer=optimizer,
              metrics=["sparse_categorical_accuracy"])


checkpoint_save_path = "./LSTM.weights.h5"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=500, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

# print(model.trainable_variables)
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
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
plt.legend()
plt.show()






