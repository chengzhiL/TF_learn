import tensorflow as tf
import os
import numpy as np
import pandas as pd
from keras.src.layers import GlobalAveragePooling2D
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

class Inception(Model):
    def __init__(self):
        super(Inception,self).__init__()
        self.c1 = Conv2D(filters=32, kernel_size=(2, 2), padding='same',activation='relu',input_shape=(10,10,1))
        self.b1 = BatchNormalization()
        self.c2 = Conv2D(filters=32, kernel_size=(2, 2), padding='same',activation='relu')
        self.b2 = BatchNormalization()
        self.p4_1 = MaxPool2D(2)
        self.c3 = Conv2D(filters=64, kernel_size=(2, 2), padding='same',activation='relu')
        self.b3 = BatchNormalization()
        self.c4 = Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu')
        self.b4 = BatchNormalization()
        self.p4_2 = MaxPool2D(2)
        self.c5 = Conv2D(filters=128, kernel_size=(2, 2), padding='same', activation='relu')
        self.b5 = BatchNormalization()
        self.c6 = Conv2D(filters=128, kernel_size=(2, 2), padding='same', activation='relu')
        self.b6 = BatchNormalization()
        self.p4_3 = MaxPool2D(2)
        self.c7 = Conv2D(filters=256, kernel_size=(2, 2), padding='same', activation='relu')
        self.b7 = BatchNormalization()
        self.c8 = Conv2D(filters=256, kernel_size=(2, 2), padding='same', activation='relu')
        self.b8 = BatchNormalization()
        self.p4_4 = MaxPool2D(2, padding='same')
        self.g1 = GlobalAveragePooling2D()
        self.f1 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.p4_1(x)
        x = self.c3(x)
        x = self.b3(x)
        x = self.c4(x)
        x = self.b4(x)
        x = self.p4_2(x)
        x = self.c5(x)
        x = self.b5(x)
        x = self.c6(x)
        x = self.b6(x)
        x = self.p4_3(x)
        x = self.c7(x)
        x = self.b7(x)
        x = self.c8(x)
        x = self.b8(x)
        x = self.p4_4(x)
        x = self.g1(x)
        y = self.f1(x)
        return  y
model = Inception()

learning_rate = 0.00001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 设置较小的学习率
model.compile(optimizer= optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])




checkpoint_save_path = "./Inception.weights.h5"

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







