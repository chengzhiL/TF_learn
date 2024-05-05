import tensorflow as tf
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, \
    GlobalAveragePooling2D
from tensorflow.keras import Sequential
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, BatchNormalization  # RNN、激活函数、全连接层模块
from tensorflow.keras.losses import SparseCategoricalCrossentropy  # loss模块
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import pandas as pd
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, GlobalAveragePooling1D, Dense, BatchNormalization
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data = pd.read_csv(r'C:\LM paper\application\model2数字\model2number\model2number.csv',encoding='gbk',index_col=0)
# 划分特征与标签
x = data.iloc[:, data.columns != "label"]
y = data.iloc[:, data.columns == "label"]

print(type(x))


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
print(type(x_train))
x_test.to_csv(r'C:\LM paper\application\model2数字\model2number\INCEPTION\x_test_split.csv')
y_test.to_csv(r'C:\LM paper\application\model2数字\model2number\INCEPTION\y_test_split.csv')

x_train =np.array(x_train.values)
x_test =np.array(x_test.values)
y_train= np.array(y_train.values)
y_test=np.array(y_test.values)
print(type(x_train))


x_train = x_train.reshape(x_train.shape[0],10,10,1)
print("x.shape", x_train.shape)
x_test = x_test.reshape(x_test.shape[0],10,10,1)
print("x.shape", x_test.shape)
model = Sequential([
    Conv2D(32, 2, padding='same', activation='relu', input_shape=(10,10,1)),
    BatchNormalization(),
    Conv2D(32, 2, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(64, 2, padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, 2, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(128, 2, padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, 2, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(256, 2, padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(256, 2, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2),padding='same'),
    GlobalAveragePooling2D(),
    Dense(10, activation='softmax')
])
learning_rate = 0.00001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer=optimizer,
              metrics=["sparse_categorical_accuracy"])

filepath="./INCEPTION/INCEPTIONMODEL1.1-{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_sparse_categorical_accuracy',verbose=1,save_best_only=True,mode='max')
callbacks_list = [checkpoint]

#Fit the model
history=model.fit(x_train, y_train,
                  validation_data=(x_test, y_test),
                  epochs= 500,
                  batch_size=128,
                  callbacks=callbacks_list,
                  verbose=1)


model.save(r"./INCEPTION/modelINCEPTION1-1.hdf5")

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

a = history.history['loss']
b = history.history['sparse_categorical_accuracy']
c = history.history['val_loss']
d = history.history['val_sparse_categorical_accuracy']
count = len(a)

df = pd.DataFrame(columns=['loss', 'sparse_categorical_accuracy', 'val_loss', 'val_sparse_categorical_accuracy'])
for i in range(count):
    df.loc[i] = {'loss': a[i], 'sparse_categorical_accuracy': b[i], 'val_loss': c[i],
                 'val_sparse_categorical_accuracy': d[i]}

writer = pd.ExcelWriter("./INCEPTION/INCEPTION1-1history1.xlsx")  # 初始化一个writer
df.to_excel(writer, float_format='%.6f')  # table输出为excel, 传入writer
writer.save()
