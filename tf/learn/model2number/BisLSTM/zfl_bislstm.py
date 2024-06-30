import tensorflow as tf
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential

from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling1D,Dense, Dropout,Flatten, Dense, BatchNormalization
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
x_test.to_csv(r'C:\LM paper\application\model2数字\model2number\BILSTM\x_test_split.csv')
y_test.to_csv(r'C:\LM paper\application\model2数字\model2number\BILSTM\y_test_split.csv')

x_train =np.array(x_train.values)
x_test =np.array(x_test.values)
y_train= np.array(y_train.values)
y_test=np.array(y_test.values)
print(type(x_train))


x_train = x_train.reshape(x_train.shape[0],10,10)
print("x.shape", x_train.shape)
x_test = x_test.reshape(x_test.shape[0],10,10)
print("x.shape", x_test.shape)




nb_lstm_outputs = 180  #神经元个数
nb_time_steps = 10 #时间序列长度
nb_input_vector = 10 #输入序列

model = Sequential()
model.add(Bidirectional(LSTM(units = nb_lstm_outputs,return_sequences=True),
                        input_shape=(nb_time_steps, nb_input_vector)))
model.add(Dropout(0.9))
model.add(BatchNormalization())
#model.add(TimeDistributed(Dense(26, activation='softmax')))


#fc
model.add(GlobalAveragePooling1D())
model.add(Dense(10, activation='softmax'))

print(model.summary())

from tensorflow.keras import regularizers

# 设置较小的学习率
learning_rate = 0.005

# 编译模型，使用Adam优化器，并设置学习率
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer=optimizer,
              metrics=["sparse_categorical_accuracy"])


# checkpoint
filepath="./BILSTM/bilstmMODEL1.1-{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_sparse_categorical_accuracy',verbose=1,save_best_only=True,mode='max')
callbacks_list = [checkpoint]

#Fit the model
history=model.fit(x_train, y_train,
                  validation_data=(x_test, y_test),
                  epochs= 100,
                  batch_size=32,
                  callbacks=callbacks_list,
                  verbose=1)


model.save(r"./BILSTM/modelbilstm1-1.hdf5")

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

writer = pd.ExcelWriter("./BILSTM/bilstm1-1history1.xlsx")  # 初始化一个writer
df.to_excel(writer, float_format='%.6f')  # table输出为excel, 传入writer
writer.save()

feature_extractor = Sequential(model.layers[:-1])
features = feature_extractor.predict(x_train)

# 使用t-SNE降维将训练后的特征表示可视化
tsne = TSNE(n_components=2, random_state=0)
features_tsne = tsne.fit_transform(features)

# 进行聚类
kmeans = KMeans(n_clusters=10, random_state=0)
labels = kmeans.fit_predict(features)

# 绘制训练后的二维数据聚类图
plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='Set3')
scatter=plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='Set3')
# plt.title('Clustering of Trained Data')
plt.title('Clustering of Trained Data', fontsize=16)

# # 添加图例
# legend_labels = ['Class {}'.format(i) for i in range(10)]
# plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, loc='upper right', title='Classes')

plt.legend()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# # 特征数据标准化
# # scaler = StandardScaler()
# # X_scaled = scaler.fit_transform(X)
#
# # t-SNE降维
# # tsne = TSNE(n_components=2, random_state=0)
# # X_tsne = tsne.fit_transform(features)
# #
# # # 聚类
# # kmeans = KMeans(n_clusters=10, random_state=0)
# # labels = kmeans.fit_predict(features)
# #
# # # 绘制图形
# # plt.figure(figsize=(8, 8))
# colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
# for i in range(10):
#     cluster_points = labels[y==i]
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label='Class {}'.format(i))
#
# plt.title('t-SNE Clustering')
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.legend()
# plt.show()

