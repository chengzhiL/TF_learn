#	example	of	loading	the	mnist	dataset
from keras.datasets import mnist
from matplotlib import pyplot as plt

#	load	dataset
(trainX, trainY), (testX, testY) = mnist.load_data()
#	summarize	loaded	dataset
print("trainX	shape", trainX.shape)
print("trainY	shape", trainY.shape)
print("testX	shape", testX.shape)
print("testY	shape", testY.shape)
#	plot	first	few	images
for i in range(9):
    #	define	subplot
    plt.subplot(330 + 1 + i)
    #	plot	raw	pixel	data
    plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
    #	show	the	figure
plt.show()

