import tensorflow as tf

a = tf.zeros([2, 3])
b = tf.ones(4)
c = tf.fill([2, 2], 9)
d = tf.fill([4,1],1)
print("a:", a)
print("b:", b)
print("c:", c)
print("d:", d)
