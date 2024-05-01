import tensorflow as tf

d = tf.random.normal([2, 2], mean=0.5, stddev=1)
print("d:", d)
# 方差为1，均值为0.5 截断
e = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
print("e:", e)
