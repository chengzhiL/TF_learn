import tensorflow as tf

x1 = tf.constant([1., 2., 3.], dtype=tf.float64)
print("x1:", x1)
# cast 执行张量数据类型的转换
x2 = tf.cast(x1, tf.int32)
print("x2", x2)
# 计算张量的最小值
print("minimum of x2：", tf.reduce_min(x2))
print("maxmum of x2:", tf.reduce_max(x2))
