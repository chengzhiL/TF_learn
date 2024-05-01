import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
# 沿垂直方向堆叠数组
c = np.vstack((a, b))
print("c:\n", c)

