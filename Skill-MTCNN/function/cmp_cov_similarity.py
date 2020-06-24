import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread("cor_matrix/cor_matrix1.png")
img2 = cv2.imread("cor_matrix/cor_matrix2.png")

H1 = cv2.calcHist([img1], [1], None, [256], [0, 256])
H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理

H2 = cv2.calcHist([img2], [1], None, [256], [0, 256])
H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理

plt.title('similarity between matrix')
plt.xlabel('pixel value')
plt.ylabel('number')

plt.plot(np.array(range(256)), H1, color="orange", linewidth=1, linestyle=':', label='cor_matrix1', marker='o')
plt.plot(np.array(range(256)), H2, color="darkblue", linewidth=0.5, linestyle='--', label='cor_matrix2', marker='+')


plt.legend(loc=2)  # 图例展示位置，数字代表第几象限
plt.show()  # 显示图像

