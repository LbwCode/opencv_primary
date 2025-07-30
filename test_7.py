import numpy as np
import cv2

img = cv2.imread("./pingpang.png")
rows, cols, channels = img.shape
p1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [rows - 1, cols - 1]])
x1, y1, x2, y2, x3, y3, x4, y4 = 0, rows * 0.3, cols * 0.8, rows * 0.2, cols * 0.15, rows * 0.7, cols * 0.8, rows * 0.8

print("x1 y1: ", x1, y1)
print("x2 y2: ", x2, y2)
print("x3 y3: ", x3, y3)
print("x4 y4: ", x4, y4)

p2 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
'''
cv.getPerspectiveTransform(p1, p2)
p1 输入四点坐标  想要变换的图像位置
p2 输出四点坐标  变换后的图像位置
左上、右上、左下、右下
'''
# 生成变换矩阵
M = cv2.getPerspectiveTransform(p1, p2)
# 进行透视变换 参数：（原图、变换矩阵、（背景高、宽））
dst = cv2.warpPerspective(img, M, (cols, rows))
# # 保存图像
cv2.imwrite('2.png',dst)
cv2.imshow('result', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

