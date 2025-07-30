import cv2
import numpy as np

# 读取图像
image = cv2.imread("./pingpang.png")


# 定义原始图像中的三个点 (x, y) - 左上角、右上角、左下角
pts1 = np.float32([[50, 50], [450, 50], [50, 350]])

# 定义变换后图像中对应的三个点 - 拉伸和旋转效果
pts2 = np.float32([[100, 100], [400, 50], [50, 400]])

# 计算仿射变换矩阵
M = cv2.getAffineTransform(pts1, pts2)

# 应用变换
rows, cols = image.shape[:2]
dst = cv2.warpAffine(image, M, (cols, rows))

# 在原图上绘制控制点
for pt in pts1:
    cv2.circle(image, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

# 在变换后的图像上绘制控制点
for pt in pts2:
    cv2.circle(dst, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

# 显示原始图像和变换后的图像
cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
cv2.imshow('img1', image)
cv2.imshow('img2', dst)

# 显示变换矩阵
print("仿射变换矩阵:")
print(M)

# 等待按键，然后关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
