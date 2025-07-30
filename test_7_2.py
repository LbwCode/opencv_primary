import cv2
import numpy as np

# 读取图像
image = cv2.imread('t1.png')

# 定义四边形的四个顶点坐标 [x, y]
points = np.array([
    [197, 137],   # 左上点
    [286, 163],   # 右上点
    [232, 278],   # 右下点
    [123, 241]     # 左下点
], dtype=np.int32)

# 在原图上绘制四边形
cv2.polylines(image, [points], True, (0, 255, 0), 2)  # 绿色线条，线宽2

# 计算四边形的边界框
x, y, w, h = cv2.boundingRect(points)

# 截取感兴趣区域(ROI)
roi = image[y:y+h, x:x+w].copy()

# 调整顶点坐标为ROI内的相对坐标
roi_points = points - np.array([[x, y]])

# 创建与ROI大小相同的掩码
mask = np.zeros_like(roi)

# 在掩码上绘制白色四边形
cv2.fillPoly(mask, [roi_points], (255, 255, 255))

# 按位与操作，提取四边形内的图像
result = cv2.bitwise_and(roi, mask)

# 显示结果
cv2.imshow('Original', image)
# cv2.waitKey(0)
cv2.imshow('ROI', roi)
# cv2.waitKey(0)
cv2.imshow('Mask', mask)
# cv2.waitKey(0)AA
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()