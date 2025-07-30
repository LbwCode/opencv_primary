import cv2
import numpy as np

# 读取图像并转为灰度图
img = cv2.imread('yuan.jpeg')
# img = cv2.imread('yuan1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯模糊减少噪声
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 霍夫圆环检测
# 参数说明:
# - cv2.HOUGH_GRADIENT: 使用梯度法检测圆（最常用）
# - 1: 累加器图像的分辨率与原图相同
# - 20: 检测到的圆之间的最小距离（避免重复检测）
# - param1=100: Canny边缘检测的高阈值（低阈值自动设为高阈值的一半）
# - param2=30: 累加器阈值（值越小，检测到的圆越多）
# - minRadius=0: 最小圆半径
# - maxRadius=0: 最大圆半径（0表示不限制）

# 霍夫圆形检测
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                           param1=100, param2=30, minRadius=0, maxRadius=0)
# print(circles)
# 绘制检测结果
if circles is not None:
    # 将圆的参数转换为整数类型 
    # 无符号 16 位整数、四舍五入
    circles = np.uint16(np.around(circles))
    print(circles.shape)
    for i in circles[0, :]:
        
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 绘制圆
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)  # 绘制圆心

# 显示结果
cv2.imshow('Detected Circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()