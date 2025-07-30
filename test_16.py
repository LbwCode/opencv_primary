import cv2
import numpy as np

# 读取图像并转换为灰度图
# img = cv2.imread('ce1.png')
img = cv2.imread('ce2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Canny边缘检测 - 霍夫变换对边缘图进行操作
# 参数2和3分别是低阈值和高阈值，用于控制边缘检测的灵敏度
# apertureSize是Sobel算子的大小，用于计算图像梯度
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 概率霍夫直线检测 - 直接返回线段端点坐标
# 参数说明：
# 1 - 距离分辨率（像素），表示ρ的精度
# np.pi/180 - 角度分辨率（弧度），表示θ的精度
# threshold=100 - 累加器阈值，高于此值的线段才会被认为是直线
# minLineLength=50 - 最小线段长度，短于该值的线段会被忽略
# maxLineGap=10 - 同一线段上的点之间允许的最大间隙，值越大越容易连接间断线段
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

# 绘制检测到的直线
if lines is not None:
    # 遍历每条检测到的线段
    for line in lines:
        # 获取线段的两个端点坐标
        x1, y1, x2, y2 = line[0]
        # 在原图上绘制红色线段（BGR格式：(0, 0, 255)）
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 显示结果
# 创建可调整大小的窗口
cv2.namedWindow('Hough Lines', cv2.WINDOW_NORMAL)
# 显示处理后的图像
cv2.imshow('Hough Lines', img)
# 等待按键退出
cv2.waitKey(0)
# 释放窗口资源
cv2.destroyAllWindows()



# import cv2
# import numpy as np

# def hough_demo(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150)
    
#     cv2.namedWindow('Hough Lines')
    
#     # 创建滑动条
#     cv2.createTrackbar('Threshold', 'Hough Lines', 100, 300, lambda x: None)
#     cv2.createTrackbar('Min Length', 'Hough Lines', 50, 200, lambda x: None)
#     cv2.createTrackbar('Max Gap', 'Hough Lines', 10, 100, lambda x: None)
    
#     while True:
#         # 获取当前参数值
#         threshold = cv2.getTrackbarPos('Threshold', 'Hough Lines')
#         min_length = cv2.getTrackbarPos('Min Length', 'Hough Lines')
#         max_gap = cv2.getTrackbarPos('Max Gap', 'Hough Lines')
        
#         # 执行霍夫变换
#         lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, min_length, max_gap)
        
#         # 绘制结果
#         result = img.copy()
#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
#         cv2.imshow('Hough Lines', result)
        
#         # 按ESC退出
#         if cv2.waitKey(100) == 27:
#             break
    
#     cv2.destroyAllWindows()

# # 使用示例
# hough_demo('ce2.png')