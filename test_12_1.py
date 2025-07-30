import cv2
import numpy as np

# 读取图像并转换为灰度图
image = cv2.imread("ce2.png", 0)
if image is None:
    print("无法读取图像，请检查路径")
    exit()

# 参数设置
sigma = 1
low_threshold = 50
high_threshold = 150

# 1. 高斯滤波 - 减少噪声
blurred = cv2.GaussianBlur(image, (5, 5), sigma)

# 2. 计算梯度幅值和方向
sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

# 计算梯度幅值和方向
gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
gradient_direction = np.arctan2(sobely, sobelx) * 180 / np.pi

# 将梯度方向量化为4个方向之一（0°, 45°, 90°, 135°）
gradient_direction = (gradient_direction + 180) % 180
for i in range(1, gradient_direction.shape[0] - 1):
    for j in range(1, gradient_direction.shape[1] - 1):
        angle = gradient_direction[i, j]
        if (0 <= angle < 22.5) or (157.5 <= angle < 180):
            gradient_direction[i, j] = 0  # 水平方向
        elif 22.5 <= angle < 67.5:
            gradient_direction[i, j] = 45  # 对角线方向
        elif 67.5 <= angle < 112.5:
            gradient_direction[i, j] = 90  # 垂直方向
        else:
            gradient_direction[i, j] = 135  # 反对角线方向

# 3. 非极大值抑制 - 细化边缘
suppressed = np.zeros_like(gradient_magnitude)
for i in range(1, gradient_magnitude.shape[0] - 1):
    for j in range(1, gradient_magnitude.shape[1] - 1):
        angle = gradient_direction[i, j]
        current = gradient_magnitude[i, j]
        
        # 根据梯度方向检查相邻像素
        if angle == 0:  # 水平方向
            neighbor1 = gradient_magnitude[i, j+1]
            neighbor2 = gradient_magnitude[i, j-1]
        elif angle == 45:  # 对角线方向
            neighbor1 = gradient_magnitude[i+1, j+1]
            neighbor2 = gradient_magnitude[i-1, j-1]
        elif angle == 90:  # 垂直方向
            neighbor1 = gradient_magnitude[i+1, j]
            neighbor2 = gradient_magnitude[i-1, j]
        else:  # 反对角线方向
            neighbor1 = gradient_magnitude[i+1, j-1]
            neighbor2 = gradient_magnitude[i-1, j+1]
        
        # 如果当前像素值是局部最大值，则保留
        if current >= neighbor1 and current >= neighbor2:
            suppressed[i, j] = current

# 4. 双阈值处理和边缘连接
edges = np.zeros_like(suppressed, dtype=np.uint8)
strong = 255
weak = 50

# 标记强边缘和弱边缘
strong_i, strong_j = np.where(suppressed >= high_threshold)
weak_i, weak_j = np.where((suppressed >= low_threshold) & (suppressed < high_threshold))

edges[strong_i, strong_j] = strong
edges[weak_i, weak_j] = weak

# # 边缘连接 - 检查弱边缘是否与强边缘相连
# for i in range(1, edges.shape[0] - 1):
#     for j in range(1, edges.shape[1] - 1):
#         if edges[i, j] == weak:
#             # 检查8邻域
#             if (edges[i-1, j-1] == strong) or (edges[i-1, j] == strong) or \
#                (edges[i-1, j+1] == strong) or (edges[i, j-1] == strong) or \
#                (edges[i, j+1] == strong) or (edges[i+1, j-1] == strong) or \
#                (edges[i+1, j] == strong) or (edges[i+1, j+1] == strong):
#                 edges[i, j] = strong
#             else:
#                 edges[i, j] = 0

# 显示原图和边缘图
# cv2.namedWindow('Canny Edges', cv2.WINDOW_NORMAL)
# cv2.imshow('Original', image)
cv2.imshow('Canny Edges', edges)

# 等待按键退出
cv2.waitKey(0)
cv2.destroyAllWindows()