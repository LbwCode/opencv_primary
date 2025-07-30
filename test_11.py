import cv2
import numpy as np


def cv_show(neme,img):
    # cv2.namedWindow(neme, cv2.WINDOW_NORMAL)
    cv2.imshow(neme, img)   
    cv2.waitKey(0)           
    cv2.destroyAllWindows() 

# 读取原始图像
img = cv2.imread("ce1.png")

# 获取图像尺寸
h, w = img.shape[:2]

# 1. 均值滤波及对比显示
blur_mean = cv2.blur(img, (3, 3))
# 创建并排显示的图像（原图+滤波结果）
combined_mean = np.hstack((img, blur_mean))
cv_show("1",combined_mean)

# 2. 方框滤波及对比显示
blur_box = cv2.boxFilter(img, -1, (3, 3), normalize=True)
combined_box = np.hstack((img, blur_box))
cv_show("2",combined_box)

# 3. 高斯滤波及对比显示
blur_gaussian = cv2.GaussianBlur(img, (5, 5), 1)
combined_gaussian = np.hstack((img, blur_gaussian))
cv_show("3",combined_gaussian)

# 4. 中值滤波及对比显示
blur_median = cv2.medianBlur(img, 5)
combined_median = np.hstack((img, blur_median))
cv_show("4",combined_median)

# 5. 双边滤波及对比显示
blur_bilateral = cv2.bilateralFilter(img, 9, 75, 75)
combined_bilateral = np.hstack((img, blur_bilateral))
cv_show("5",combined_bilateral)