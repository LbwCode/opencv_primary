"""
图像缩放、插值方式缩放
"""

import cv2


# image = cv2.imread("./ce.png")
# image = cv2.imread("pingpang.png")
image = cv2.imread("yuan1.png")

height, width = image.shape[0], image.shape[1]
print(height, width)

cv2.imshow("neme", image)   


# 可选的插值方式如下：
# INTER_NEAREST - 最近邻插值
# INTER_LINEAR - 线性插值（默认值）
# img_new = cv2.resize(image, (300, 640), cv2.INTER_NEAREST)
# cv2.imshow("neme1", img_new)   

# 按比例缩放
# 缩放到原来的二分之一，输出尺寸格式为（宽，高）
img_new = cv2.resize(image, (100, 600))


cv2.imwrite('ce3.png',img_new)

cv2.imshow("neme2", img_new)   
cv2.waitKey(0) 
