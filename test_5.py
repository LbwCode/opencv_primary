"""
图像平移、图像旋转、水平翻转
"""
import cv2
import numpy as np


def cv_show(neme,img):
    cv2.namedWindow(neme, cv2.WINDOW_NORMAL)
    cv2.imshow(neme, img)   
    cv2.waitKey(0)           
    cv2.destroyAllWindows() 


img = cv2.imread("./pingpang.png")
rows, cols = img.shape[0], img.shape[1]

# M = np.float  ([[1,0,x轴 左向右移动多少],[0,1,y轴 上向下移动多少]]) # 必须是foat32
M = np.float32([[1, 0, 100], [0, 1, 0]])

# 平移
# # 移动函数：参数(原图、平移多少、（背景高、宽）)
# dst = cv2.warpAffine(img, M, (cols, rows))
# cv_show("neme",dst)

# # 只有在输入移动多少的时候，需要foat32，但平移后的图像依然是uint8类型
# print(dst.dtype)
# print(img.dtype)


# 旋转
rotate = cv2.getRotationMatrix2D((rows*0.5, cols*0.5), 45, 1)
'''
第一个参数：旋转中心点
第二个参数：旋转角度
第三个参数：缩放比例
'''
res = cv2.warpAffine(img, rotate, (cols, rows))
cv_show("neme",res)

# # 水平翻转
# img_flip =  cv2.flip(img,1)
# cv_show("neme",img_flip)
