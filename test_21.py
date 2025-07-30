import cv2
import numpy as np


def cv_show(neme, img):
    cv2.imshow(neme, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 形态学--腐蚀化   一般是黑色背景，白色字体
# 卷积核设定
# np.uint8是将这个处理的过程就是图像的float类型转变为uint8类型过程。
# float类型取值范围 ：-1 到1 或者 0到1
# uint8类型取值范围：0到255

# 这里使用 np.float32、np.uint8、或者不写 默认float64 都可以
# kernel = np.ones((5, 5), np.float32)
kernel = np.ones((5, 5))
print(np.dtype(kernel[0][0]))

img = cv2.imread("ce1.png")

# 膨胀操作 cv2.erode   迭代次数 iterations= 多少次
sb = cv2.erode(img, kernel, iterations=1)
cv_show('sda', sb)

# 腐蚀操作
kernel1 = np.ones((3, 3), np.uint8)
# 换个封装函数，腐蚀      当前载入 膨胀后--腐蚀
erosion = cv2.dilate(sb, kernel1, iterations=1)
cv_show('sda', erosion)

# 开运算与闭运算
# 开运算=先腐蚀，再膨胀
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv_show('sda', opening)

# 闭运算=先膨胀，再腐蚀
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv_show('sda', opening)

# 梯度= 膨胀-腐蚀
kernel = np.ones((7, 7), np.uint8)
gr = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
cv_show('sd', gr)

# 礼帽与黑帽
# 礼帽=原始输入-开运算结果
kernel = np.ones((7, 7), np.uint8)
gr = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv_show('sd', gr)

# 黑帽=原始输入结果 - 闭运算
kernel = np.ones((7, 7), np.uint8)
gr = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv_show('sd', gr)
