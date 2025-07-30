import cv2
import numpy as np

def cv_show(neme, img):
    cv2.imshow(neme, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 生成一张图像  全黑
img = np.zeros([300, 300, 3])

xy1 = 10, 40
xy2 = 100, 40

xy1_ = 10, 80
xy2_ = 100, 80


# 画线 参数：图像、坐标1、坐标2、颜色(蓝B,绿G,红R)、粗细
cv2.line(img, xy1, xy2, (0, 255, 0), 5)
cv2.line(img, xy1_, xy2_, (0, 0, 255), 2)

# 矩形 左上角、右下角、颜色、线条粗细或填充
cv2.rectangle(img, (200, 200), (230, 230), (255, 0, 0), -1)

# 图像、圆心xy、半径、颜色、粗细
img = cv2.circle(img, (150, 150), 30, (0, 0, 255), 1)

cv_show("neme", img)