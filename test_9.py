import cv2


def cv_show(neme, img):
    cv2.namedWindow(neme, cv2.WINDOW_NORMAL)
    cv2.imshow(neme, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread("./pingpang.png")
# 灰度化
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化
# 大于127=0,小于127=255
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# 大于127=255,小于127=0
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# 大于127=127,否则不变
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)

# 小于127=0,否则不变
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)

# 大于127=0,否则不变
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

cv_show("s", img)
cv_show("s1", thresh1)
cv_show("s2", thresh2)
cv_show("s3", thresh3)
cv_show("s4", thresh4)
cv_show("s5", thresh5)
