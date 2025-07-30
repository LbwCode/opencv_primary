"""
ROI区域 去除通道
"""
import cv2


def cv_show(neme,img):
    cv2.namedWindow(neme, cv2.WINDOW_NORMAL)
    cv2.imshow(neme, img)   
    cv2.waitKey(0)           
    cv2.destroyAllWindows() 
    

image = cv2.imread("./pingpang.png")

cv_show("name", image)



x1, y1, x2, y2 = 831, 645, 882, 704
# 截图 y1 y2 x1 x2
# ball = image[y1:y2, x1:x2]
# cv_show("n", ball)

# bgr
# image[y1:y2, x1:x2, 1] = 0
# cv_show("name", image)



# # 只保留单通道 3原色
# # 将3种的两个置0 剩下的就是单通道啦
cur_img = image.copy()  # 想保留哪个通道，就注释掉哪个
cur_img[:, :, 0] = 0  # 0蓝B
cur_img[:, :, 1] = 0  # 1绿G
# cur_img[:, :, 2] = 0  # 2红R
cv_show("n", cur_img)
