"""
灰度图 保存图像
"""

import cv2


def cv_show(neme,img):
    cv2.namedWindow(neme, cv2.WINDOW_NORMAL)
    cv2.imshow(neme, img)   
    cv2.waitKey(0)           
    cv2.destroyAllWindows() 
    
# # 灰度图 方法1
# image = cv2.imread("./pingpang.png", 0)
# print(image.shape)
# cv_show("name", image)
 
# 灰度图 方法2
image1 = cv2.imread("./pingpang.png")
#image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)  # 灰度 cv2.COLOR_BGR2GRAY
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)  # HSV cv2.COLOR_BGR2HSV
cv_show("name", image1)

# # # 保存图像
# cv2.imwrite('1.png',image1)
