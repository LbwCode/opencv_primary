
import cv2


def cv_show(neme,img):
    # 调整宽高（再次运行也只会加载你调整后的宽高）
    # cv2.namedWindow(neme, cv2.WINDOW_NORMAL)
    cv2.imshow(neme, img)    # 必要参数：名字和变量名
    cv2.imshow("name", img)
    cv2.waitKey(0)           # 括号中0=任意键终止，单位为毫秒级别
    cv2.destroyAllWindows()  # 关闭所有窗口--图片
    
# cv2读取图像是BGR  不是RGB
image = cv2.imread("./pingpang.png")

print(image.shape)  # 高、宽、维度
# print(image[0][0])


cv_show("name", image)

