import cv2
import numpy as np

def cv_show(neme, img):
    cv2.imshow(neme, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 图像轮廓
img = cv2.imread('ce3.png')
# 灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 二值处理
ret, thresh = cv2.threshold(gray, 157, 255, cv2.THRESH_BINARY)
# cv_show('sad', thresh)

# 轮廓查询
# cv2.findContours 只接受二值化后的图像
# 所有轮廓、外轮廓、内外轮廓、所有轮廓 爷、父、儿、孙
print(cv2.RETR_LIST, cv2.RETR_EXTERNAL, cv2.RETR_CCOMP, cv2.RETR_TREE)
#  cv2.CHAIN_APPROX_NONE，如果是一条直线, 它会取所有端点
#  cv2.CHAIN_APPROX_SIMPLE 如果是一条直线, 它只取两个端点
# 根据版本，返回的参数有可能是3个
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for i in range(len(contours)):
    cnt = contours[i]
    # 轮廓面积
    area = cv2.contourArea(cnt)
    # 轮廓周长
    perimeter = cv2.arcLength(cnt, True)
    print(area, perimeter)

    if 10 < area < 100000:
        # 精准绘制轮廓(需要绘制的图像,轮廓,-1是所有轮廓或者第几个,颜色通道蓝绿红,粗细)
        cv2.drawContours(img, cnt, -1, (0, 0, 255), 3)

        # 用于显示
        # 获取4点
        x, y, w, h = cv2.boundingRect(cnt)
        # 轮廓外接矩形 正矩形,不可倾斜
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv_show('sad', img)

        # # 旋转最小外接矩形
        # rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)  # 对box进行处理 这一步一定要进行
        # # 精准绘制轮廓
        # cv2.drawContours(img, [box], -1, (0, 255, 0), 1)
        # cv_show("s", img)


        # # 轮廓外接圆
        # (x, y), radius = cv2.minEnclosingCircle(cnt)
        # center = (int(x), int(y))
        # radius = int(radius)
        # # 图像、圆心xy、半径、颜色、粗细
        # img = cv2.circle(img, center, radius, (0, 0, 255), 1)
        # cv_show('sad', img)

        # # 轮廓近似，可倾斜，非矩形
        # epsilon = 0.1 * cv2.arcLength(cnt, True)
        # approx = cv2.approxPolyDP(cnt, epsilon, True)
        # # 绘制多边形
        # cv2.polylines(img, [approx], True, (0, 255, 255), 2)
        # cv_show('sad', img)

        # # 绘制凸包
        # hull = cv2.convexHull(cnt)
        # # 在原始图像上绘制凸包，将凸包的顶点依次连接起来
        # cv2.polylines(img, [hull], True, (0, 255, 0), 2)

        # # 绘制多边形
        # i = cv2.polylines(img, [hull], True, (255, 255, 0), 2)
        # cv_show("s", img)


# cv_show('sad', img)
