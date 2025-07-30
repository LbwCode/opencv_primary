import cv2
import numpy as np

# 读取图像
img = cv2.imread('t1.png')
h, w = img.shape[:2]

# 顺时针
p1 = np.float32([
    [220, 16],   # 左上点 
    [408, 45],   # 右上点
    [312, 372],  # 右下点
    [7, 251]     # 左下点 
])

# 位置1
# p2 = np.float32([
# [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]  
# ])

# 位置2
p2 = np.float32([
[379, 274], [468, 278], [469, 349], [375, 347]  
])

# 绘制 可选

# 复制原图用于标记
img_with_src = img.copy()
img_with_dst = img.copy()

# 绘制源点并标记数字
for i, (x, y) in enumerate(p1):
    cv2.circle(img_with_src, (int(x), int(y)), 10, (0, 255, 0), -1)  # 绿色实心圆
    cv2.putText(img_with_src, str(i+1), (int(x)+15, int(y)), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 标记数字

# 绘制目标点并标记数字
for i, (x, y) in enumerate(p2):
    cv2.circle(img_with_dst, (int(x), int(y)), 10, (0, 0, 255), -1)  # 红色实心圆
    if i == 0:
        cv2.putText(img_with_dst, str(i+1), (int(x)+15, int(y)+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # 标记数字
    if i == 1:
        cv2.putText(img_with_dst, str(i+1), (int(x)-20, int(y)+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # 标记数字
    if i == 2:
        cv2.putText(img_with_dst, str(i+1), (int(x)-20, int(y)-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # 标记数字
    if i == 3:
        cv2.putText(img_with_dst, str(i+1), (int(x)+15, int(y)-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # 标记数字

# 绘制源四边形和目标四边形
cv2.polylines(img_with_src, [p1.astype(np.int32)], True, (0, 255, 0), 2)
cv2.polylines(img_with_dst, [p2.astype(np.int32)], True, (0, 0, 255), 2)


# 计算透视变换矩阵
M = cv2.getPerspectiveTransform(p1, p2)
# 应用变换
warped = cv2.warpPerspective(img, M, (w, h))

print(img.shape)
print(img_with_src.shape)
print(img_with_dst.shape)
print(warped.shape)

# 缩放
img_new = cv2.resize(warped, (400, 600))
print(img_new.shape)


# 显示结果
cv2.imshow('1', img_with_src)
cv2.waitKey(0)
cv2.imshow('2', img_with_dst)
cv2.waitKey(0)
cv2.imshow('3', warped)
cv2.waitKey(0)
cv2.imshow('4', img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()    