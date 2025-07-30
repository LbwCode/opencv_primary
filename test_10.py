import cv2


def nothing(x):
    pass


# img = cv2.imread("./pingpang.png", 0)
img = cv2.imread("./ce3.png", 0)
# 截图
# img = img[0:579, 315:704]

cv2.namedWindow('res')

cv2.createTrackbar('max', 'res', 127, 255, nothing)
cv2.createTrackbar('min', 'res', 127, 255, nothing)

# img = cv2.resize(img, (640, 360))
maxVal = 200
minVal = 100

while True:
    key = cv2.waitKey(1) & 0xff
    if key == ord(" "):
        break
    maxVal = cv2.getTrackbarPos('min', 'res')
    minVal = cv2.getTrackbarPos('max', 'res')

    ret, edge = cv2.threshold(img, minVal, maxVal, cv2.THRESH_BINARY)
    cv2.imshow('res', edge)
