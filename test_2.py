import cv2

# http://admin:admin@192.168.31.117:8081
# 海康摄像头
# url = 'rtsp://admin:tsg610@192.168.31.57:554//Streaming/Channels/2'
# url = 'rtsp://admin:mima123@172.46.30.210:554/Streaming/Channels/101?transportmode=multica

# 打开摄像头
cap = cv2.VideoCapture("./pingpang.mp4")

if not cap.isOpened():
    print("无法打开摄像头或视频文件")
    exit()

while True:
    # 读取一帧
    ret, frame = cap.read()
    
    if not ret:
        print("无法获取帧")
        break
    
    # 显示帧
    cv2.imshow('Video', frame)
    
    # 按 'q' 键退出循环
    if cv2.waitKey(30) == ord('q'):
        break

# 释放资源并关闭窗口
cap.release()            # 释放摄像头或者视频文件资源
cv2.destroyAllWindows()  # 关闭所有由 OpenCV 创建的窗口