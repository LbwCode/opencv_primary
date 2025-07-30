import cv2
import numpy as np

def edge_detection_comparison(image_path, blur_size=3, threshold=30, canny_threshold1=50, canny_threshold2=150):
    """
    比较四种边缘检测算法: Sobel, Scharr, Laplacian 和 Canny
    
    参数:
        image_path: 输入图像路径
        blur_size: 高斯模糊核大小
        threshold: 二值化阈值
        canny_threshold1: Canny算法低阈值
        canny_threshold2: Canny算法高阈值
    """
    # 读取图像并转换为灰度图
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    # 1. Sobel算子 - 计算水平和垂直梯度
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # 水平梯度
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # 垂直梯度
    sobel = np.sqrt(sobelx**2 + sobely**2)  # 梯度幅值
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # 归一化到0-255
    _, sobel_binary = cv2.threshold(sobel, threshold, 255, cv2.THRESH_BINARY)  # 二值化
    
    # 2. Scharr算子 - 比Sobel更敏感的梯度算子
    scharrx = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)  # 水平梯度
    scharry = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)  # 垂直梯度
    scharr = np.sqrt(scharrx**2 + scharry**2)  # 梯度幅值
    scharr = cv2.normalize(scharr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # 归一化
    _, scharr_binary = cv2.threshold(scharr, threshold, 255, cv2.THRESH_BINARY)  # 二值化
    
    # 3. Laplacian算子 - 各向同性的二阶导数算子
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)  # 计算拉普拉斯变换
    laplacian = np.absolute(laplacian)  # 取绝对值
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # 归一化
    _, laplacian_binary = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY)  # 二值化
    
    # 4. Canny边缘检测 - 多阶段边缘检测算法
    canny = cv2.Canny(blurred, canny_threshold1, canny_threshold2)  # 应用Canny算法
    
    # 创建一个2×3的网格布局来显示所有结果
    h, w = img.shape[:2]
    result = np.zeros((h*2, w*3, 3), dtype=np.uint8)
    
    # 第一行
    result[0:h, 0:w] = img  # 原始图像
    result[0:h, w:2*w] = cv2.cvtColor(sobel_binary, cv2.COLOR_GRAY2BGR)  # Sobel结果
    result[0:h, 2*w:3*w] = cv2.cvtColor(scharr_binary, cv2.COLOR_GRAY2BGR)  # Scharr结果
    
    # 第二行
    result[h:2*h, 0:w] = cv2.cvtColor(laplacian_binary, cv2.COLOR_GRAY2BGR)  # Laplacian结果
    result[h:2*h, w:2*w] = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)  # Canny结果
    
    # 添加英文标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'Original', (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(result, 'Sobel', (w+10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(result, 'Scharr', (2*w+10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(result, 'Laplacian', (10, h+30), font, 1, (0, 255, 0), 2)
    cv2.putText(result, 'Canny', (w+10, h+30), font, 1, (0, 255, 0), 2)
    
    # 显示结果
    cv2.namedWindow('Edge Detection Comparison', cv2.WINDOW_NORMAL)
    cv2.imshow('Edge Detection Comparison', result)
    
    # 调整窗口大小以适应屏幕
    cv2.resizeWindow('Edge Detection Comparison', 1500, 1000)
    
    # 等待按键退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 请替换为您自己的图像路径
    image_path = "./ce1.png"
    edge_detection_comparison(image_path)