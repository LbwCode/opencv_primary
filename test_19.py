import numpy as np
import cv2

# 加载已训练的KNN模型
knn = cv2.ml.KNearest_create()
knn = knn.load('knn_model.xml')

# 读取测试图像并转换为灰度图
image_path = 'num_img.png'  # 请替换为你的测试图像路径
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 假设图像包含多个20x20的数字，将其分割为单个数字
# 这里简单假设图像是50x100的网格，每个单元格20x20
cells = []
rows = np.vsplit(gray, 50)
for row in rows:
    columns = np.hsplit(row, 100)
    for col in columns:
        cells.append(col)

# 转换为NumPy数组并重塑为模型所需的形状
test_data = np.array(cells).reshape(-1, 400).astype(np.float32)

# 使用KNN模型进行预测
ret, result, neighbours, dist = knn.findNearest(test_data, k=5)



# 可视化每个预测结果（放大到300x300像素）
for i in range(len(test_data)):
    # 重塑为20x20图像
    sample_img = test_data[i].reshape(20, 20).astype(np.uint8)
    
    # 放大到300x300像素
    enlarged_img = cv2.resize(sample_img, (300, 300), interpolation=cv2.INTER_NEAREST)
    
    # 添加预测结果数字（仅显示数字）
    cv2.putText(
        enlarged_img, 
        f"{int(result[i])}",
        (140, 200),  # 数字显示位置（居中）
        cv2.FONT_HERSHEY_SIMPLEX,
        3.0,         # 字体大小
        (255),       # 文本颜色（白色，因图像为灰度图）
        3            # 文本线宽
    )
    
    # 添加样本编号（仅数字）
    cv2.putText(
        enlarged_img, 
        f"{i+1}/{len(test_data)}",
        (10, 30),    # 文本位置（左上角）
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,         # 字体大小
        (255),       # 文本颜色
        1            # 文本线宽
    )
    
    # 显示图像（按任意键继续查看下一个）
    cv2.imshow("Digit Prediction", enlarged_img)
    key = cv2.waitKey(0)  # 按任意键继续
    
    # 按ESC键或'q'键退出循环
    if key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()
