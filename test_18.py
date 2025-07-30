import numpy as np
import cv2

# 加载手写数字图像（假设图像包含5000个20×20像素的手写数字）
img = cv2.imread('num_img.png')

# 将彩色图像转换为灰度图（简化处理，降低维度）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 数据分割：将图像切割为5000个20×20的单元格
# np.vsplit(gray, 50)：垂直方向分割为50行（每行高度为20像素）
# 对每行再用np.hsplit分割为100列（每列宽度为20像素）
# cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

cells = []
# 垂直分割图像为50行
rows = np.vsplit(gray, 50)
for row in rows:
    # 对每行水平分割为100列
    columns = np.hsplit(row, 100)
    cells.append(columns)

# 转换为NumPy数组，形状为 (50行, 100列, 20高, 20宽)
x = np.array(cells)
print(x.shape)

# 显示单个图
# for i in range(50):
#     for j in range(100): 
#         cell = x[i][j]
#         cell = cv2.resize(cell, (300, 300))
#         cv2.imshow(f'Cell ({i},{j})', cell)
#         cv2.waitKey(0)
#         cv2.destroyWindow(f'Cell ({i},{j})')



# 划分训练集和测试集：
# 前50列作为训练集，后50列作为测试集
# reshape(-1, 400)：将每个20×20的图像展平为400维向量（20×20=400）
# astype(np.float32)：转换为float32类型（符合OpenCV的KNN输入要求）
train = x[:, :50].reshape(-1, 400).astype(np.float32)  # 训练集：2500个样本（50×50）
test = x[:, 50:100].reshape(-1, 400).astype(np.float32)  # 测试集：2500个样本（50×50）

# 创建标签：0-9每个数字各250个样本（与训练数据对应）
k = np.arange(10)  # 生成0-9的数组
train_labels = np.repeat(k, 250)[:, np.newaxis]  # 重复每个数字250次，形状为(2500, 1)
test_labels = train_labels.copy()  # 测试标签与训练标签结构一致

# 初始化KNN模型
knn = cv2.ml.KNearest_create()

# 训练模型
# 参数说明：
# - train：训练数据（每行一个样本）
# - cv2.ml.ROW_SAMPLE：指定样本按行组织
# - train_labels：训练数据对应的标签
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

# 使用测试集进行预测（k=5表示取5个最近邻样本投票）
# 返回值：
# - ret：预测结果的置信度（此处未使用）
# - result：预测的标签（形状与test_labels一致）
# - neighbours：每个样本的k个最近邻的标签
# - dist：每个样本与k个最近邻的距离
ret, result, neighbours, dist = knn.findNearest(test, k=5)

# 计算模型准确率
matches = result == test_labels  # 比较预测结果与真实标签（布尔数组）
correct = np.count_nonzero(matches)  # 统计正确预测的数量
accuracy = correct * 100.0 / result.size  # 计算准确率（百分比）
print(f"模型准确率: {accuracy:.2f}%")

# 保存训练数据和测试数据（用于后续复用）
np.savez('knn_data.npz', train=train, train_labels=train_labels, test=test)

# 验证数据保存是否成功（加载数据并检查）
with np.load('knn_data.npz') as data:
    print("保存的数据文件包含：", data.files)  # 输出保存的变量名
    loaded_train = data['train']
    loaded_train_labels = data['train_labels']
    loaded_test = data['test']
    
    # 验证数据形状是否正确
    print(f"加载的训练数据形状: {loaded_train.shape}（预期：(2500, 400)）")
    print(f"加载的测试数据形状: {loaded_test.shape}（预期：(2500, 400)）")

# 保存训练好的KNN模型（方便后续直接使用，无需重新训练）
knn.save('knn_model.xml')



# 可视化部分测试结果（可选功能）
def visualize_predictions(num_samples=10):
    """可视化前num_samples个测试样本的预测结果"""
    for i in range(num_samples):
        # 将400维向量重塑为20×20的图像
        sample_img = test[i].reshape(20, 20).astype(np.uint8)  # 转换为uint8以正常显示
        
        # 在图像上添加预测结果标签
        cv2.putText(
            sample_img, 
            f"Pred: {int(result[i])}",  # 预测的数字
            (2, 18),  # 文本位置（左上角）
            cv2.FONT_HERSHEY_SIMPLEX,  # 字体
            0.6,  # 字体大小
            (255),  # 文本颜色（白色，因图像为灰度图）
            1  # 文本线宽
        )
        
        # 显示图像
        cv2.imshow(f"测试样本 #{i+1}", sample_img)
        cv2.waitKey(1000)  # 每个样本显示1秒
    cv2.destroyAllWindows()  # 关闭所有窗口

# 取消下面一行的注释以运行可视化功能
# visualize_predictions(10)



