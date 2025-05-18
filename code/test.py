import os
import json
from sklearn import metrics

def calculate_metrics(data_folder):
    """
    计算预测结果的准确率、召回率、精确率、F1，以及混淆矩阵。

    :param data_folder: 包含 JSON 文件的文件夹路径
    :return: (accuracy, classification_report, confusion_matrix)
    """
    true_labels = []
    predicted_labels = []

    # 遍历文件夹中的所有 JSON 文件
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.json'):  # 仅处理 JSON 文件
            file_path = os.path.join(data_folder, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    # 提取真实标签
                    true_labels.append(int(data['label']))
                    # 处理预测标签
                    final_degree = data.get('final_degree')
                    if final_degree and isinstance(final_degree, str) and final_degree.isdigit():
                        predicted_labels.append(int(final_degree))
                    else:
                        # 打印出错的文件名
                        print(f"文件 {file_name} 的 'final_degree' 不合法或缺失，跳过此文件")
                        continue
            except (ValueError, TypeError) as e:
                print(f"文件 {file_name} 出现错误：{e}")
                continue

    # 如果没有合法的预测值，则无法计算指标
    if not true_labels or not predicted_labels:
        print("没有足够的数据来计算指标")
        return None, None, None

    # 计算准确率
    accuracy = metrics.accuracy_score(true_labels, predicted_labels)
    # 分类报告
    classification_report = metrics.classification_report(true_labels, predicted_labels, digits=4)
    # 混淆矩阵
    confusion_matrix = metrics.confusion_matrix(true_labels, predicted_labels)

    # 打印结果
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report)
    print("\nConfusion Matrix:")
    print(confusion_matrix)

    return accuracy, classification_report, confusion_matrix

# 示例用法
# 请将 "output" 替换为你的 JSON 文件夹路径
calculate_metrics("/home/wp/item/debate/Chapter/output")
