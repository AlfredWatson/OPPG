# coding: UTF-8
import torch
import numpy as np
from importlib import import_module
# from utils import build_dataset, build_iterator, get_time_dif
from utils_yudong import build_dataset, build_iterator, get_time_dif
import time
import torch.nn as nn
from sklearn import metrics
import warnings

warnings.filterwarnings("ignore")

from collections import OrderedDict

def add_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # 如果没有 "module." 前缀，则加上
        if not k.startswith("module."):
            new_state_dict["module." + k] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

# 定义测试函数
def test(config, model, test_iter, witch_epoch, print_all=True):
    print('Test Report for ' + witch_epoch)
    # 加载指定epoch的模型参数
    # model.load_state_dict(torch.load(config.save_path + witch_epoch))
    #
    # model.load_state_dict(torch.load(witch_epoch))
    #+++++++++++
    state_dict = torch.load(witch_epoch)
    # 为参数名称加上 "module." 前缀
    state_dict = add_module_prefix(state_dict)
    # 加载到模型中
    model.load_state_dict(state_dict)
    #+++++++++++++++++++++

    model.eval()
    start_time = time.time()

    # 调用评估函数，获取测试结果
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)

    # 打印测试损失和准确率
    msg = 'Test Loss: {0:>5.4},  Test Acc: {1:>6.4%}'
    print(msg.format(test_loss, test_acc))

    # 如果print_all为True，打印详细的评估报告
    if print_all:
        print("Precision, Recall and F1-Score...")
        print(test_report)
        print("Confusion Matrix...")
        print(test_confusion)

    # 打印时间消耗
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


# 定义评估函数
def evaluate(config, model, data_iter, test=False):
    model.eval()  # 设置模型为评估模式
    loss_total = 0  # 累计损失
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    # 禁用梯度计算（加快计算速度和节省显存）
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            labels = labels-1
            loss = nn.functional.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    # 计算准确率
    acc = metrics.accuracy_score(labels_all, predict_all)

    # 如果是测试阶段，返回详细的报告和混淆矩阵
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion

    # 否则只返回准确率和损失
    return acc, loss_total / len(data_iter)


if __name__ == '__main__':
    # 数据集和模型配置
    dataset = 'yudong'  # 数据集名称
    model_name = 'bert'  # 使用的模型名称

    print('---Text Difficulty Classification for ' + dataset + '---')
    print('---Using ' + model_name + '---')

    x = import_module(model_name)
    config = x.Config(dataset)

    print("Loading data...")
    # 加载数据集
    train_data, dev_data, test_data = build_dataset(config)
    test_iter = build_iterator(test_data, config)

    # 定义模型并加载到设备
    model = x.Model(config).to(config.device)

    # 如果使用多GPU，启用DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # 进行测试，指定保存的模型文件名（如 "best_model.pth"）
    best_model_path = "/home/wp/item/yu/saved_dict/results/bert-ling-0.682/model_12e.ckpt"
    # best_model_path = "./best_models_ckpt/mlf_bert/best_model.ckpt"
    test(config, model, test_iter, witch_epoch=best_model_path)
