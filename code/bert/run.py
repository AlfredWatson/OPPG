# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train
from importlib import import_module
import random
# from utils import build_dataset, build_iterator, get_time_dif
from utils_yudong import build_dataset, build_iterator, get_time_dif
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn

if __name__ == '__main__':
    time_start = time.time()
    #x修改的hsk_all
    dataset = 'yudong'  # 数据集

    model_name = 'bert'  # bert
    # linguistic_where = 'self-attention' # bert-embedding or self-attention
    # level = 'character-word-grammar'
    print('---Text Difficulty Classification for ' + dataset + '---')
    print('---Using ' + model_name + '---')
    x = import_module(model_name)
    config = x.Config(dataset)

    new_seed = 42  # 改成其他值
    random.seed(new_seed)
    np.random.seed(new_seed)
    torch.manual_seed(new_seed)
    torch.cuda.manual_seed_all(new_seed)
    torch.backends.cudnn.deterministic = True  # 如果需要可复现性，保持不变

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    # train
    model = x.Model(config).to(config.device)

    #+++++++++++++++++++
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        # 使用 DataParallel 来实现多 GPU 计算
        model = nn.DataParallel(model)
    #+++++++++++++++++++
    train(config, model, train_iter, dev_iter, test_iter)

    time_end = time.time()
    time_sum = time_end - time_start
    print('Finish! Using %.2fs' % time_sum)