import os
import torch
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
from datetime import timedelta
import time
from char_word_gra_level import new_character, new_word, new_grammar, level_list2matrix

PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'  # BERT符号


def load_data(data_dir, config):
    """加载新数据集"""
    sentences, labels = [], []
    level_mapping = {
        '01': 1, '02': 1, '03': 2, '04': 2, '05': 3, '06': 3,
        '07': 4, '08': 4, '09': 4, '10': 5, '11': 5, '12': 5
    }
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            try:
                level_key = filename.split('-')[1][:2]
            except IndexError:
                print(f"Skipping invalid filename: {filename}")
                continue
            label = level_mapping.get(level_key)
            if label is not None:
                # label -= 1  # 调整为从 0 开始的标签
                with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        sentences.append(content)
                        labels.append(label)
    print(f"Loaded {len(sentences)} samples from {data_dir}")
    return sentences, labels
#
# preprocess_call_count = 0
# def preprocess_sample(content, label, config):
#     """对单个样本进行预处理"""
#     global preprocess_call_count
#     preprocess_call_count += 1  # Increment the counter
#     print(f"preprocess_sample has been called {preprocess_call_count} times.")
#     tokenizer = config.tokenizer
#     pad_size = config.pad_size
#
#     # Tokenization
#     token = tokenizer.tokenize(content)
#
#     # Embedding Layer Processing
#     if config.with_linguistic_information_embedding_layer:
#         character_level_ids = new_character(token)
#         word_level_ids = new_word(content, token)
#         grammar_level_ids = new_grammar(content, token, only_max=config.only_max)
#
#         # Padding for Embedding Layer
#         character_level_ids = [0] + character_level_ids + [0]
#         word_level_ids = [0] + word_level_ids + [0]
#         # grammar_level_ids = [0] + grammar_level_ids + [0]
#         if len(token) + 2 < pad_size:
#             character_level_ids += [0] * (pad_size - len(character_level_ids))
#             word_level_ids += [0] * (pad_size - len(word_level_ids))
#             grammar_level_ids += [0] * (pad_size - len(grammar_level_ids))
#         else:
#             character_level_ids = character_level_ids[:pad_size]
#             word_level_ids = word_level_ids[:pad_size]
#             grammar_level_ids = grammar_level_ids[:pad_size]
#
#     # Self-Attention Layer Processing
#     if config.with_linguistic_information_selfattention_layer:
#         character_level_matrix = level_list2matrix(content, token, 'character')
#         word_level_matrix = level_list2matrix(content, token, 'word')
#         grammar_level_matrix = level_list2matrix(content, token, 'grammar')
#
#         # Padding for Attention Layer
#         character_level_matrix = np.pad(character_level_matrix, ((1, 1), (1, 1)))
#         word_level_matrix = np.pad(word_level_matrix, ((1, 1), (1, 1)))
#         grammar_level_matrix = np.pad(grammar_level_matrix, ((1, 1), (1, 1)))
#         if len(token) + 2 < pad_size:
#             character_level_matrix = np.pad(character_level_matrix,
#                                             ((0, pad_size - len(token) - 2), (0, pad_size - len(token) - 2)))
#             word_level_matrix = np.pad(word_level_matrix,
#                                        ((0, pad_size - len(token) - 2), (0, pad_size - len(token) - 2)))
#             grammar_level_matrix = np.pad(grammar_level_matrix,
#                                           ((0, pad_size - len(token) - 2), (0, pad_size - len(token) - 2)))
#         else:
#             character_level_matrix = character_level_matrix[:pad_size, :pad_size]
#             word_level_matrix = word_level_matrix[:pad_size, :pad_size]
#             grammar_level_matrix = grammar_level_matrix[:pad_size, :pad_size]
#
#     # Token IDs
#     token = [CLS] + token + [SEP]
#     token_ids = tokenizer.convert_tokens_to_ids(token)
#     seq_len = len(token_ids)
#     mask = [1] * seq_len
#
#     if len(token_ids) < pad_size:
#         mask += [0] * (pad_size - len(token_ids))
#         token_ids += [0] * (pad_size - len(token_ids))
#     else:
#         mask = mask[:pad_size]
#         token_ids = token_ids[:pad_size]
#         seq_len = pad_size
#
#     # Append based on configurations
#     if config.with_linguistic_information_embedding_layer and not config.with_linguistic_information_selfattention_layer:
#         return token_ids, seq_len, mask, character_level_ids, word_level_ids,grammar_level_ids, label
#     elif config.with_linguistic_information_selfattention_layer and not config.with_linguistic_information_embedding_layer:
#         return token_ids, seq_len, mask, character_level_matrix, word_level_matrix, grammar_level_matrix, label
#     elif config.with_linguistic_information_embedding_layer and config.with_linguistic_information_selfattention_layer:
#         return token_ids, seq_len, mask, character_level_ids, word_level_ids, grammar_level_ids, character_level_matrix, word_level_matrix, grammar_level_matrix, label
#     else:
#         return token_ids,seq_len, mask,label



preprocess_call_count = 0
def preprocess_sample(content, label, config):
    """对单个样本进行预处理"""
    global preprocess_call_count
    preprocess_call_count += 1
    print(f"preprocess_sample has been called {preprocess_call_count} times.")
    tokenizer = config.tokenizer
    pad_size = config.pad_size

    # Tokenization
    token = tokenizer.tokenize(content)

    # Embedding Layer Processing
    if config.with_linguistic_information_embedding_layer:
        character_level_ids = new_character(token)
        word_level_ids = new_word(content, token)

        # Padding for Embedding Layer
        character_level_ids = [0] + character_level_ids + [0]
        word_level_ids = [0] + word_level_ids + [0]
        if len(token) + 2 < pad_size:
            character_level_ids += [0] * (pad_size - len(character_level_ids))
            word_level_ids += [0] * (pad_size - len(word_level_ids))
        else:
            character_level_ids = character_level_ids[:pad_size]
            word_level_ids = word_level_ids[:pad_size]

    # Self-Attention Layer Processing
    if config.with_linguistic_information_selfattention_layer:
        character_level_matrix = level_list2matrix(content, token, 'character')
        word_level_matrix = level_list2matrix(content, token, 'word')

        # Padding for Attention Layer
        character_level_matrix = np.pad(character_level_matrix, ((1, 1), (1, 1)))
        word_level_matrix = np.pad(word_level_matrix, ((1, 1), (1, 1)))
        if len(token) + 2 < pad_size:
            character_level_matrix = np.pad(character_level_matrix,
                                            ((0, pad_size - len(token) - 2), (0, pad_size - len(token) - 2)))
            word_level_matrix = np.pad(word_level_matrix,
                                       ((0, pad_size - len(token) - 2), (0, pad_size - len(token) - 2)))
        else:
            character_level_matrix = character_level_matrix[:pad_size, :pad_size]
            word_level_matrix = word_level_matrix[:pad_size, :pad_size]

    # Token IDs
    token = [CLS] + token + [SEP]
    token_ids = tokenizer.convert_tokens_to_ids(token)
    seq_len = len(token_ids)
    mask = [1] * seq_len

    if len(token_ids) < pad_size:
        mask += [0] * (pad_size - len(token_ids))
        token_ids += [0] * (pad_size - len(token_ids))
    else:
        mask = mask[:pad_size]
        token_ids = token_ids[:pad_size]
        seq_len = pad_size

    # Append based on configurations
    if config.with_linguistic_information_embedding_layer and not config.with_linguistic_information_selfattention_layer:
        return token_ids, seq_len, mask, character_level_ids, word_level_ids, label
    elif config.with_linguistic_information_selfattention_layer and not config.with_linguistic_information_embedding_layer:
        return token_ids, seq_len, mask, character_level_matrix, word_level_matrix, label
    elif config.with_linguistic_information_embedding_layer and config.with_linguistic_information_selfattention_layer:
        return token_ids, seq_len, mask, character_level_ids, word_level_ids, character_level_matrix, word_level_matrix, label
    else:
        return token_ids, seq_len, mask, label


def build_dataset(config):
    """加载训练、验证、测试集"""
    # train_sentences, train_labels = load_data(config.train_path, config)
    # dev_sentences, dev_labels = load_data(config.dev_path, config)
    test_sentences, test_labels = load_data(config.test_path, config)

    # 构造样本
    # train_data = [preprocess_sample(content, label, config) for content, label in zip(train_sentences, train_labels)]
    # dev_data = [preprocess_sample(content, label, config) for content, label in zip(dev_sentences, dev_labels)]
    test_data = [preprocess_sample(content, label, config) for content, label in zip(test_sentences, test_labels)]

    # return train_data, dev_data, test_data
    return test_sentences, test_labels,test_data


#
# class DatasetIterater:
#     def __init__(self, dataset, batch_size, device):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.device = device
#         self.n_batches = len(dataset) // batch_size
#         self.residue = len(dataset) % batch_size != 0
#         self.index = 0
#
#     def _to_tensor(self, batch):
#         data = [self.dataset[i] for i in batch]
#         if len(data[0]) == 4:
#             x, seq_len, mask, labels = zip(*data)
#             return (torch.tensor(x).to(self.device),
#                     torch.tensor(seq_len).to(self.device),
#                     torch.tensor(mask).to(self.device)), torch.tensor(labels).to(self.device)
#         elif len(data[0]) == 7:
#             x, seq_len, mask, c_level, w_level, g_level, labels = zip(*data)
#             return (torch.tensor(x).to(self.device),
#                     torch.tensor(seq_len).to(self.device),
#                     torch.tensor(mask).to(self.device),
#                     torch.tensor(c_level).to(self.device),
#                     torch.tensor(w_level).to(self.device),
#                     torch.tensor(g_level).to(self.device)), torch.tensor(labels).to(self.device)
#         elif len(data[0]) == 10:
#             x, seq_len, mask, c_ids, w_ids, g_ids, c_mat, w_mat, g_mat, labels = zip(*data)
#             return (torch.tensor(x).to(self.device),
#                     torch.tensor(seq_len).to(self.device),
#                     torch.tensor(mask).to(self.device),
#                     torch.tensor(c_ids).to(self.device),
#                     torch.tensor(w_ids).to(self.device),
#                     torch.tensor(g_ids).to(self.device),
#                     torch.tensor(c_mat).to(self.device),
#                     torch.tensor(w_mat).to(self.device),
#                     torch.tensor(g_mat).to(self.device)), torch.tensor(labels).to(self.device)
#
#     def __next__(self):
#         if self.index < self.n_batches:
#             start = self.index * self.batch_size
#             end = start + self.batch_size
#             batch = self._to_tensor(range(start, end))
#             self.index += 1
#             return batch
#         elif self.residue and self.index == self.n_batches:
#             batch = self._to_tensor(range(self.index * self.batch_size, len(self.dataset)))
#             self.index += 1
#             return batch
#         else:
#             self.index = 0
#             raise StopIteration
#
#     def __iter__(self):
#         return self
#
#     def __len__(self):
#         return self.n_batches + int(self.residue)
#
#
# def build_iterator(dataset, config):
#     return DatasetIterater(dataset, config.batch_size, config.device)
#
#
# def get_time_dif(start_time):
#     end_time = time.time()
#     return timedelta(seconds=int(round(end_time - start_time)))



class DatasetIterater:
    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.n_batches = len(dataset) // batch_size
        self.residue = len(dataset) % batch_size != 0
        self.index = 0

    def _to_tensor(self, batch):
        data = [self.dataset[i] for i in batch]
        if len(data[0]) == 4:
            x, seq_len, mask, labels = zip(*data)
            return (torch.tensor(x).to(self.device),
                    torch.tensor(seq_len).to(self.device),
                    torch.tensor(mask).to(self.device)), torch.tensor(labels).to(self.device)
        elif len(data[0]) == 6:
            x, seq_len, mask, c_level, w_level, labels = zip(*data)
            return (torch.tensor(x).to(self.device),
                    torch.tensor(seq_len).to(self.device),
                    torch.tensor(mask).to(self.device),
                    torch.tensor(c_level).to(self.device),
                    torch.tensor(w_level).to(self.device)), torch.tensor(labels).to(self.device)
        elif len(data[0]) == 8:
            x, seq_len, mask, c_ids, w_ids, c_mat, w_mat, labels = zip(*data)
            return (torch.tensor(x).to(self.device),
                    torch.tensor(seq_len).to(self.device),
                    torch.tensor(mask).to(self.device),
                    torch.tensor(c_ids).to(self.device),
                    torch.tensor(w_ids).to(self.device),
                    torch.tensor(c_mat).to(self.device),
                    torch.tensor(w_mat).to(self.device)), torch.tensor(labels).to(self.device)

    def __next__(self):
        if self.index < self.n_batches:
            start = self.index * self.batch_size
            end = start + self.batch_size
            batch = self._to_tensor(range(start, end))
            self.index += 1
            return batch
        elif self.residue and self.index == self.n_batches:
            batch = self._to_tensor(range(self.index * self.batch_size, len(self.dataset)))
            self.index += 1
            return batch
        else:
            self.index = 0
            raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches + int(self.residue)


def build_iterator(dataset, config):
    return DatasetIterater(dataset, config.batch_size, config.device)


def get_time_dif(start_time):
    end_time = time.time()
    return timedelta(seconds=int(round(end_time - start_time)))
