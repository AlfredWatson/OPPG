import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import random

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 设置随机种子

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

level_mapping = {
    '01': 1, '02': 1, '03': 2, '04': 2, '05': 3, '06': 3,
    '07': 4, '08': 4, '09': 4, '10': 5, '11': 5, '12': 5
}

# 数据集定义
class ReadabilityDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=2048, chunk_size=2048, overlap=0):
        # 加载数据
        self.sentences, self.labels = self.load_data(data_dir, chunk_size, overlap)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_data(self, data_dir, chunk_size, overlap):
        sentences, labels = [], []
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                level_key = filename.split('-')[1][:2]
                label = level_mapping.get(level_key)
                if label is not None:
                    label -= 1
                    with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        chunks = self.split_text(content, chunk_size, overlap)
                        if chunks:  # 如果生成了有效的块
                            sentences.extend(chunks)
                            labels.extend([label] * len(chunks))  # 每个分块都有对应标签
                        else:
                            print(f"Warning: No chunks generated for file: {filename}")
        # 检查 sentences 和 labels 是否一致
        assert len(sentences) == len(labels), "Mismatch between sentences and labels"
        return sentences, labels

    @staticmethod
    def split_text(text, max_length, overlap):
        """分块逻辑"""
        if overlap >= max_length:
            raise ValueError("Overlap must be smaller than max_length.")
        return [
            text[i: i + max_length]
            for i in range(0, len(text), max_length - overlap)
        ]

    def __len__(self):
        # 返回数据集长度
        return len(self.sentences)

    def __getitem__(self, idx):
        # 检查 idx 是否越界
        if idx >= len(self):
            raise IndexError(f"Index {idx} is out of range for dataset with length {len(self)}")

        # 获取句子和标签
        sentence = self.sentences[idx]
        label = self.labels[idx]
        # 分词并进行填充/截断
        input_ids = self.tokenizer(sentence)
        if len(input_ids) < self.max_length:
            input_ids += [0] * (self.max_length - len(input_ids))
        else:
            input_ids = input_ids[:self.max_length]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# 分词器
class Tokenizer:
    def __init__(self, vocab_file="vocab.txt"):
        self.word2idx = self.build_vocab(vocab_file)

    def build_vocab(self, vocab_file):
        word2idx = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                word = line.strip()
                word2idx[word] = i + 1
        return word2idx

    def __call__(self, sentence):
        return [self.word2idx.get(word, 0) for word in sentence]

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import random

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 设置随机种子

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import os
import torch
from torch.utils.data import Dataset

level_mapping = {
    '01': 1, '02': 1, '03': 2, '04': 2, '05': 3, '06': 3,
    '07': 4, '08': 4, '09': 4, '10': 5, '11': 5, '12': 5
}

class ReadabilityDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512):
        # 加载完整文本数据
        self.sentences, self.labels = self.load_data(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_data(self, data_dir):
        """加载完整的文本数据"""
        sentences, labels = [], []
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                # 提取文件的等级标签
                level_key = filename.split('-')[1][:2]
                label = level_mapping.get(level_key)
                if label is not None:
                    label -= 1  # 调整为从 0 开始的标签
                    with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:  # 检查内容非空
                            sentences.append(content)
                            labels.append(label)
                        else:
                            print(f"Warning: Empty file detected: {filename}")
        # 检查数据长度一致性
        assert len(sentences) == len(labels), "Mismatch between sentences and labels"
        print(f"Loaded {len(sentences)} samples from {data_dir}")
        return sentences, labels

    def __len__(self):
        # 数据集样本数
        return len(self.sentences)

    def __getitem__(self, idx):
        # 获取文本和标签
        sentence = self.sentences[idx]
        label = self.labels[idx]
        # 对文本进行分词
        input_ids = self.tokenizer(sentence)
        # 对超长文本进行截断，对不足的进行填充
        if len(input_ids) < self.max_length:
            input_ids += [0] * (self.max_length - len(input_ids))
        else:
            input_ids = input_ids[:self.max_length]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# 示例分词器
class Tokenizer:
    def __init__(self, vocab_file="vocab.txt"):
        self.word2idx = self.build_vocab(vocab_file)

    def build_vocab(self, vocab_file):
        word2idx = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                word = line.strip()
                word2idx[word] = i + 1
        return word2idx

    def __call__(self, sentence):
        # 将句子中的词映射到词表索引
        return [self.word2idx.get(word, 0) for word in sentence]


# 模型定义
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x).unsqueeze(1)
        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(c, dim=2)[0] for c in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


# 训练和验证函数
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for input_ids, labels in tqdm(loader, desc="Training"):
        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), accuracy, f1


def evaluate(model, loader, criterion, device, detailed=False):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for input_ids, labels in tqdm(loader, desc="Evaluating"):
            input_ids, labels = input_ids.to(device), labels.to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    if detailed:
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds))
    return total_loss / len(loader), accuracy, precision, recall, f1

# 设置数据与参数
tokenizer = Tokenizer("bert_pretrain/vocab.txt")
VOCAB_SIZE = len(tokenizer.word2idx) + 1
EMBEDDING_DIM = 300
NUM_FILTERS = 256
FILTER_SIZES = [2, 3, 4, 5]
OUTPUT_DIM = 5
DROPOUT = 0.5
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64

# 初始化模型
model = TextCNN(VOCAB_SIZE, EMBEDDING_DIM, NUM_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=10)
scaler = torch.cuda.amp.GradScaler()

# 加载数据
train_dataset = ReadabilityDataset("yudong/data/train", tokenizer)
dev_dataset = ReadabilityDataset("yudong/data/dev", tokenizer)
test_dataset = ReadabilityDataset("yudong/data/test", tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 开始训练
for epoch in range(EPOCHS):
    train_loss, train_acc, train_f1 = train(model, train_loader, optimizer, criterion, device)
    dev_loss, dev_acc, dev_prec, dev_rec, dev_f1 = evaluate(model, dev_loader, criterion, device)
    scheduler.step()
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
    print(f"Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.4f}, Dev Precision: {dev_prec:.4f}, Dev Recall: {dev_rec:.4f}, Dev F1: {dev_f1:.4f}")

# 测试集评估
test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion, device, detailed=True)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Precision: {test_prec:.4f}, Test Recall: {test_rec:.4f}, Test F1: {test_f1:.4f}")


