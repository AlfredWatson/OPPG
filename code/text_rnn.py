import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据处理部分
def split_text(text, max_length=512, overlap=50):
    if overlap >= max_length:
        raise ValueError("Overlap must be smaller than max_length.")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

level_mapping = {
    '01': 1, '02': 1, '03': 2, '04': 2, '05': 3, '06': 3,
    '07': 4, '08': 4, '09': 4, '10': 5, '11': 5, '12': 5
}

class ReadabilityDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=50, chunk_size=512, overlap=50):
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
                    label = label - 1
                    with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        chunks = split_text(content, max_length=chunk_size, overlap=overlap)
                        sentences.extend(chunks)
                        labels.extend([label] * len(chunks))
        return sentences, labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        input_ids = self.tokenizer(sentence)
        if len(input_ids) < self.max_length:
            input_ids = input_ids + [0] * (self.max_length - len(input_ids))
        else:
            input_ids = input_ids[:self.max_length]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

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

tokenizer = Tokenizer("bert_pretrain/vocab.txt")

train_dir, dev_dir, test_dir = "yudong/data/train", "yudong/data/dev", "yudong/data/test"
train_dataset = ReadabilityDataset(train_dir, tokenizer, max_length=50, chunk_size=512, overlap=0)
dev_dataset = ReadabilityDataset(dev_dir, tokenizer, max_length=50, chunk_size=512, overlap=0)
test_dataset = ReadabilityDataset(test_dir, tokenizer, max_length=50, chunk_size=512, overlap=0)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 模型定义
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(self, lstm_out):
        attention_scores = self.attention_weights(lstm_out).squeeze(2)
        attention_weights = torch.softmax(attention_scores, dim=1)
        weighted_sum = torch.sum(lstm_out * attention_weights.unsqueeze(2), dim=1)
        return weighted_sum, attention_weights

class TextRNNWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(TextRNNWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.attention = AttentionLayer(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        attention_out, _ = self.attention(lstm_out)
        residual_out = lstm_out[:, -1, :] + attention_out
        output = self.fc(self.dropout(residual_out))
        return output

VOCAB_SIZE = len(tokenizer.word2idx) + 1
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 5
NUM_LAYERS = 2
DROPOUT = 0.5

model = TextRNNWithAttention(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练和验证函数
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for input_ids, labels in tqdm(loader):
        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), acc, f1

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), acc, precision, recall, f1

# 训练模型
EPOCHS = 100
for epoch in range(EPOCHS):
    train_loss, train_acc, train_f1 = train(model, train_loader, optimizer, criterion, device)
    dev_loss, dev_acc, dev_precision, dev_recall, dev_f1 = evaluate(model, dev_loader, criterion, device)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
    print(f"Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.4f}, Precision: {dev_precision:.4f}, Recall: {dev_recall:.4f}, F1: {dev_f1:.4f}")

# 测试模型
test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
