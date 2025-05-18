import tqdm
import torch
from openprompt.data_utils.utils import InputExample
from sklearn.metrics import *
import argparse
import os
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer, PtuningTemplate
from openprompt.prompts import ManualTemplate
from openprompt.plms import load_plm
from transformers import AdamW, get_linear_schedule_with_warmup

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# 参数配置
parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=5)
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model_name_or_path", default='chinese-roberta-wwm-ext')
parser.add_argument("--max_epochs", type=int, default=10)
parser.add_argument("--learning_rate", default=4e-5, type=float)
parser.add_argument("--batch_size", default=16, type=int)
args = parser.parse_args()

# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(args.seed)

# 数据映射
level_mapping = {
    '01': 1, '02': 1, '03': 2, '04': 2, '05': 3, '06': 3,
    '07': 4, '08': 4, '09': 4, '10': 5, '11': 5, '12': 5
}



# 数据加载类
class ReadabilityDataset:
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.sentences, self.labels = self.load_data(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self.create_examples()

    def load_data(self, data_dir):
        """加载完整文本数据"""
        sentences, labels = [], []
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                level_key = filename.split('-')[1][:2]
                label = level_mapping.get(level_key)
                if label is not None:
                    label -= 1  # 从 0 开始的标签
                    with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            sentences.append(content)
                            labels.append(label)
                        else:
                            print(f"Warning: Empty file: {filename}")
        assert len(sentences) == len(labels), "Mismatch between sentences and labels"
        print(f"Loaded {len(sentences)} samples from {data_dir}")
        return sentences, labels

    def create_examples(self):
        examples = []
        for i, (sentence, label) in enumerate(zip(self.sentences, self.labels)):
            examples.append(InputExample(text_a=sentence, label=label, guid=i))
        return examples

# 加载预训练语言模型
plm, tokenizer, model_config, WrapperClass = load_plm('bert', args.model_name_or_path)

# 加载数据集
train_dataset = ReadabilityDataset("yudong/data/train", tokenizer, max_length=512)
dev_dataset = ReadabilityDataset("yudong/data/dev", tokenizer, max_length=512)
test_dataset = ReadabilityDataset("yudong/data/test", tokenizer, max_length=512)

class_labels = ['1', '2', '3', '4', '5']
batch_s = args.batch_size

# 模板和 Verbalizer
mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer).from_file(f"scripts/ptuning_template.txt", choice=0)
myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, pred_temp=1.0).from_file(
    path=f"scripts/verbalizer_yudong.txt")
from openprompt import PromptForClassification
# Prompt 模型
use_cuda = torch.cuda.is_available()
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False,
                                       plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model = prompt_model.cuda()

# DataLoader 设置
train_dataloader = PromptDataLoader(dataset=train_dataset.examples, template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=512,
                                    batch_size=batch_s, shuffle=True, truncate_method="tail")

validation_dataloader = PromptDataLoader(dataset=dev_dataset.examples, template=mytemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=512,
                                         batch_size=batch_s, shuffle=False, truncate_method="tail")

test_dataloader = PromptDataLoader(dataset=test_dataset.examples, template=mytemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=512,
                                   batch_size=batch_s, shuffle=False, truncate_method="tail")

# 评估函数
def evaluate(prompt_model, dataloader, desc):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    pbar = tqdm.tqdm(dataloader, desc=desc)
    for step, inputs in enumerate(pbar):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    acc = accuracy_score(alllabels, allpreds)
    pre = precision_score(alllabels, allpreds, average=None)
    recall = recall_score(alllabels, allpreds, average=None)
    f1 = f1_score(alllabels, allpreds, average=None)
    report = classification_report(alllabels, allpreds, target_names=class_labels, digits=4)

    print("Accuracy:", acc)
    print("\nClassification Report:\n", report)
    return acc, pre, recall, f1

# 优化器和调度器
loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
tot_step = len(train_dataloader) * args.max_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=tot_step)

# 训练和验证
best_val_acc = 0
for epoch in range(args.max_epochs):
    prompt_model.train()
    tot_loss = 0
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
        tot_loss += loss.item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    val_acc, val_pre, val_recall, val_f1 = evaluate(prompt_model, validation_dataloader, desc="Validation")
    if val_acc > best_val_acc:
        torch.save(prompt_model.state_dict(), f"saved_dict/best_prompt_model.ckpt")
        best_val_acc = val_acc
    print(f"Epoch {epoch + 1}, Validation Accuracy: {val_acc:.4f}")

# 加载最佳模型并测试
prompt_model.load_state_dict(torch.load(f"saved_dict/best_prompt_model.ckpt"))
test_acc, test_pre, test_recall, test_f1 = evaluate(prompt_model, test_dataloader, desc="Test")
print(f"Test Accuracy: {test_acc:.4f}")
