import torch
import warnings
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW


# 自定义NER Dataset类
class NERDataset(Dataset):
    """自定义数据集类，用于存储NER任务的数据编码。"""

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def read_data(file_path):
    """读取数据文件并进行分词。"""
    # 初始化数据列表
    sentences = []
    labels = []

    # 读取文件并进行分词
    with open(file_path, "r", encoding="utf-8") as file:
        sentence, label = [], []
        for line in file:
            # 去除首尾空格和换行符
            line = line.strip()
            if line == "":
                # 空行代表句子结束，将当前句子和标签加入列表
                sentences.append(sentence)
                labels.append(label)
                sentence, label = [], []
            else:
                # 按空格分token和tag
                tok, tag = line.split(" ")
                sentence.append(tok)
                label.append(tag)
        # 最后一句话
        if sentence:
            sentences.append(sentence)
            labels.append(label)

    return sentences, labels


def build_pytorch(sentences, labels, model_path, max_length=512, stride=64):
    """使用滑动窗口对文章进行分割，并生成模型输入和标签对齐"""
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    label_map = {"O": 0, "B-POLICY_TOOL": 1, "I-POLICY_TOOL": 2, "B-POLICY_GOAL": 3, "I-POLICY_GOAL": 4}

    encodings = {"input_ids": [], "attention_mask": [], "labels": []}
    full_text = []
    full_labels = []

    # 整合所有句子和标签
    for sentence, label in zip(sentences, labels):
        full_text.extend(sentence)
        full_labels.extend([label_map[tag] for tag in label])

    # 对整篇文章应用滑动窗口
    tokenized = tokenizer(full_text, is_split_into_words=True, return_overflowing_tokens=True,
                          max_length=max_length, stride=stride, padding="max_length", truncation=True,
                          return_tensors="pt")

    # 遍历每个窗口并进行标签对齐
    for i in range(len(tokenized["input_ids"])):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # 忽略特殊token
            elif word_idx != prev_word_idx:
                label_ids.append(full_labels[word_idx])  # 取每个词的标签
            else:
                label_ids.append(-100)  # 同一词的子词标记为 -100
            prev_word_idx = word_idx

        # 将窗口的编码和标签添加到encodings中
        encodings["input_ids"].append(tokenized["input_ids"][i])
        encodings["attention_mask"].append(tokenized["attention_mask"][i])
        encodings["labels"].append(torch.tensor(label_ids))

    # 将列表转换为Tensor
    encodings["input_ids"] = torch.stack(encodings["input_ids"])
    encodings["attention_mask"] = torch.stack(encodings["attention_mask"])
    encodings["labels"] = torch.stack(encodings["labels"])

    return encodings, tokenizer


def evaluate(model, data_loader, device, include_o=False):
    """评估模型在验证集上的性能。"""
    model.eval()
    true_labels, pred_labels = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1)

            for i, label in enumerate(labels):
                true_label_ids = label[label != -100].cpu().numpy()
                pred_label_ids = predictions[i][label != -100].cpu().numpy()

                if not include_o:
                    # 过滤掉O标签
                    true_label_ids = [l for l in true_label_ids if l != 0]
                    pred_label_ids = [p for idx, p in enumerate(pred_label_ids) if true_label_ids[idx] != 0]

                true_labels.extend(true_label_ids)
                pred_labels.extend(pred_label_ids)

    # 生成目标名称列表，按需包含O标签
    target_names = ["O", "B-POLICY_TOOL", "I-POLICY_TOOL", "B-POLICY_GOAL", "I-POLICY_GOAL"] if include_o else \
        ["B-POLICY_TOOL", "I-POLICY_TOOL", "B-POLICY_GOAL", "I-POLICY_GOAL"]

    # 显式指定labels以匹配target_names
    labels_list = [0, 1, 2, 3, 4] if include_o else [1, 2, 3, 4]

    # 计算并输出分类报告
    print(classification_report(
        true_labels,
        pred_labels,
        labels=labels_list,
        target_names=target_names,
        zero_division=0  # 防止出现0除错
    ))


# 主训练函数
def train(model, data_loader, optimizer, device):
    """训练模型的主循环。"""
    warnings.filterwarnings("ignore", category=UserWarning)
    model.train()
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    """主函数，执行NER数据的读取、处理和模型训练。"""
    # 读取数据
    file_path = "example.txt"
    sentences, labels = read_data(file_path)

    # 构建PyTorch数据集
    model_path = './pre_model'
    encodings, tokenizer = build_pytorch(sentences, labels, model_path)

    # 定义数据加载器
    dataset = NERDataset(encodings)

    # 划分训练集和测试集（80% 训练，20% 测试）
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 定义数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 定义模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForTokenClassification.from_pretrained(model_path, num_labels=5)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5, no_deprecation_warning=True)

    # 开始训练
    epochs = 3
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train(model, train_loader, optimizer, device)

        # 每个epoch结束后进行评估
        evaluate(model, test_loader, device, True)

    # 保存模型
    model.save_pretrained("./ner_model")
    tokenizer.save_pretrained("./ner_model")
    print("模型已保存到 'ner_model/' 目录")


if __name__ == '__main__':
    main()
