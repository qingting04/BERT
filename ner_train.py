import warnings
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader, random_split
from torchcrf import CRF
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel, AdamW


# 定义模型类 bert-crf
class BertCRF(nn.Module):
    def __init__(self, bert_model_path, num_labels):
        super(BertCRF, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.byte())
            return loss
        else:
            prediction = self.crf.decode(emissions, mask=attention_mask.byte())
            return prediction


# 自定义NER Dataset类
class NERDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])


def read_data(file_path):
    sentences, labels = [], []
    with open(file_path, "r", encoding="utf-8") as file:
        sentence, label = [], []
        for line in file:
            line = line.strip()
            if line == "":
                sentences.append(sentence)
                labels.append(label)
                sentence, label = [], []
            else:
                tok, tag = line.split(" ")
                sentence.append(tok)
                label.append(tag)
        if sentence:
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels


def build_pytorch(sentences, labels, model_path, max_length=512, stride=32):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    label_map = {"O": 0, "B-POLICY_TOOL": 1, "I-POLICY_TOOL": 2, "B-POLICY_GOAL": 3, "I-POLICY_GOAL": 4}
    encodings = {"input_ids": [], "attention_mask": [], "labels": []}
    full_text, full_labels = [], []
    for sentence, label in zip(sentences, labels):
        full_text.extend(sentence)
        full_labels.extend([label_map[tag] for tag in label])
    tokenized = tokenizer(
        full_text, is_split_into_words=True, return_overflowing_tokens=True,
        max_length=max_length, stride=stride, padding="max_length", truncation=True,
        return_tensors="pt"
    )
    for i in tqdm(range(len(tokenized["input_ids"])), desc="Build PyTorch"):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids, prev_word_idx = [], None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(full_labels[word_idx])
            else:
                label_ids.append(-100)
            prev_word_idx = word_idx
        encodings["input_ids"].append(tokenized["input_ids"][i])
        encodings["attention_mask"].append(tokenized["attention_mask"][i])
        encodings["labels"].append(torch.tensor(label_ids))
    encodings["input_ids"] = torch.stack(encodings["input_ids"])
    encodings["attention_mask"] = torch.stack(encodings["attention_mask"])
    encodings["labels"] = torch.stack(encodings["labels"])
    return encodings, tokenizer


def evaluate(model, data_loader, device, include_o=False):
    model.eval()
    true_labels, pred_labels = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            predictions = model(input_ids=input_ids, attention_mask=attention_mask)
            for i, label in enumerate(labels):
                true_label_ids = label[label != -100].cpu().numpy()
                pred_label_ids = predictions[i]
                if not include_o:
                    true_label_ids = [li for li in true_label_ids if li != 0]
                    pred_label_ids = [p for idx, p in enumerate(pred_label_ids) if true_label_ids[idx] != 0]
                true_labels.extend(true_label_ids)
                pred_labels.extend(pred_label_ids)
    target_names = ["O", "B-POLICY_TOOL", "I-POLICY_TOOL", "B-POLICY_GOAL", "I-POLICY_GOAL"] if include_o else \
        ["B-POLICY_TOOL", "I-POLICY_TOOL", "B-POLICY_GOAL", "I-POLICY_GOAL"]
    labels_list = [0, 1, 2, 3, 4] if include_o else [1, 2, 3, 4]
    print(
        classification_report(true_labels, pred_labels, labels=labels_list, target_names=target_names, zero_division=0))


def train(model, data_loader, optimizer, device):
    warnings.filterwarnings("ignore", category=UserWarning)
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    print(f"Average training loss: {avg_loss:.4f}")


def main():
    file_path = "example.txt"
    sentences, labels = read_data(file_path)
    model_path = './pre_model/chinese-roberta'
    encodings, tokenizer = build_pytorch(sentences, labels, model_path)
    dataset = NERDataset(encodings)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertCRF(model_path, num_labels=5).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    epochs = 30
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train(model, train_loader, optimizer, device)
        evaluate(model, test_loader, device, True)
    torch.save(model.state_dict(), "./ner_model/bert_crf_model.pth")
    tokenizer.save_pretrained("./ner_model")
    print("模型已保存到 'ner_model/' 目录")


if __name__ == '__main__':
    main()
