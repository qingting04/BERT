import torch
import hanlp
from transformers import BertTokenizerFast
from ner_train import BertCRF


def load_model_and_tokenizer(model_path, num_labels=5):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertCRF(model_path, num_labels=num_labels)
    model.load_state_dict(torch.load(f"{model_path}/bert_crf_model.pth"))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def tokenize_text(text, tokenizer, max_length=512, stride=64):
    encodings = tokenizer(
        text, is_split_into_words=True, return_tensors="pt",
        padding="max_length", truncation=True,
        max_length=max_length, return_overflowing_tokens=True, stride=stride
    )
    return encodings


def predict_with_sliding_window(text, tokenizer, model, device, max_length=512, stride=64):
    encodings = tokenize_text(text, tokenizer, max_length, stride)
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    predictions = []
    with torch.no_grad():
        for i in range(input_ids.size(0)):
            predicted_labels = model(input_ids[i:i + 1], attention_mask=attention_mask[i:i + 1])
            predictions.append(predicted_labels[0])  # 获取每个窗口的标签预测结果

    merged_predictions = merge_predictions(predictions, stride, max_length)
    return merged_predictions


def merge_predictions(predictions, stride, max_length):
    merged_predictions = []
    for i, pred in enumerate(predictions):
        start_index = i * stride
        if i == 0:
            merged_predictions.extend(pred)
        else:
            merged_predictions.extend(pred[stride:])
    if len(merged_predictions) > max_length:
        merged_predictions = merged_predictions[:max_length]
    return merged_predictions


def main():
    model_path = './ner_model'
    LABEL_MAP = {0: "O", 1: "B-POLICY_TOOL", 2: "I-POLICY_TOOL", 3: "B-POLICY_GOAL", 4: "I-POLICY_GOAL"}
    text = '这是一个例子，用于展示如何使用预训练的BERT模型进行命名实体识别。'
    tokenizer, model, device = load_model_and_tokenizer(model_path)
    tok_h = hanlp.load(hanlp.pretrained.tok.MSR_TOK_ELECTRA_BASE_CRF)
    words = tok_h(text)
    predictions = predict_with_sliding_window(words, tokenizer, model, device)
    decoded_predictions = [LABEL_MAP[label] for label in predictions]
    print(f"Input text: {text}")
    print("Predicted labels:", decoded_predictions)


if __name__ == '__main__':
    main()
