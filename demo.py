# -*- coding: utf-8 -*-

import torch
import pkuseg
from transformers import BertTokenizerFast, BertForTokenClassification

'''# 加载模型和分词器
model_path = "./ner_model"  # 替换为您的模型保存路径
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForTokenClassification.from_pretrained(model_path)
model.eval()  # 将模型设置为评估模式

# 将模型移到适当的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 准备输入数据
word = "这是一个例子，用于展示如何使用预训练的BERT模型进行命名实体识别。"
tok = pkuseg.pkuseg('./tok_model')  # 加载分词器
words = tok.cut(word)  # 分词
inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt")


# 将输入数据移到适当的设备
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# 进行推理
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

# 处理输出
predictions = torch.argmax(logits, dim=-1)

# 获取标签映射
label_map = {0: "O", 1: "B-POLICY_TOOL", 2: "I-POLICY_TOOL", 3: "B-POLICY_GOAL", 4: "I-POLICY_GOAL"}

# 将预测结果转换为标签
predicted_labels = [label_map[label.item()] for label in predictions[0]]

# 打印结果
print(f"Input text: {word}")
print("Predicted labels:", predicted_labels)'''


def sliding_window_predictions(text, tokenizer, model, max_length=512, stride=64):
    # 对输入文本进行编码
    encodings = tokenizer(text, is_split_into_words=True, return_tensors="pt", padding="max_length", truncation=True,
                          max_length=max_length, return_overflowing_tokens=True, stride=stride)

    # 获取每个窗口的输入和注意力掩码
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # 将模型设置为评估模式
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(input_ids.size(0)):
            # 将输入送入模型进行预测
            outputs = model(input_ids[i:i + 1], attention_mask=attention_mask[i:i + 1])
            logits = outputs.logits

            # 获取预测标签
            predicted_labels = torch.argmax(logits, dim=-1).squeeze().tolist()
            predictions.append(predicted_labels)

    return predictions


def merge_predictions(predictions, stride, max_length):
    merged_predictions = []
    for i, pred in enumerate(predictions):
        # 计算当前窗口的起始位置
        start_index = i * stride

        # 如果是第一个窗口，直接添加预测
        if i == 0:
            merged_predictions.extend(pred)
        else:
            # 从第二个窗口开始，处理重叠部分
            merged_predictions.extend(pred[max(0, stride):])  # 忽略前面的重叠部分

    # 去掉多余的标签，如果最后一个窗口的长度小于max_length
    if len(merged_predictions) > max_length:
        merged_predictions = merged_predictions[:max_length]

    return merged_predictions


def main():
    # 加载模型和分词器
    model_path = './ner_model'
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForTokenClassification.from_pretrained(model_path)
    tok = pkuseg.pkuseg('./tok_model')  # 加载分词器

    # 输入文本
    text = '''
国务院办公厅关于复制推广

营商环境创新试点改革举措的通知

国办发〔2022〕35号



各省、自治区、直辖市人民政府，国务院各部委、各直属机构：

优化营商环境是培育和激发市场主体活力、增强发展内生动力的关键之举，党中央、国务院对此高度重视。2021年，国务院部署在北京、上海、重庆、杭州、广州、深圳6个城市开展营商环境创新试点。相关地方和部门认真落实各项试点改革任务，积极探索创新，着力为市场主体减负担、破堵点、解难题，取得明显成效，形成了一批可复制推广的试点经验。为进一步扩大改革效果，推动全国营商环境整体改善，经国务院同意，决定在全国范围内复制推广一批营商环境创新试点改革举措。现就有关事项通知如下：

一、复制推广的改革举措

（一）进一步破除区域分割和地方保护等不合理限制（4项）。“开展‘一照多址’改革”、“便利企业分支机构、连锁门店信息变更”、“清除招投标和政府采购领域对外地企业设置的隐性门槛和壁垒”、“推进客货运输电子证照跨区域互认与核验”等。

（二）健全更加开放透明、规范高效的市场主体准入和退出机制（9项）。“拓展企业开办‘一网通办’业务范围”、“进一步便利企业开立银行账户”、“优化律师事务所核名管理”、“企业住所（经营场所）标准化登记”、“推行企业登记信息变更网上办理”、“推行企业年度报告‘多报合一’改革”、“探索建立市场主体除名制度”、“进一步便利破产管理人查询破产企业财产信息”、“进一步完善破产管理人选任制度”等。

（三）持续提升投资和建设便利度（7项）。“推进社会投资项目‘用地清单制’改革”、“分阶段整合相关测绘测量事项”、“推行水电气暖等市政接入工程涉及的行政审批在线并联办理”、“开展联合验收‘一口受理’”、“进一步优化工程建设项目联合验收方式”、“简化实行联合验收的工程建设项目竣工验收备案手续”、“对已满足使用功能的单位工程开展单独竣工验收”等。

（四）更好支持市场主体创新发展（2项）。“健全知识产权质押融资风险分担机制和质物处置机制”、“优化科技企业孵化器及众创空间信息变更管理模式”等。

（五）持续提升跨境贸易便利化水平（5项）。“优化进出口货物查询服务”、“加强铁路信息系统与海关信息系统的数据交换共享”、“推进水铁空公多式联运信息共享”、“进一步深化进出口货物‘提前申报’、‘两步申报’、‘船边直提’、‘抵港直装’等改革”、“探索开展科研设备、耗材跨境自由流动，简化研发用途设备和样本样品进出口手续”等。

（六）维护公平竞争秩序（3项）。“清理设置非必要条件排斥潜在竞争者行为”、“推进招投标全流程电子化改革”、“优化水利工程招投标手续”等。

（七）进一步加强和创新监管（5项）。“在部分领域建立完善综合监管机制”、“建立市场主体全生命周期监管链”、“在部分重点领域建立事前事中事后全流程监管机制”、“在税务监管领域建立‘信用+风险’监管体系”、“实行特种设备作业人员证书电子化管理”等。

（八）依法保护各类市场主体产权和合法权益（2项）。“建立健全政务诚信诉讼执行协调机制”、“畅通知识产权领域信息交换渠道”等。

（九）优化经常性涉企服务（13项）。“简化检验检测机构人员信息变更办理程序”、“简化不动产非公证继承手续”、“对个人存量房交易开放电子发票功能”、“实施不动产登记、交易和缴纳税费‘一网通办’”、“开展不动产登记信息及地籍图可视化查询”、“推行非接触式发放税务UKey”、“深化‘多税合一’申报改革”、“推行全国车船税缴纳信息联网查询与核验”、“进一步拓展企业涉税数据开放维度”、“对代征税款试行实时电子缴税入库的开具电子完税证明”、“推行公安服务‘一窗通办’”、“推行企业办事‘一照通办’”、“进一步扩大电子证照、电子签章等应用范围”等。

二、切实抓好复制推广工作的组织实施

（一）高度重视复制推广工作。各地区要将复制推广工作作为进一步打造市场化法治化国际化营商环境的重要举措，主动对标先进，加强学习借鉴，细化改革举措，确保复制推广工作取得实效。国务院各有关部门要结合自身职责，及时出台改革配套政策，支持指导地方做好复制推广工作；涉及调整部门规章和行政规范性文件，以及向地方开放系统接口和授权数据使用的，要抓紧按程序办理，确保2022年底前落实到位。

（二）用足用好营商环境创新试点机制。各试点城市要围绕推动有效市场和有为政府更好结合，持续一体推进“放管服”改革，进一步对标高标准国际经贸规则，聚焦市场主体所需所盼，加大先行先试力度，为全国优化营商环境工作积累更多创新经验。国务院办公厅要加强统筹协调和跟踪督促，及时总结推广典型经验做法，推动全国营商环境持续改善。

（三）完善改革配套监管措施。各地区、各有关部门要结合实际稳步推进复制推广工作，对于涉及管理方式、管理权限、管理层级调整的相关改革事项，要夯实监管责任，逐项明确监管措施，完善监管机制，实现事前事中事后全链条全领域监管，确保改革平稳有序推进。

复制推广工作中的重要情况，各地区、各有关部门要及时向国务院请示报告。



附件：首批在全国复制推广的营商环境创新试点改革举措清单'''
    text = tok.cut(text)

    # 使用滑动窗口进行预测
    predictions = sliding_window_predictions(text, tokenizer, model)

    # 合并预测结果
    final_predictions = merge_predictions(predictions, stride=64, max_length=512)

    # 打印最终预测
    label_map_inv = {0: "O", 1: "B-POLICY_TOOL", 2: "I-POLICY_TOOL", 3: "B-POLICY_GOAL", 4: "I-POLICY_GOAL"}
    decoded_predictions = [label_map_inv[label] for label in final_predictions]
    print(decoded_predictions)


if __name__ == '__main__':
    main()
