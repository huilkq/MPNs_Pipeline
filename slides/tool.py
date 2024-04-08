import copy
import json

from sklearn.model_selection import train_test_split


# 读取数据
def read11(json_file, pred_file=None):
    gold_docs = [json.loads(line) for line in open(json_file)]
    if pred_file is None:
        return gold_docs

    pred_docs = [json.loads(line) for line in open(pred_file)]
    merged_docs = []
    for gold, pred in zip(gold_docs, pred_docs):
        assert gold["doc_key"] == pred["doc_key"]
        assert gold["sentences"] == pred["sentences"]
        merged = copy.deepcopy(gold)
        for k, v in pred.items():
            if "predicted" in k:
                merged[k] = v
        merged_docs.append(merged)
    print(merged_docs)
    return merged_docs


oraddad = read11("D:\pythonProject\MPNs_PURE\Datasets\output.json")
print(oraddad)

# with open("/MPNs_PURE/Datasets/MNPs/train11.json", 'w') as f:
#     # 遍历每个样本
#     for sample in oraddad:
#         # 创建新的样本
#         new_sample = {"clusters": [], "sentences": [sample['sentences']], "ner": [sample['ner']],
#                       "relations": [sample['relations']], "doc_key": "M"+str(sample['doc_key'])}
#
#         json.dump(new_sample, f)
#         f.write("\n")


train_data, data = train_test_split(oraddad, test_size=0.3, random_state=42)

val_data, test_data = train_test_split(data, test_size=0.33, random_state=42)

# 保存划分后的数据集


with open("D:\pythonProject\MPNs_PURE\Datasets\MNPs\\train.json", 'w') as f:
    # 遍历每个样本
    for sample in train_data:
        # 创建新的样本
        new_sample = {"clusters": [], "sentences": [sample['sentences']], "ner": [sample['ner']],
                      "relations": [sample['relations']], "doc_key": "M"+str(sample['doc_key'])}

        json.dump(new_sample, f)
        f.write("\n")

with open("D:\pythonProject\MPNs_PURE\Datasets\MNPs\devel.json", 'w') as f:
    # 遍历每个样本
    for sample in val_data:
        # 创建新的样本
        new_sample = {"clusters": [], "sentences": [sample['sentences']], "ner": [sample['ner']],
                      "relations": [sample['relations']], "doc_key": "M"+str(sample['doc_key'])}

        json.dump(new_sample, f)
        f.write("\n")

with open("D:\pythonProject\MPNs_PURE\Datasets\MNPs\\test.json", 'w') as f:
    # 遍历每个样本
    for sample in test_data:
        # 创建新的样本
        new_sample = {"clusters": [], "sentences": [sample['sentences']], "ner": [sample['ner']],
                      "relations": [sample['relations']], "doc_key": "M"+str(sample['doc_key'])}

        json.dump(new_sample, f)
        f.write("\n")