import json
import nltk
from nltk.tokenize import regexp_tokenize
from sklearn.model_selection import train_test_split

nltk.download('punkt')
# 加载原始的JSON数据集
with open('D:\pythonProject\MPNs_PURE\slides\MPNs.json', 'r') as f:
    original_data = json.load(f)
# 分词器
# tokenizer = TreebankWordTokenizer()

# 定义正则表达式模式
pattern = r'\w+|[^\w\s]'
# 创建新的数据集
new_data = []
with open('D:\pythonProject\MPNs_PURE\slides/train.json', 'w') as f:
    # 遍历每个样本
    for sample in original_data:
        if 'label' in sample:
            text = sample['text']
            entities = sample['label']
            spans = []
            # 分词
            words = regexp_tokenize(text, pattern)

            # 更新实体标注的起始位置
            for entity in entities:
                start, end, label = entity['start'], entity['end'], entity['labels']
                # print(start, end, label)
                new_start = len(regexp_tokenize(text[:start], pattern))
                new_end = len(regexp_tokenize(text[:end], pattern)) - 1

                entity['start'] = new_start
                entity['end'] = new_end
                entity['labels'] = label[0]

            for entity in entities:
                new_item = [entity['start'], entity['end'], entity['labels']]
                spans.append(new_item)
            # 创建新的样本
            new_sample = {
                "clusters": [],
                "sentences": [words],
                "ner": [spans],
                "relations": [[]],
                "doc_key": "M"+str(sample['id'])
            }

            json.dump(new_sample, f)
            f.write("\n")


# print(new_data)
# 保存新的JSON数据集
# 划分数据集
# train_data, val_data = train_test_split(new_data, test_size=0.3, random_state=42)
#
# # 保存划分后的数据集
# with open('D:\pythonProject\MPNs_PURE\Datasets/output.json', 'w') as f:
#     json.dump(train_data, f)
#
# with open('D:\pythonProject\MPNs_PURE\Datasets/devel.json', 'w') as f:
#         json.dump(new_sample, f)
#         f.write("\n")
