import json
import nltk
from nltk.tokenize import regexp_tokenize
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from BIONER_ER.config import model_config

nltk.download('punkt')
# 加载原始的JSON数据集
with open('D:\pythonProject\MPNs_PURE\slides\MPNs.json', 'r') as f:
    original_data = json.load(f)

# 分词器
# tokenizer = TreebankWordTokenizer()
tokenizer = AutoTokenizer.from_pretrained(model_config.MATSCIBERT)

# 定义正则表达式模式
pattern = r'\w+|[^\w\s]'
# 创建新的数据集
new_data = []
with open('D:\pythonProject\MPNs_PURE\slides/train.json', 'w') as f:
    # 遍历每个样本
    for sample in original_data:
        text = sample['data']['text']
        original_entities = sample['annotations']
        spans = []
        re_spans = []
        # 分词
        words = regexp_tokenize(text, pattern)
        for original_entity in original_entities:
            value_entities = original_entity['result']
            print(value_entities)
            for value_entity in value_entities:
                if value_entity['type'] == 'labels':
                    entity = value_entity['value']
                    # # 更新实体标注的起始位置
                    # for entity in entities:
                    start, end, label = entity['start'], entity['end'], entity['labels']
                    # print(start, end, label)
                    new_start = len(regexp_tokenize(text[:start], pattern))
                    new_end = len(regexp_tokenize(text[:end], pattern)) - 1

                    entity['start'] = new_start
                    entity['end'] = new_end
                    entity['labels'] = label[0]
                    new_item = [entity['start'], entity['end'], entity['labels']]
                    spans.append(new_item)
                else:
                    if 'from_id' in value_entity and 'to_id' in value_entity:
                        from_id = value_entity['from_id']
                        to_id = value_entity['to_id']
                        label = value_entity['labels']
                        new_label = label[0]
                        print(value_entity)

                        for value in value_entities:
                            if value['id'] == from_id:
                                start = value['value']['start']
                                end = value['value']['end']
                                break

                        for value in value_entities:
                            if value['id'] == to_id:
                                start_to = value['value']['start']
                                end_to = value['value']['end']
                                break

                        item = [start, end, start_to, end_to, new_label]
                        print(item)
                        re_spans.append(item)


        # 创建新的样本
        new_sample = {
            "clusters": [],
            "sentences": words,
            "ner": spans,
            "relations": re_spans,
            "doc_key": "M" + str(sample['id'])
        }

        json.dump(new_sample, f)
        f.write("\n")


