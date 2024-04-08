import argparse
import os

import numpy as np
import pandas as pd
import torch
import json
from flask_mysqldb import MySQL
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer

from MPNs_PURE.entity.models import EntityModel
from MPNs_PURE.entity.utils import batchify, convert_dataset_to_samples
from MPNs_PURE.relation.models import BertForRelation
from MPNs_PURE.relation.utils import generate_pre_relation_data
from MPNs_PURE.run_relation import convert_examples_to_features, evaluate
from MPNs_PURE.run_relation import print_pred_json
from MPNs_PURE.shared.const import get_labelmap, task_ner_labels
from MPNs_PURE.shared.data_structures import Pre_data, Dataset
from flask import Flask, request, jsonify
from flask_cors import CORS

from MPNs_PURE.entity.models import EntityModel

app = Flask(__name__)
CORS(app)

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='mpns', choices=['ace04', 'ace05', 'scierc', 'mpns'])
parser.add_argument('--model', type=str, default='entity_output',
                    help="the base model name (a huggingface model)")
parser.add_argument("--auto_model", default='D:\pythonProject\MPNs_PURE\matscibert', type=str)
parser.add_argument('--bert_model_dir', type=str, default=None,
                    help="the base model directory")
parser.add_argument('--use_albert', action='store_true',
                    help="whether to use ALBERT model")
parser.add_argument('--max_span_length', type=int, default=16,
                    help="spans w/ length up to max_span_length are considered as candidates")
parser.add_argument("--output_dir", default='relation_output', type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_length", default=228, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument('--add_new_tokens', action='store_true',
                    help="Whether to add new tokens as marker tokens instead of using [unusedX] tokens.")
parser.add_argument("--eval_batch_size", default=32, type=int,
                    help="Total batch size for eval.")
parser.add_argument("--prediction_file", type=str, default="predictions_test.json",
                    help="The prediction filename for the relation model")
args = parser.parse_args()
num_ner_labels = len(task_ner_labels[args.task]) + 1
model = EntityModel(args, num_ner_labels=num_ner_labels)
device = torch.device("cpu")
ner_label2id, ner_id2label = get_labelmap(task_ner_labels['mpns'])


# label2id = {label: i for i, label in enumerate(label_list)}

def get_prediction_ner(model, data):
    """
    Save the prediction as a json file
    """
    test_samples, test_ner = convert_dataset_to_samples(data, args.max_span_length, ner_label2id=ner_label2id,
                                                        context_window=100)
    batches = batchify(test_samples, 32)
    ner_result = {}
    tot_pred_ett = 0
    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        for sample, preds in zip(batches[i], pred_ner):
            off = sample['sent_start_in_doc'] - sample['sent_start']
            k = sample['doc_key'] + '-' + str(sample['sentence_ix'])
            ner_result[k] = []
            for span, pred in zip(sample['spans'], preds):
                span_id = '%s::%d::(%d,%d)' % (sample['doc_key'], sample['sentence_ix'], span[0] + off, span[1] + off)
                if pred == 0:
                    continue
                ner_result[k].append([span[0] + off, span[1] + off, ner_id2label[pred]])
            tot_pred_ett += len(ner_result[k])

    # print(ner_result)
    js = list(data.js)
    for i, doc in enumerate(js):
        doc["predicted_ner"] = []
        doc["predicted_relations"] = []
        for j in range(len(doc["sentences"])):
            k = doc['doc_key'] + '-' + str(j)
            if k in ner_result:
                doc["predicted_ner"].append(ner_result[k])
            else:
                doc["predicted_ner"].append([])

            doc["predicted_relations"].append([])

        js[i] = doc

    return js


def get_prediction_re(data):
    if os.path.exists(os.path.join(args.output_dir, 'label_list.json')):
        with open(os.path.join(args.output_dir, 'label_list.json'), 'r') as f:
            label_list = json.load(f)
    re_label2id = {label: i for i, label in enumerate(label_list)}
    re_id2label = {i: label for i, label in enumerate(label_list)}
    re_num_labels = len(label_list)

    if os.path.exists(os.path.join(args.output_dir, 'special_tokens.json')):
        with open(os.path.join(args.output_dir, 'special_tokens.json'), 'r') as f:
            special_tokens = json.load(f)
    else:
        special_tokens = {}
    with open(os.path.join(args.output_dir, 'special_tokens.json'), 'w') as f:
        json.dump(special_tokens, f)

    tokenizer = AutoTokenizer.from_pretrained(args.auto_model)
    eval_dataset, eval_examples, eval_nrel = generate_pre_relation_data(data, context_window=100)
    eval_features = convert_examples_to_features(
        eval_examples, re_label2id, args.max_seq_length, tokenizer, special_tokens,
        unused_tokens=not (args.add_new_tokens))
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_sub_idx = torch.tensor([f.sub_idx for f in eval_features], dtype=torch.long)
    all_obj_idx = torch.tensor([f.obj_idx for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sub_idx,
                              all_obj_idx)
    eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
    eval_label_ids = all_label_ids
    model = BertForRelation.from_pretrained(args.output_dir, num_rel_labels=re_num_labels)
    model.to(device)
    model.eval()
    nb_eval_steps = 0
    preds = []
    for input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        sub_idx = sub_idx.to(device)
        obj_idx = obj_idx.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None, sub_idx=sub_idx, obj_idx=obj_idx)
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    logits = preds[0]
    preds = np.argmax(preds[0], axis=1)
    # print(preds)
    mpns_result = print_pred_json(eval_dataset, eval_examples, preds, re_id2label)
    return mpns_result

def data_tran(data):
    # 创建新的样本
    sample = []
    sentences = data.split(".")
    sentences = [sentence.strip() for sentence in sentences if sentence.strip() != ""]
    i = 1
    for sentence in sentences:
        sentence_tokens = sentence.split()
        new_sample = {
            "clusters": [],
            "sentences": [sentence_tokens],
            "ner": [[]],
            "relations": [[]],
            "doc_key": "M" + str(i)
        }
        i = i + 1
        sample.append(new_sample)
    return sample


def get_prediction(data):
    ner_results = get_prediction_ner(model, data)
    print(ner_results)
    results = get_prediction_re(ner_results)
    return results

def data_processing(data):
    # 解析成传统三元组
    triples = []
    for item in data:
        # print(item)
        sentences = item['sentences']
        for relation_list in item['predicted_relations']:
            for relation in relation_list:
                # 提取关系两端实体的位置
                start_entity1, end_entity1 = relation[0], relation[1]
                start_entity2, end_entity2 = relation[2], relation[3]
                # 提取关系类型
                relation_type = relation[4]

                # 直接从句子中提取实体名
                entity1 = " ".join(sentences[0][start_entity1:end_entity1 + 1])
                entity2 = " ".join(sentences[0][start_entity2:end_entity2 + 1])

                # 构建三元组并添加到列表
                triples.append((entity1, entity2, relation_type))

    # 初始化一个列表来存储每个MPNs的数据字典
    mpns_data_list = []

    # 定义关系到字段的映射
    relation_to_field = {
        "MPN_C": "moc",
        "MPN_P": "polyphenol",
        "MPN_M": "metal",
        "MPN_A": "application",
        "MPN_E": "modifier"
    }

    # 对于每个三元组
    for entity1, _, relation in triples:
        # 查找是否已经有该MPNs的数据字典
        mpns_data = next((item for item in mpns_data_list if item["mpns"] == entity1), None)

        # 如果没有找到，创建一个新的数据字典
        if not mpns_data:
            mpns_data = {
                "mpns": entity1,
                "metal": set(),
                "polyphenol": set(),
                "modifier": set(),
                "moc": set(),
                "application": set()
            }
            mpns_data_list.append(mpns_data)

        # 添加数据到对应的字段
        field = relation_to_field.get(relation)
        if field:
            mpns_data[field].add(_)

    # 将存储有多个值的集合转换为逗号分隔的字符串
    for mpns_data in mpns_data_list:
        for key in ["metal", "polyphenol", "modifier", "moc", "application"]:
            if mpns_data[key]:
                mpns_data[key] = ','.join(mpns_data[key])
            else:
                mpns_data[key] = ""  # 或者你可以选择其他方式来表示空值

    return mpns_data_list


@app.route('/api/predict', methods=['POST'])
def predict():
    # 从请求中获取数据
    data = request.form.get('data')
    print(data)
    # 调用预测接口
    predatas = data_tran(data)
    test_data = Pre_data(predatas)
    result = get_prediction(test_data)
    fianl_result = data_processing(result)
    print(fianl_result)
    # 返回 JSON 响应
    return jsonify({'result': fianl_result})


# MySQL configurations
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'hzh991118'
app.config['MYSQL_DB'] = 'mpn_dataset'

mysql = MySQL(app)


@app.route('/api/users', methods=['GET'])
def all_mpns():
    cur = mysql.connection.cursor()
    cur.execute("select * from mpns")
    rows = cur.fetchall()
    # 获取列名
    column_names = [i[0] for i in cur.description]

    # 将元组转换为字典列表，包括列名
    result = []
    for row in rows:
        row_dict = {column_names[i]: row[i] for i in range(len(row))}
        result.append(row_dict)
    # cur.close()
    return jsonify(result)


@app.route('/api/search', methods=['GET'])
def search():
    query_param = request.args.get('query', '')  # 从查询参数中获取搜索关键字
    cur = mysql.connection.cursor()
    like_string = f"%{query_param}%"
    cur.execute(
        "SELECT * FROM mpns WHERE mpns LIKE %s OR metal LIKE %s OR polyphenol LIKE %s  OR modifier LIKE %s  OR moc LIKE %s  OR application LIKE %s ",
        (like_string, like_string, like_string, like_string, like_string, like_string))
    results = cur.fetchall()
    # cur.close()

    # 转换查询结果为字典列表
    column_names = [desc[0] for desc in cur.description]
    mpns = [dict(zip(column_names, row)) for row in results]
    return jsonify(mpns)


if __name__ == '__main__':
    app.run(debug=True)

    # text = "In this work, we report DFT32 calculations of spectroscopic properties (1H NMR, IR, and UV-Vis spectra) of distinct Zn(II)-kaempferol complexes: [ZnKaempferol(H2O)n]+ (n = 2 or 4) and Zn(Kaempferol)2(H2O)n (n = 0 or 2).We compared our theoretical spectroscopic results with experimental data in solution published recently.As 1H NMR chemical shifts for the kaempferol molecule are very sensitive to the molecular chemical environment due to the presence of the metal center, the best match between experimental and theoretical 1H NMR profile can lead to information on the Zn(II)-kaempferol complex molecular structure in solution, which is difficult to achieve on experimental basis only.In most studies reported so far on flavonoids, coplanarity of the aromatic rings of polyphenols has been assumed.Our new theoretical results pointed out that in solution the B-ring of Kaempferol deviates significantly from planarity, a relevant information reported for the first time in the literature.This is valuable a information for future studies involving structure-activity relationship and interaction mechanism with DNA."
    # predatas = data_tran(text)
    # print(predatas)
    # test_data = Pre_data(predatas)
    # get_prediction(test_data)
