""" 对模型输出结果做benchmark
比较两个BERT文件格式的结果，支持输出标签中存在嵌套：如果一个token有多个标签，中间使用"|"分隔

"""
import os
import math
import json
import codecs


def benchmark(ground_truth: list, predictions: list) -> dict:
    """输入预测与真实结果，输出dict格式的常见metrics
    输入格式要求：列表中的每个元素是一个字典，包含str格式的'query'与set格式的'entities'，entities中每个元素是(start,end,type)
      如：{'query': '白宫发言人蓬佩奥', 'entities': {(0,8,PER), (5,8,PER)}}
    """
    # 构造标签集合
    all_labels = {'all'}
    for g in ground_truth:
        for entity in g:
            all_labels.add(entity[2])
    if 'self' in all_labels:
        all_labels.remove('self')

    label_metrics = {label: dict() for label in all_labels}
    for label in all_labels:
        tp, fp, fn = 0, 0, 0
        for idx, (p, g) in enumerate(zip(predictions, ground_truth)):
            pe, ge = p, g
            for cur_ge in ge:
                if cur_ge[2] == label or label == "all":
                    if cur_ge in pe:
                        tp += 1
                    else:
                        fn += 1
            for cur_pe in pe:
                if cur_pe[2] == label or (label == "all" and cur_pe[2] in all_labels):
                    if cur_pe not in ge:
                        fp += 1
        p, r, f1 = 0.0, 0.0, 0.0
        if tp + fp != 0:
            p = tp / (tp + fp)
        if tp + fn != 0:
            r = tp / (tp + fn)
        if math.fabs(p + r - 0.0) > 0.0001:
            f1 = 2 * p * r / (p + r)
        label_metrics[label]['precision'] = p
        label_metrics[label]['recall'] = r
        label_metrics[label]['f1'] = f1
        label_metrics[label]['base info'] = {'tp': tp, 'fp': fp, 'fn': fn}
    return label_metrics


def load_labels(path: str) -> list:
    fin = codecs.open(path, 'r', encoding='utf-8')
    labels = list()
    for line in fin.readlines():
        line = json.loads(line)
        labels.append([tuple(label) for label in line['labels']])
    return labels


def evaluate(gold: list, pred: list):
    assert len(gold) == len(pred)


if __name__ == '__main__':
    assert os.path.exists('results/predictions.txt') and os.path.exists('data/test.json')

    pred = load_labels('results/predictions.txt')
    gold = load_labels('data/test.json')
    metrics = benchmark(gold, pred)
    for label in metrics:
        print(label, metrics[label])

