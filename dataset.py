import os
import json
import codecs
import pickle
import tensorflow as tf
from submodule.bert.tokenization import FullTokenizer


class BertNERDataGenerator(object):

    def __init__(self, data_dir: str, model_dir: str, vocab_path: str, max_seq_length: int):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.max_length = max_seq_length
        os.makedirs(self.model_dir, exist_ok=True)

        self.tokenizer = FullTokenizer(vocab_path)
        self.word2id = self.tokenizer.vocab
        self.parse()

    @staticmethod
    def data_loader(data_path: str) -> list:
        return [json.loads(line) for line in codecs.open(data_path, "r", encoding="utf-8").readlines()]

    def parse(self):
        """将原始数据解析为模型需要的格式"""
        for data_name in ["dev", "test", "train"]:
            fin_path = os.path.join(self.data_dir, data_name + ".json")
            fout_path = os.path.join(self.model_dir, "tf_record." + data_name)
            # FIXME
            if not os.path.exists(fout_path):
            # if True:
                data = self.data_loader(fin_path)
                self.label2id = self._get_label_ids(data)
                self.label_size = len(self.label2id)
                # 写入到TF_record文件
                writer = tf.io.TFRecordWriter(fout_path)
                for idx, _data in enumerate(data):
                    example = self._parse_single_example(_data)
                    writer.write(example.SerializeToString())
                print(data_name, len(data))
            else:
                self.label2id = self._get_label_ids([])     # 我赌你的枪里没有子弹
                self.label_size = len(self.label2id)

    def _parse_single_example(self, _data: dict) -> tf.train.Example:
        """将一条数据转化为模型输入格式
        由于bert输入最大长度为512，因此超过该长度的将被截断
        """
        tokens = _data['text']
        token_ids = list(map(lambda x: self.word2id.get(x, self.word2id['[UNK]']), tokens))[:self.max_length - 2]
        labels = ['O'] * len(token_ids)
        for label in _data['labels']:
            start, end, types = label[0], label[1], label[2]
            if types == 'self':
                continue
            if label[1] >= len(labels):
                continue
            for idx in range(start, end + 1):
                if idx == start:
                    labels[idx] = 'B-' + types
                else:
                    labels[idx] = 'I-' + types
        label_ids = list(map(lambda x: self.label2id[x], labels))[:self.max_length - 2]
        token_ids = [self.word2id['[CLS]']] + token_ids + [self.word2id['[SEP]']]
        label_ids = [self.label2id['O']] + label_ids + [self.label2id['O']]
        seq_mask = len(token_ids)
        for _ in range(seq_mask, self.max_length):
            token_ids.append(self.word2id['[PAD]'])
            label_ids.append(self.label2id['O'])
        segment_ids = [0] * self.max_length
        assert len(token_ids) == self.max_length and len(label_ids) == len(token_ids)

        symptom_ids = list(map(lambda x: self.word2id.get(x, self.word2id['[UNK]']), _data['symptom']['val']))
        symptom_mask = len(symptom_ids)
        for _ in range(symptom_mask, self.max_length):
            symptom_ids.append(self.word2id['[PAD]'])
        assert len(symptom_ids) == self.max_length

        example = tf.train.Example(features=tf.train.Features(feature={
            'seq_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_mask])),
            'token_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=token_ids)),
            'symptom_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=symptom_ids)),
            'symptom_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=[symptom_mask])),
            'label_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=label_ids)),
            'segment_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=segment_ids))
        }))
        return example

    def _get_label_ids(self, data: list):
        """从训练数据中获得当前的标签集合"""
        file = os.path.join(self.model_dir, "label2id.pkl")
        if os.path.exists(file):
            return pickle.load(open(file, "rb"))

        labels = {label[2] for d in data for label in d['labels']}
        label2id = dict()
        for label in labels:
            label2id['B-' + label] = len(label2id)
            label2id['I-' + label] = len(label2id)
        label2id['O'] = len(label2id)
        pickle.dump(label2id, open(file, "wb"))
        return label2id

    def load_record_file(self, mode: str) -> tf.data.Dataset:
        """根据输入的mode参数来决定加载的文件"""
        file_path = os.path.join(self.model_dir, "tf_record." + mode)
        if not os.path.exists(file_path):
            raise FileNotFoundError("Load tf_record file failed, please check you running the right way.")
        return tf.data.TFRecordDataset(file_path)

    def get_train_example_size(self) -> int:
        """获得保存在tf_record.train（训练文件）中的example数量
        调用当前方法一般用于计算最终的模型训练step数
        """
        file_path = os.path.join(self.model_dir, "tf_record.train")
        train_size = 0
        for _ in tf.io.tf_record_iterator(file_path):
            train_size += 1
        return train_size


if __name__ == '__main__':
    data_generator = BertNERDataGenerator(
        data_dir=r'data',
        model_dir=r'results',
        vocab_path=r'resources/chinese_L-12_H-768_A-12/vocab.txt',
        max_seq_length=512
    )
