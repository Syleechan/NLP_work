import json
import codecs
import functools
import tensorflow as tf

import os
from dataset import BertNERDataGenerator
from layers import Encoder, AttentionLayer
from submodule.bert.optimization import create_optimizer


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

flags = tf.flags.FLAGS
# data params
tf.flags.DEFINE_string("data_dir", None, "Input data path. Parse train.txt & test.txt in this folder.")
tf.flags.DEFINE_string("model_dir", None, "Path to save the model and prediction file.")
tf.flags.DEFINE_string("bert_dir", None, "Bert model path which containing: model, vocab and config")
tf.flags.DEFINE_integer("max_seq_length", 256, "Max length of input sequence.")
# model params
tf.flags.DEFINE_string("mode", "train", "Option mode: train, test, train_and_test")
tf.flags.DEFINE_integer("epochs", 1, "Number of train epochs")
tf.flags.DEFINE_integer("batch_size", 10, "Number of examples per batch")
tf.flags.DEFINE_float("learning_rate", 2e-5, "Learning rate for the bert model")
# attention params
tf.flags.DEFINE_string("attention_type", "self_scaled_dot", "Which attention score method to use.")
tf.flags.DEFINE_integer("attention_dim", 256, "Attention dimension.")
# constrains
tf.flags.mark_flag_as_required("data_dir")
tf.flags.mark_flag_as_required("bert_dir")
tf.flags.mark_flag_as_required("model_dir")


def input_fn(data_generator: BertNERDataGenerator, mode: str):
    """按照input_fn要求，将TF Record中保存的数据组装为feature与label返回"""
    def parse_example(example) -> tuple:
        parsed_features = tf.parse_single_example(example, name2features)
        for name in list(parsed_features.keys()):
            t = parsed_features[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            parsed_features[name] = t

        features = {
            'seq_mask': parsed_features['seq_mask'],
            'token_ids': parsed_features['token_ids'],
            'segment_ids': parsed_features['segment_ids'],
            'symptom_ids': parsed_features['symptom_ids'],
            'symptom_mask': parsed_features['symptom_mask']
        }
        labels = parsed_features['label_ids']
        return features, labels

    name2features = {
        'seq_mask': tf.FixedLenFeature([1], tf.int64),
        'token_ids': tf.FixedLenFeature([flags.max_seq_length], tf.int64),
        'segment_ids': tf.FixedLenFeature([flags.max_seq_length], tf.int64),
        'label_ids': tf.FixedLenFeature([flags.max_seq_length], tf.int64),
        'symptom_ids': tf.FixedLenFeature([flags.max_seq_length], tf.int64),
        'symptom_mask': tf.FixedLenFeature([1], tf.int64)
    }
    dataset = data_generator.load_record_file(mode=mode).map(parse_example)

    if mode == "train":
        dataset = dataset.repeat(flags.epochs)
        dataset = dataset.shuffle(buffer_size=200)
        dataset = dataset.batch(flags.batch_size)
    else:
        dataset = dataset.batch(1)

    return dataset


def model_fn(features: dict, labels: tf.Tensor, mode, params) -> tf.estimator.EstimatorSpec:
    """tf.estimator所需的model fn"""
    token_ids = features['token_ids']
    segment_ids = features['segment_ids']
    seq_mask = tf.squeeze(features['seq_mask'], axis=-1)
    symptom_ids = features['symptom_ids']
    symptom_mask = tf.squeeze(features['symptom_mask'], axis=-1)
    max_seq_length, label_size = token_ids.get_shape()[1], params['label_size']

    input_mask = tf.sequence_mask(seq_mask, maxlen=max_seq_length, dtype=tf.int32)
    symptom_mask = tf.sequence_mask(symptom_mask, maxlen=max_seq_length, dtype=tf.int32)
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    encoder = Encoder(flags.bert_dir, bert_layers=10)
    encoder_s = Encoder(flags.bert_dir, bert_layers=4)
    attention = AttentionLayer(flags.attention_type, flags.attention_dim)

    # Encoder
    token_encode = encoder.encode(token_ids, input_mask, segment_ids, is_training)
    symptom_encode = encoder_s.encode(symptom_ids, symptom_mask, segment_ids, is_training, return_type='pooled')

    # Attention
    attention_output = attention.attention(token_encode, symptom_encode, input_mask, is_training)

    # Logits
    logits = tf.layers.dense(attention_output, label_size,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "label_probs": tf.nn.softmax(logits),
            "label_ids": tf.argmax(logits, axis=-1)
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.one_hot(labels, depth=label_size, dtype=tf.float32)
    loss = - tf.reduce_sum(labels * log_probs, axis=-1)
    loss_mask = tf.sequence_mask(seq_mask, maxlen=max_seq_length, dtype=tf.float32)
    loss = tf.reduce_sum(loss * loss_mask)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = create_optimizer(loss=loss, init_lr=flags.learning_rate, num_train_steps=params['train_steps'],
                                    num_warmup_steps=params['warm_up_steps'], use_tpu=False)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {}        # TODO：选择合适的metrics
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)


def main(_):
    data_converter = BertNERDataGenerator(
        data_dir=flags.data_dir,
        vocab_path=os.path.join(flags.bert_dir, "vocab.txt"),
        max_seq_length=flags.max_seq_length,
        model_dir=flags.model_dir
    )

    params = {
        # train config
        "learning_rate": flags.learning_rate,
        "train_steps": data_converter.get_train_example_size() * flags.epochs // flags.batch_size,
        "label_size": data_converter.label_size
    }
    params['warm_up_steps'] = int(params['train_steps'] * 0.1)

    run_config = tf.estimator.RunConfig(
        model_dir=flags.model_dir,
        save_checkpoints_steps=5000,
        keep_checkpoint_max=5,
        log_step_count_steps=100
    )
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=params,
        warm_start_from=None
    )
    general_input_fn = functools.partial(input_fn, data_generator=data_converter)

    if "train" in flags.mode:
        train_input_fn = functools.partial(general_input_fn, mode="train")
        estimator.train(input_fn=train_input_fn)

    if "test" in flags.mode:
        test_input_fn = functools.partial(general_input_fn, mode="test")
        predictions = estimator.predict(input_fn=test_input_fn)
        test_data = data_converter.data_loader(os.path.join(flags.data_dir, "test.json"))
        id2label = {data_converter.label2id[label]: label for label in data_converter.label2id.keys()}
        fout = codecs.open(os.path.join(flags.model_dir, "predictions.txt"), "w", encoding="utf-8")
        for idx, (predict, ground_truth) in enumerate(zip(predictions, test_data)):
            gold_text = ground_truth['text']
            predict_label_ids = predict['label_ids'][1:len(gold_text) + 1]
            predict_labels = [id2label[idx] for idx in predict_label_ids]
            # parse predicted labels
            labels = list()
            idx = 0
            while idx < len(predict_labels):
                if predict_labels[idx].startswith('B-'):
                    start, types = idx, predict_labels[idx].split('-')[1]
                    idx += 1
                    while idx < len(predict_labels) and predict_labels[idx].startswith('I-') and predict_labels[idx].split('-')[1] == types:
                        idx += 1
                    labels.append([start, idx - 1, types])
                else:
                    idx += 1
            ground_truth['labels'] = labels
            fout.write(json.dumps(ground_truth, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run()
