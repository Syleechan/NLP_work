import os
import math
import tensorflow as tf
from submodule.bert.modeling import BertModel, BertConfig
from submodule.bert.modeling import get_assignment_map_from_checkpoint


class Encoder(object):
    """数据的Encoder
    使用BERT的embedding与encoder
    """
    def __init__(self, bert_dir: str, bert_layers: int = None):
        """"""
        self.bert_dir = bert_dir
        self.bert_config = BertConfig.from_json_file(os.path.join(bert_dir, "bert_config.json"))
        if bert_layers is not None:
            self.bert_config.num_hidden_layers = bert_layers

    def encode(self, token_ids, input_mask, segment_ids, is_training: bool, return_type: str = 'sequence'):
        """一个bert :)"""
        bert_model = BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=token_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False
        )

        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, self.bert_dir)
        tf.train.init_from_checkpoint(self.bert_dir, assignment_map)

        if return_type == 'sequence':
            encoder_output = bert_model.get_sequence_output()
        else:
            encoder_output = bert_model.get_pooled_output()

        if is_training:
            encoder_output = tf.nn.dropout(encoder_output, rate=0.2)

        return encoder_output


class AttentionLayer(object):
    """Attention层"""

    def __init__(self, att_type: str, hidden_dim: int, **kwargs):
        self.att_type = att_type
        self.hidden_dim = hidden_dim

    @staticmethod
    def get_mahanttan_similarity(key, query):
        """基于euclid距离[sim]计算相似度：1/(1+sim)，输出前会经过softmax"""
        dis = tf.reduce_sum(tf.abs(query - key), axis=-1)
        sim = 1.0 / (0.5 + dis)
        return sim

    def attention(self, token_encode: tf.Tensor, symptom_encode: tf.Tensor, token_mask, is_training: bool):
        max_len = token_encode.shape[1]
        symptom_encode = tf.expand_dims(symptom_encode, axis=1)

        if self.att_type == 'add':
            token_encode += symptom_encode
            att_output = token_encode

        elif self.att_type == 'mean':
            token_encode = (token_encode + symptom_encode) / 2.0
            att_output = token_encode

        elif self.att_type == 'euclid':
            symptom_encode = tf.tile(symptom_encode, (1, max_len, 1))
            token_mask = tf.cast(token_mask, dtype=tf.float32)
            attention_weight = self.get_mahanttan_similarity(token_encode, symptom_encode)
            attention_weight *= token_mask
            attention_weight = tf.expand_dims(attention_weight, axis=-1)
            att_output = token_encode + symptom_encode * attention_weight

            # token_mask = tf.cast(token_mask, dtype=tf.float32)
            # att_mask = (1.0 - token_mask) * -10000.0
            # attention_weight = self.get_mahanttan_similarity(token_encode, symptom_encode)
            # attention_weight += att_mask
            # attention_weight = tf.nn.softmax(attention_weight, axis=-1)
            # attention_weight = tf.expand_dims(attention_weight, axis=-1)
            # att_output = token_encode + attention_weight * symptom_encode

        else:
            raise NotImplementedError()

        if is_training:
            att_output = tf.nn.dropout(att_output, rate=0.1)
        return att_output
