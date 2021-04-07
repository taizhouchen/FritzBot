# encoding=utf-8

import tensorflow as tf
import numpy as np
import codecs
import pickle
import os
import re
from datetime import datetime

from bert_base.train.models import create_model, InputFeatures
from bert_base.bert import tokenization, modeling
from bert_base.train.train_helper import get_args_parser
from bert_base.train.bert_lstm_ner import get_pos_info, get_pos_list,lemmatization

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

args = get_args_parser()

model_dir = '/media/ivonchan/My Disk/Taizhou_research/Sketronic/NLP/FritzBot/BERT-BiLSTM-CRF-NER/output_student_withpos_CRF'
bert_dir = '/media/ivonchan/My Disk/Taizhou_research/Sketronic/NLP/FritzBot/BERT-BiLSTM-CRF-NER/checkpoint/uncased_L-12_H-768_A-12'
num_layers = 4
lstm_size = 128

doLemmatization = False
crf_only = True
is_training = False
use_one_hot_embeddings = False
batch_size=1

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess=tf.Session(config=gpu_config)
model=None

global graph
input_ids_p, input_mask_p, label_ids_p, segment_ids_p = None, None, None, None


print('checkpoint path:{}'.format(os.path.join(model_dir, "checkpoint")))
if not os.path.exists(os.path.join(model_dir, "checkpoint")):
    raise Exception("failed to get checkpoint. going to return ")

with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}

with codecs.open(os.path.join(model_dir, 'label_list.pkl'), 'rb') as rf:
    label_list = pickle.load(rf)
num_labels = len(label_list) + 1

graph = tf.get_default_graph()
with graph.as_default():
    print("going to restore checkpoint")
    #sess.run(tf.global_variables_initializer())
    input_ids_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="input_mask")
    pos_ids_p = tf.placeholder(tf.int32, [batch_size, args.max_seq_length], name="pos_ids") # edit by Taizhou

    bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json'))
    (total_loss, logits, trans, pred_ids) = create_model(
        bert_config=bert_config, is_training=False, input_ids=input_ids_p, input_mask=input_mask_p, segment_ids=None,
        labels=None, pos_ids=pos_ids_p, num_labels=num_labels, use_one_hot_embeddings=False, dropout_rate=1.0, lstm_size=lstm_size, num_layers=num_layers, use_pos_feature=True, crf_only=crf_only) # edit by Taizhou

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))


tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=args.do_lower_case)

pos_list, pos_one_hot_list = get_pos_list('./pos2id.txt')

def clean_str(text):
    text = text.lower()
    # Clean the text
    # text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"won't", "would not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    # text = re.sub(r",", " , ", text)
    text = re.sub(r"\.", "", text)
    text = re.sub(r"!", " ! ", text)
    # text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\“", "", text)
    text = re.sub(r"\”", "", text)

    text = re.sub(r"\"", "", text)
    text = re.sub(r"\&", "and", text)
    text = re.sub(r"\,", "", text)

    return text.strip()

def predict_online_terminal():
    """
    do online prediction. each time make prediction for one instance.
    you can change to a batch if you want.

    :param line: a list. element is: [dummy_label,text_a,text_b]
    :return:
    """
    def convert(line):
        feature = convert_single_example(0, line, label_list, args.max_seq_length, tokenizer, 'p')
        input_ids = np.reshape([feature.input_ids],(batch_size, args.max_seq_length))
        input_mask = np.reshape([feature.input_mask],(batch_size, args.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids],(batch_size, args.max_seq_length))
        label_ids = np.reshape([feature.label_ids],(batch_size, args.max_seq_length))
        pos_ids = np.reshape([feature.pos_ids],(batch_size, args.max_seq_length)) # edit by Taizhou
        return input_ids, input_mask, segment_ids, label_ids, pos_ids

    global graph
    with graph.as_default():
        print(id2label)
        while True:
            print('input the test sentence:')
            sentence = str(input())
            start = datetime.now()
            if len(sentence) < 2:
                print(sentence)
                continue

            # sentence = tokenizer.tokenize(sentence)
            from nltk.tokenize import word_tokenize
            sentence = word_tokenize(sentence)

            # print('your input is:{}'.format(sentence))
            input_ids, input_mask, segment_ids, label_ids, pos_ids = convert(sentence) #edit by Taizhou

            feed_dict = {input_ids_p: input_ids,
                         input_mask_p: input_mask,
                         pos_ids_p: pos_ids} # edit by Taizhou
            # run session get current feed_dict result
            pred_ids_result = sess.run([pred_ids], feed_dict)
            pred_label_result = convert_id_to_label(pred_ids_result, id2label)
            print(pred_label_result)
            # result = strage_combined_link_org_loc(sentence, pred_label_result[0])
            CAU_list, EFF_list = get_CAU_EFF(sentence, pred_label_result[0])

            print('CAU: ')
            print(CAU_list)
            print('EFF: ')
            print(EFF_list)

            print('time used: {} sec'.format((datetime.now() - start).total_seconds()))

# edit by Taizhou
def predict_online(sentence):
    """
    prediction for a single sentence
    :param sentence:
    :return:
    """
    def convert(line):
        feature = convert_single_example(0, line, label_list, args.max_seq_length, tokenizer, 'p', doLemmatization)
        input_ids = np.reshape([feature.input_ids],(batch_size, args.max_seq_length))
        input_mask = np.reshape([feature.input_mask],(batch_size, args.max_seq_length))
        segment_ids = np.reshape([feature.segment_ids],(batch_size, args.max_seq_length))
        label_ids = np.reshape([feature.label_ids],(batch_size, args.max_seq_length))
        pos_ids = np.reshape([feature.pos_ids],(batch_size, args.max_seq_length)) # edit by Taizhou
        return input_ids, input_mask, segment_ids, label_ids, pos_ids

    global graph
    with graph.as_default():
        # print('input the test sentence:')
        # sentence = str(input())
        start = datetime.now()
        if len(sentence) < 2:
            print(sentence)
            return -1

        # sentence = tokenizer.tokenize(sentence)
        from nltk.tokenize import word_tokenize
        sentence = word_tokenize(sentence)
        
        # print('your input is:{}'.format(sentence))
        input_ids, input_mask, segment_ids, label_ids, pos_ids = convert(sentence) #edit by Taizhou

        feed_dict = {input_ids_p: input_ids,
                     input_mask_p: input_mask,
                     pos_ids_p: pos_ids} # edit by Taizhou
        # run session get current feed_dict result
        pred_ids_result = sess.run([pred_ids], feed_dict)
        pred_label_result = convert_id_to_label(pred_ids_result, id2label)
        # print(pred_label_result)
        # result = strage_combined_link_org_loc(sentence, pred_label_result[0])
        CAU_list, EFF_list, CAU_id_list, EFF_id_list = get_CAU_EFF(sentence, pred_label_result[0])

        timetaken = (datetime.now() - start).total_seconds()

        return CAU_id_list, CAU_list, EFF_id_list, EFF_list, timetaken

# edit by Taizhou
def get_CAU_EFF(sentence, label):
    """
    get the cause words effect words from the sentence
    :param sentence:
    :param label:
    :return:
    """

    bcau_indexs = [i for i, x in enumerate(label) if x == 'B-CAU']
    beff_indexs = [i for i, x in enumerate(label) if x == 'B-EFF']

    CAU_list = []
    EFF_list = []
    CAU_id_list = []
    EFF_id_list = []

    for bcau_id in bcau_indexs:
        CAU = []
        CAU_id = []
        CAU.append(sentence[bcau_id])
        CAU_id.append(bcau_id)
        while bcau_id+1 < len(label) and label[bcau_id+1] == 'I-CAU':
            bcau_id = bcau_id + 1
            CAU.append(sentence[bcau_id])
            CAU_id.append(bcau_id)
        CAU_list.append(CAU)
        CAU_id_list.append(CAU_id)

    for beff_id in beff_indexs:
        EFF = []
        EFF_id = []
        EFF.append(sentence[beff_id])
        EFF_id.append(beff_id)
        while beff_id+1 < len(label) and label[beff_id+1] == 'I-EFF':
            beff_id = beff_id + 1
            EFF.append(sentence[beff_id])
            EFF_id.append(beff_id)
        EFF_list.append(EFF)
        EFF_id_list.append(EFF_id)

    return CAU_list, EFF_list, CAU_id_list, EFF_id_list

def convert_id_to_label(pred_ids_result, idx2label):
    """
    将id形式的结果转化为真实序列结果
    :param pred_ids_result:
    :param idx2label:
    :return:
    """
    result = []
    for row in range(batch_size):
        curr_seq = []
        for ids in pred_ids_result[row][0]:
            if ids == 0:
                break
            curr_label = idx2label[ids]
            if curr_label in ['[CLS]', '[SEP]']:
                continue
            curr_seq.append(curr_label)
        result.append(curr_seq)
    return result

def strage_combined_link_org_loc(tokens, tags):
    """
    组合策略
    :param pred_label_result:
    :param types:
    :return:
    """
    def print_output(data, type):
        line = []
        line.append(type)
        for i in data:
            line.append(i.word)
        print(', '.join(line))

    params = None
    eval = Result(params)
    if len(tokens) > len(tags):
        tokens = tokens[:len(tags)]
    person, loc, org = eval.get_result(tokens, tags)
    print_output(loc, 'LOC')
    print_output(person, 'PER')
    print_output(org, 'ORG')

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode, doLemmatization):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param mode:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(model_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    tokens = example
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(0)
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)

    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    if doLemmatization:
        tokens = lemmatization(tokens)

    # edit by Taizhou
    # get pos feature
    pos, pos_ids, pos_one_hot = get_pos_info(tokens, pos_list, pos_one_hot_list)

    while len(pos_ids) < max_seq_length:
        pos_ids.append(0)

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
        # edit by Taizhou
        pos=pos,
        pos_ids=pos_ids,
        pos_one_hot=pos_one_hot
    )
    return feature


class Pair(object):
    def __init__(self, word, start, end, type, merge=False):
        self.__word = word
        self.__start = start
        self.__end = end
        self.__merge = merge
        self.__types = type

    @property
    def start(self):
        return self.__start
    @property
    def end(self):
        return self.__end
    @property
    def merge(self):
        return self.__merge
    @property
    def word(self):
        return self.__word

    @property
    def types(self):
        return self.__types
    @word.setter
    def word(self, word):
        self.__word = word
    @start.setter
    def start(self, start):
        self.__start = start
    @end.setter
    def end(self, end):
        self.__end = end
    @merge.setter
    def merge(self, merge):
        self.__merge = merge

    @types.setter
    def types(self, type):
        self.__types = type

    def __str__(self) -> str:
        line = []
        line.append('entity:{}'.format(self.__word))
        line.append('start:{}'.format(self.__start))
        line.append('end:{}'.format(self.__end))
        line.append('merge:{}'.format(self.__merge))
        line.append('types:{}'.format(self.__types))
        return '\t'.join(line)

class Result(object):
    def __init__(self, config):
        self.config = config
        self.person = []
        self.loc = []
        self.org = []
        self.others = []
    def get_result(self, tokens, tags, config=None):
        # 先获取标注结果
        self.result_to_json(tokens, tags)
        return self.person, self.loc, self.org

    def result_to_json(self, string, tags):
        """
        将模型标注序列和输入序列结合 转化为结果
        :param string: 输入序列
        :param tags: 标注结果
        :return:
        """
        item = {"entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        last_tag = ''

        for char, tag in zip(string, tags):
            if tag[0] == "S":
                self.append(char, idx, idx+1, tag[2:])
                item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
            elif tag[0] == "B":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "O":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
            last_tag = tag
        if entity_name != '':
            self.append(entity_name, entity_start, idx, last_tag[2:])
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
        return item

    def append(self, word, start, end, tag):
        if tag == 'LOC':
            self.loc.append(Pair(word, start, end, 'LOC'))
        elif tag == 'PER':
            self.person.append(Pair(word, start, end, 'PER'))
        elif tag == 'ORG':
            self.org.append(Pair(word, start, end, 'ORG'))
        else:
            self.others.append(Pair(word, start, end, tag))


if __name__ == "__main__":
    # predict_online_terminal()
    while True:
        print('input the test sentence:')
        sentence = str(input())
        if sentence == 'exit':
            exit()

        sentence = clean_str(sentence)

        CAU_id_list, CAU_list, EFF_id_list, EFF_list, time_taken = predict_online(sentence)

        print('CAU: ')
        print(CAU_list, CAU_id_list)
        print('EFF: ')
        print(EFF_list, EFF_id_list)

        print('time used: {} sec'.format(time_taken))



