#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : data_loader
# @Author   : 张志毅
# @Time     : 2020/9/12 19:14
import json

import torch
from torchtext.vocab import Vectors

from baseline.config.config import config, DEVICE
from ..utils.log import logger
from torchtext.data import Field, BucketIterator, Example, Dataset

def x_tokenizer(sentence):
    return [word for word in sentence]

def y_tokenizer(tag: str):
    return [tag]


TEXT = Field(sequential=True, use_vocab=True, tokenize=x_tokenizer, include_lengths=True)
TAG = Field(sequential=True, tokenize=y_tokenizer, use_vocab=True, is_target=True, pad_token=None)
Fields2 = [('text', TEXT), ('tag', TAG)]


class REDataset(Dataset):
    def __init__(self, path, is_bioes, fields, encoding="utf-8", **kwargs):
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for jsonstr in f.readlines():
                jsonstr = json.loads(jsonstr)
                text_list, tag_list = _get_list(jsonstr)
                examples.append(Example.fromlist((text_list, tag_list), fields))
        f.close()
        super(REDataset, self).__init__(examples, fields, **kwargs)

def _get_list(jsonstr):
    text = jsonstr['text']
    tag_list = ['O' for i in range(len(text))]
    spo_lists = jsonstr['spo_list']
    predicates = []
    subject = []
    subject_type = []
    object = []
    for index, spo_list in enumerate(spo_lists):
        predicate = (spo_list['predicate'])
        subject = (spo_list['subject'])
        subject_type = (spo_list['subject_type'])
        object = spo_list['object']['@value']
        object_type = spo_list['object_type']['@value']

        subject_start_poses = _find_all_index(text, subject)
        subject_end_poses = [start + len(subject) - 1 for start in subject_start_poses]
        for j in range(len(subject_start_poses)):
            x = subject_start_poses[j]
            tag_list[x] = 'B_' + subject_type
            x += 1
            while x <= subject_end_poses[j]:
                tag_list[x] = 'I_' + subject_type
                x += 1
        object_start_poses = _find_all_index(text, object)
        object_end_poses = [start + len(object) - 1 for start in object_start_poses]
        for j in range(len(object_start_poses)):
            x = object_start_poses[j]
            if predicate == '手术治疗':
                tag_list[x] = 'B_' + predicate+'_p'
            else:
                tag_list[x] = 'B_' + predicate
            x += 1
            while x <= object_end_poses[j]:
                if predicate == '手术治疗':
                    tag_list[x] = 'I_' + predicate+'_p'
                else:
                    tag_list[x] = 'I_' + predicate
                x += 1
    text_list = list(text)
    return text_list, tag_list

def _find_all_index(str1, str2):
    # 子串str2， in str1查找目标字符串
    starts = []
    start = 0
    over = 0
    while True:
        index = str1.find(str2, start, len(str1))
        if index != -1:
            starts.append(index + over)
            if index + len(str2) <= len(str1) - 1:
                str1 = str1[index + len(str2):]
                over += index + len(str2)
            else:
                break
        else:
            break
    return starts

class Tool():
    def __init__(self):

           self.Fields = Fields2

    def load_data(self, path: str, is_bioes):
        fields = self.Fields
        dataset = REDataset(path, is_bioes, fields=fields)
        return dataset

    def get_text_vocab(self, *dataset):

        TEXT.build_vocab(*dataset)
        return TEXT.vocab

    def get_tag_vocab(self, *dataset):
        TAG.build_vocab(*dataset)
        return TAG.vocab



    def get_iterator(self, dataset: Dataset, batch_size=1,
                     sort_key=lambda x: len(x.text), sort_within_batch=True):
        iterator = BucketIterator(dataset, batch_size=batch_size, sort_key=sort_key,
                              sort_within_batch=sort_within_batch, device=DEVICE)
        return iterator

tool = Tool()