#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : module
# @Author   : 张志毅
# @Time     : 2020/9/13 15:39
import json
import warnings

import numpy
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import classification_report
from tqdm import tqdm

from baseline.config.config import config, DEVICE
from baseline.data.data_loader import tool
from baseline.model.transformer_bilstm_crf import TransformerEncoderModel
from ..utils.log import logger

warnings.filterwarnings('ignore')


class CHIP2020_RE():
    def __init__(self):
        self.config = config
        self.model = None
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.unlabeled_data = None
        self.word_vocab = None
        self.tag_vocab = None
        self.bi_gram_vocab = None
        self.lattice_vocab = None
        self.train_iter = None
        self.dev_iter = None
        self.test_iter = None
        self.unlabeled_iter = None
        self.model_name = config.model_name
        self.experiment_name = config.experiment_name

    def init_model(self, config=None, word_vocab=None, vocab_size=None, tag_num=None, vectors_path=None):
        model_name = config.model_name
        models = {
            'TransformerEncoderModel': TransformerEncoderModel,
        }
        model = models[model_name](config, word_vocab, vocab_size, tag_num, vectors_path).to(DEVICE)
        return model

    def train(self):
        logger.info('Loading data ...')
        self.train_data = tool.load_data(config.train_path, config.is_bioes)
        self.dev_data = tool.load_data(config.dev_path, config.is_bioes)

        self.word_vocab = tool.get_text_vocab(self.train_data, self.dev_data)
        self.tag_vocab = tool.get_tag_vocab(self.train_data, self.dev_data)
        self.train_iter = tool.get_iterator(self.train_data, batch_size=config.batch_size)
        self.dev_iter = tool.get_iterator(self.dev_data, batch_size=config.batch_size)
        model = self.init_model(self.config, self.word_vocab, len(self.word_vocab), len(self.tag_vocab),
                                config.vector_path)
        self.model = model
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)
        f1_max = 0
        p_max = 0
        r_max = 0
        logger.info('Beginning train ...')
        for epoch in range(config.epoch):
            model.train()
            acc_loss = 0
            for item in tqdm(self.train_iter):
                optimizer.zero_grad()
                tag = item.tag
                text = item.text[0]
                text_len = item.text[1]
                loss = self.model.loss(text, text_len, tag)
                acc_loss += loss.view(-1).cpu().data.tolist()[0]
                loss.backward()
                optimizer.step()

            entity_micro_prf_dict, prf_dict = self.evaluate()
            lable_report = prf_dict['micro avg']
            entity_report = entity_micro_prf_dict['micro avg']
            entity_f1 = entity_report['f1-score']
            entity_p = entity_report['precision']
            entity_r = entity_report['recall']

            # lable_f1 = lable_report['f1-score']
            # lable_p = lable_report['precision']
            # lable_r = lable_report['recall']

            logger.info(
                'epoch: {} precision: {:.4f} recall: {:.4f} f1: {:.4f} loss: {}'.format(epoch + 1, entity_p, entity_r,
                                                                                        entity_f1, acc_loss))
            if entity_f1 > f1_max:
                f1_max = entity_f1
                p_max = entity_p
                r_max = entity_r
                best_epoch = epoch + 1
                logger.info('save best model...')
                torch.save(self.model.state_dict(),
                           config.save_model_path + 'model_{}.pkl'.format(self.experiment_name))
                logger.info(
                    'best model: precision: {:.4f} recall: {:.4f} f1: {:.4f} epoch: {}'.format(p_max, r_max, f1_max,
                                                                                               best_epoch))

    def evaluate(self):
        self.model.eval()
        tag_true_all = []
        tag_pred_all = []
        entity_micro_prf_dict = {}
        # S : 预测输出的结果
        # G ：人工标注的正确的结果
        # entities_total = {'疾病': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '其他': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '其他治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '手术治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '流行学病': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '症状': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '社会学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '药物': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '预防': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '阶段': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '就诊科室': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '同义词': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '辅助治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '化疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '放射治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '手术治疗_p': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '实验室检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '影像学检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '组织学检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '内窥镜检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '筛查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '多发群体': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '发病率': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '发病年龄': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '多发地区': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '发病性别倾向': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '死亡率': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '多发季节': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '传播途径': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '并发症': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '病理分型': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '相关（导致）': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '鉴别诊断': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '相关（转化）': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '相关（症状）': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '临床表现': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '治疗后症状': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '侵及周围组织转移的症状': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '病因': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '高危因素': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '风险评估因素': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '病史': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '遗传因素': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '发病机制': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '病理生理': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '药物治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '发病部位': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '转移部位': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '外侵部位': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '预后状况': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '预后生存率': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '部位': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '流行病学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '辅助检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '相关（导致)': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   }
        # entities_total = {'疾病_预防': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_阶段': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_就诊科室': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '其他_同义词': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_辅助治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_化疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_放射治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '其他治疗_同义词': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_手术治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '手术治疗_同义词': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_实验室检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_影像学检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_辅助检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_组织学检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '检查_同义词': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_内窥镜检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_筛查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_多发群体': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_发病率': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_发病年龄': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_多发地区': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_发病性别倾向': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_死亡率': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_多发季节': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_传播途径': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '流行病学_同义词': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_同义词': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_并发症': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_病理分型': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_相关（导致）': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_鉴别诊断': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_相关（转化）': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_相关（症状）': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_临床表现': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_治疗后症状': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_侵及周围组织转移的症状': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '症状_同义词': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_病因': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_高危因素': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_风险评估因素': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_病史': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_遗传因素': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '社会学_同义词': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_发病机制': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_病理生理': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_药物治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '药物_同义词': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_发病部位': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_外侵部位': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '部位_同义词': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_预后状况': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '疾病_预后生存率': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '预防_其他': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '阶段_其他': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '就诊科室_其他': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '同义词_其他': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '辅助治疗_其他治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '化疗_其他治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '放射治疗_其他治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '同义词_其他治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '手术治疗_手术治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '同义词_手术治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '实验室检查_检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '影像学检查_检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '辅助检查_检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '组织学检查_检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '同义词_检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '内窥镜检查_检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '筛查_检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '多发群体_流行病学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '发病率_流行病学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '发病年龄_流行病学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '多发地区_流行病学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '发病性别倾向_流行病学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '死亡率_流行病学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '多发季节_流行病学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '传播途径_流行病学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '同义词_流行病学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '同义词_疾病': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '并发症_疾病': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '病理分型_疾病': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '相关（导致）_疾病': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '鉴别诊断_疾病': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '相关（转化）_疾病': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '相关（症状）_疾病': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '临床表现_症状': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '治疗后症状_症状': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '侵及周围组织转移的症状_症状': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '同义词_症状': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '病因_社会学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '高危因素_社会学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '风险评估因素_社会学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '病史_社会学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '遗传因素_社会学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '同义词_社会学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '发病机制_社会学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '病理生理_社会学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '药物治疗_药物': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '同义词_药物': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '发病部位_部位': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '转移部位_部位': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '外侵部位_部位': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '同义词_部位': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '预后状况_预后': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #                   '预后生存率_预后': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
        #
        #                   }
        entities_total = {'疾病': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '其他': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '部位': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '其他治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '手术治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '流行病学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '症状': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '社会学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '药物': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '预防_其他': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '阶段_其他': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '就诊科室_其他': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '同义词_其他': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '辅助治疗_其他治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '化疗_其他治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '放射治疗_其他治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '同义词_其他治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '手术治疗_手术治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '同义词_手术治疗': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '实验室检查_检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '影像学检查_检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '辅助检查_检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '组织学检查_检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '同义词_检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '内窥镜检查_检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '筛查_检查': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '多发群体_流行病学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '发病率_流行病学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '发病年龄_流行病学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '多发地区_流行病学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '发病性别倾向_流行病学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '死亡率_流行病学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '多发季节_流行病学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '传播途径_流行病学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '同义词_流行病学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '同义词_疾病': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '并发症_疾病': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '病理分型_疾病': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '相关（导致）_疾病': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '鉴别诊断_疾病': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '相关（转化）_疾病': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '相关（症状）_疾病': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '临床表现_症状': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '治疗后症状_症状': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '侵及周围组织转移的症状_症状': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '同义词_症状': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '病因_社会学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '高危因素_社会学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '风险评估因素_社会学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '病史_社会学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '遗传因素_社会学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '同义词_社会学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '发病机制_社会学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '病理生理_社会学': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '药物治疗_药物': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '同义词_药物': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '发病部位_部位': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '转移部位_部位': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '外侵部位_部位': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '同义词_部位': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '预后状况_预后': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},
                          '预后生存率_预后': {'TP': 0, 'S': 0, 'G': 0, 'p': 0, 'r': 0, 'f1': 0},

                          }
        model = self.model
        for index, iter in enumerate(tqdm(self.dev_iter)):
            if iter.tag.shape[1] == self.config.batch_size:

                text = iter.text[0]
                tag = torch.transpose(iter.tag, 0, 1)
                text_len = iter.text[1]
                result = model(text, text_len)
                for i, result_list in zip(range(text.size(1)), result):
                    text1 = text.permute(1, 0)
                    sentence = [self.word_vocab.itos[w] for w in text1[i][:text_len[i]]]
                    tag_list = tag[i][:text_len[i]]
                    assert len(tag_list) == len(result_list), 'tag_list: {} != result_list: {}'.format(
                        len(tag_list), len(result_list))
                    tag_true = [self.tag_vocab.itos[k] for k in tag_list]
                    tag_true_all.extend(tag_true)
                    tag_pred = [self.tag_vocab.itos[k] for k in result_list]
                    tag_pred_all.extend(tag_pred)
                    entities = self._evaluate(sentence=sentence, tag_true=tag_true, tag_pred=tag_pred)
                    assert len(entities_total) == len(entities), 'entities_total: {} != entities: {}'.format(
                        len(entities_total), len(entities))
                    for entity in entities_total:
                        entities_total[entity]['TP'] += entities[entity]['TP']
                        entities_total[entity]['S'] += entities[entity]['S']
                        entities_total[entity]['G'] += entities[entity]['G']
        TP = 0
        S = 0
        G = 0
        macro_p = 0
        macro_r = 0
        macro_f1 = 0
        print('\n--------------------------------------------------')
        print('\tp\t\t\tr\t\t\tf1\t\t\tlabel_type')
        for entity in entities_total:
            entities_total[entity]['p'] = entities_total[entity]['TP'] / entities_total[entity]['S'] \
                if entities_total[entity]['S'] != 0 else 0
            entities_total[entity]['r'] = entities_total[entity]['TP'] / entities_total[entity]['G'] \
                if entities_total[entity]['G'] != 0 else 0
            entities_total[entity]['f1'] = 2 * entities_total[entity]['p'] * entities_total[entity]['r'] / (
                    entities_total[entity]['p'] + entities_total[entity]['r']) \
                if entities_total[entity]['p'] + entities_total[entity]['r'] != 0 else 0
            print('\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t{}'.format(entities_total[entity]['p'], entities_total[entity]['r'],
                                                              entities_total[entity]['f1'], entity))
            entity_dict = {'precision': entities_total[entity]['p'], 'recall': entities_total[entity]['r'],
                           'f1-score': entities_total[entity]['f1'], 'support': ''}
            entity_micro_prf_dict[entity] = entity_dict
            TP += entities_total[entity]['TP']
            S += entities_total[entity]['S']
            G += entities_total[entity]['G']
            macro_p = entities_total[entity]['p']
            macro_r = entities_total[entity]['r']
            macro_f1 = entities_total[entity]['f1']

        macro_p_new = macro_p / len(entities_total)
        macro_r_new = macro_r / len(entities_total)
        macro_f1_new = macro_f1 / len(entities_total)
        p = TP / S if S != 0 else 0
        r = TP / G if G != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        print('\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t\tmacro avg'.format(macro_p_new, macro_r_new, macro_f1_new))

        print('\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t\tmicro avg'.format(p, r, f1))
        print('--------------------------------------------------')
        entity_micro_prf_dict['micro avg'] = {'precision': p, 'recall': r, 'f1-score': f1, 'support': ''}

        labels = []
        for index, label in enumerate(self.tag_vocab.itos):
            labels.append(label)
        labels.remove('O')
        prf_dict = classification_report(tag_true_all, tag_pred_all, labels=labels, output_dict=True)
        # print(classification_report(tag_true_all, tag_pred_all, labels=labels))
        return entity_micro_prf_dict, prf_dict

    def _evaluate(self, sentence=None, tag_true=None, tag_pred=None):
        """
        先对true进行还原成 [{}] 再对pred进行还原成 [{}]
        :param tag_true: list[]
        :param tag_pred: list[]
        :return:
        """
        true_list = self._build_list_dict(len(tag_true), tag_true, sentence)
        pred_list = self._build_list_dict(len(tag_pred), tag_pred, sentence)
        # entities = {
        #     '疾病_预防': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_阶段': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_就诊科室': {'TP': 0, 'S': 0, 'G': 0},
        #     '其他_同义词': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_辅助治疗': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_化疗': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_放射治疗': {'TP': 0, 'S': 0, 'G': 0},
        #     '其他治疗_同义词': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_手术治疗': {'TP': 0, 'S': 0, 'G': 0},
        #     '手术治疗_同义词': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_实验室检查': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_影像学检查': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_辅助检查': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_组织学检查': {'TP': 0, 'S': 0, 'G': 0},
        #     '检查_同义词': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_内窥镜检查': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_筛查': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_多发群体': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_发病率': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_发病年龄': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_多发地区': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_发病性别倾向': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_死亡率': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_多发季节': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_传播途径': {'TP': 0, 'S': 0, 'G': 0},
        #     '流行病学_同义词': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_同义词': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_并发症': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_病理分型': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_相关（导致）': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_鉴别诊断': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_相关（转化）': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_相关（症状）': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_临床表现': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_治疗后症状': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_侵及周围组织转移的症状': {'TP': 0, 'S': 0, 'G': 0},
        #     '症状_同义词': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_病因': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_高危因素': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_风险评估因素': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_病史': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_遗传因素': {'TP': 0, 'S': 0, 'G': 0},
        #     '社会学_同义词': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_发病机制': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_病理生理': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_药物治疗': {'TP': 0, 'S': 0, 'G': 0},
        #     '药物_同义词': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_发病部位': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_外侵部位': {'TP': 0, 'S': 0, 'G': 0},
        #     '部位_同义词': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_预后状况': {'TP': 0, 'S': 0, 'G': 0},
        #     '疾病_预后生存率': {'TP': 0, 'S': 0, 'G': 0},
        #     '预防_其他': {'TP': 0, 'S': 0, 'G': 0},
        #     '阶段_其他': {'TP': 0, 'S': 0, 'G': 0},
        #     '就诊科室_其他': {'TP': 0, 'S': 0, 'G': 0},
        #     '同义词_其他': {'TP': 0, 'S': 0, 'G': 0},
        #     '辅助治疗_其他治疗': {'TP': 0, 'S': 0, 'G': 0},
        #     '化疗_其他治疗': {'TP': 0, 'S': 0, 'G': 0},
        #     '放射治疗_其他治疗': {'TP': 0, 'S': 0, 'G': 0},
        #     '同义词_其他治疗': {'TP': 0, 'S': 0, 'G': 0},
        #     '手术治疗_手术治疗': {'TP': 0, 'S': 0, 'G': 0},
        #     '同义词_手术治疗': {'TP': 0, 'S': 0, 'G': 0},
        #     '实验室检查_检查': {'TP': 0, 'S': 0, 'G': 0},
        #     '影像学检查_检查': {'TP': 0, 'S': 0, 'G': 0},
        #     '辅助检查_检查': {'TP': 0, 'S': 0, 'G': 0},
        #     '组织学检查_检查': {'TP': 0, 'S': 0, 'G': 0},
        #     '同义词_检查': {'TP': 0, 'S': 0, 'G': 0},
        #     '内窥镜检查_检查': {'TP': 0, 'S': 0, 'G': 0},
        #     '筛查_检查': {'TP': 0, 'S': 0, 'G': 0},
        #     '多发群体_流行病学': {'TP': 0, 'S': 0, 'G': 0},
        #     '发病率_流行病学': {'TP': 0, 'S': 0, 'G': 0},
        #     '发病年龄_流行病学': {'TP': 0, 'S': 0, 'G': 0},
        #     '多发地区_流行病学': {'TP': 0, 'S': 0, 'G': 0},
        #     '发病性别倾向_流行病学': {'TP': 0, 'S': 0, 'G': 0},
        #     '死亡率_流行病学': {'TP': 0, 'S': 0, 'G': 0},
        #     '多发季节_流行病学': {'TP': 0, 'S': 0, 'G': 0},
        #     '传播途径_流行病学': {'TP': 0, 'S': 0, 'G': 0},
        #     '同义词_流行病学': {'TP': 0, 'S': 0, 'G': 0},
        #     '同义词_疾病': {'TP': 0, 'S': 0, 'G': 0},
        #     '并发症_疾病': {'TP': 0, 'S': 0, 'G': 0},
        #     '病理分型_疾病': {'TP': 0, 'S': 0, 'G': 0},
        #     '相关（导致）_疾病': {'TP': 0, 'S': 0, 'G': 0},
        #     '鉴别诊断_疾病': {'TP': 0, 'S': 0, 'G': 0},
        #     '相关（转化）_疾病': {'TP': 0, 'S': 0, 'G': 0, },
        #     '相关（症状）_疾病': {'TP': 0, 'S': 0, 'G': 0},
        #     '临床表现_症状': {'TP': 0, 'S': 0, 'G': 0},
        #     '治疗后症状_症状': {'TP': 0, 'S': 0, 'G': 0},
        #     '侵及周围组织转移的症状_症状': {'TP': 0, 'S': 0, 'G': 0},
        #     '同义词_症状': {'TP': 0, 'S': 0, 'G': 0},
        #     '病因_社会学': {'TP': 0, 'S': 0, 'G': 0},
        #     '高危因素_社会学': {'TP': 0, 'S': 0, 'G': 0},
        #     '风险评估因素_社会学': {'TP': 0, 'S': 0, 'G': 0},
        #     '病史_社会学': {'TP': 0, 'S': 0, 'G': 0},
        #     '遗传因素_社会学': {'TP': 0, 'S': 0, 'G': 0},
        #     '同义词_社会学': {'TP': 0, 'S': 0, 'G': 0},
        #     '发病机制_社会学': {'TP': 0, 'S': 0, 'G': 0},
        #     '病理生理_社会学': {'TP': 0, 'S': 0, 'G': 0},
        #     '药物治疗_药物': {'TP': 0, 'S': 0, 'G': 0},
        #     '同义词_药物': {'TP': 0, 'S': 0, 'G': 0},
        #     '发病部位_部位': {'TP': 0, 'S': 0, 'G': 0},
        #     '转移部位_部位': {'TP': 0, 'S': 0, 'G': 0},
        #     '外侵部位_部位': {'TP': 0, 'S': 0, 'G': 0},
        #     '同义词_部位': {'TP': 0, 'S': 0, 'G': 0},
        #     '预后状况_预后': {'TP': 0, 'S': 0, 'G': 0},
        #     '预后生存率_预后': {'TP': 0, 'S': 0, 'G': 0},
        # }
        entities = {
            '疾病': {'TP': 0, 'S': 0, 'G': 0},
            '检查': {'TP': 0, 'S': 0, 'G': 0},
            '部位': {'TP': 0, 'S': 0, 'G': 0},
            '其他': {'TP': 0, 'S': 0, 'G': 0},
            '其他治疗': {'TP': 0, 'S': 0, 'G': 0},
            '手术治疗': {'TP': 0, 'S': 0, 'G': 0},
            '流行病学': {'TP': 0, 'S': 0, 'G': 0},
            '症状': {'TP': 0, 'S': 0, 'G': 0},
            '社会学': {'TP': 0, 'S': 0, 'G': 0},
            '药物': {'TP': 0, 'S': 0, 'G': 0},
            '预防_其他': {'TP': 0, 'S': 0, 'G': 0},
            '阶段_其他': {'TP': 0, 'S': 0, 'G': 0},
            '就诊科室_其他': {'TP': 0, 'S': 0, 'G': 0},
            '同义词_其他': {'TP': 0, 'S': 0, 'G': 0},
            '辅助治疗_其他治疗': {'TP': 0, 'S': 0, 'G': 0},
            '化疗_其他治疗': {'TP': 0, 'S': 0, 'G': 0},
            '放射治疗_其他治疗': {'TP': 0, 'S': 0, 'G': 0},
            '同义词_其他治疗': {'TP': 0, 'S': 0, 'G': 0},
            '手术治疗_手术治疗': {'TP': 0, 'S': 0, 'G': 0},
            '同义词_手术治疗': {'TP': 0, 'S': 0, 'G': 0},
            '实验室检查_检查': {'TP': 0, 'S': 0, 'G': 0},
            '影像学检查_检查': {'TP': 0, 'S': 0, 'G': 0},
            '辅助检查_检查': {'TP': 0, 'S': 0, 'G': 0},
            '组织学检查_检查': {'TP': 0, 'S': 0, 'G': 0},
            '同义词_检查': {'TP': 0, 'S': 0, 'G': 0},
            '内窥镜检查_检查': {'TP': 0, 'S': 0, 'G': 0},
            '筛查_检查': {'TP': 0, 'S': 0, 'G': 0},
            '多发群体_流行病学': {'TP': 0, 'S': 0, 'G': 0},
            '发病率_流行病学': {'TP': 0, 'S': 0, 'G': 0},
            '发病年龄_流行病学': {'TP': 0, 'S': 0, 'G': 0},
            '多发地区_流行病学': {'TP': 0, 'S': 0, 'G': 0},
            '发病性别倾向_流行病学': {'TP': 0, 'S': 0, 'G': 0},
            '死亡率_流行病学': {'TP': 0, 'S': 0, 'G': 0},
            '多发季节_流行病学': {'TP': 0, 'S': 0, 'G': 0},
            '传播途径_流行病学': {'TP': 0, 'S': 0, 'G': 0},
            '同义词_流行病学': {'TP': 0, 'S': 0, 'G': 0},
            '同义词_疾病': {'TP': 0, 'S': 0, 'G': 0},
            '并发症_疾病': {'TP': 0, 'S': 0, 'G': 0},
            '病理分型_疾病': {'TP': 0, 'S': 0, 'G': 0},
            '相关（导致）_疾病': {'TP': 0, 'S': 0, 'G': 0},
            '鉴别诊断_疾病': {'TP': 0, 'S': 0, 'G': 0},
            '相关（转化）_疾病': {'TP': 0, 'S': 0, 'G': 0, },
            '相关（症状）_疾病': {'TP': 0, 'S': 0, 'G': 0},
            '临床表现_症状': {'TP': 0, 'S': 0, 'G': 0},
            '治疗后症状_症状': {'TP': 0, 'S': 0, 'G': 0},
            '侵及周围组织转移的症状_症状': {'TP': 0, 'S': 0, 'G': 0},
            '同义词_症状': {'TP': 0, 'S': 0, 'G': 0},
            '病因_社会学': {'TP': 0, 'S': 0, 'G': 0},
            '高危因素_社会学': {'TP': 0, 'S': 0, 'G': 0},
            '风险评估因素_社会学': {'TP': 0, 'S': 0, 'G': 0},
            '病史_社会学': {'TP': 0, 'S': 0, 'G': 0},
            '遗传因素_社会学': {'TP': 0, 'S': 0, 'G': 0},
            '同义词_社会学': {'TP': 0, 'S': 0, 'G': 0},
            '发病机制_社会学': {'TP': 0, 'S': 0, 'G': 0},
            '病理生理_社会学': {'TP': 0, 'S': 0, 'G': 0},
            '药物治疗_药物': {'TP': 0, 'S': 0, 'G': 0},
            '同义词_药物': {'TP': 0, 'S': 0, 'G': 0},
            '发病部位_部位': {'TP': 0, 'S': 0, 'G': 0},
            '转移部位_部位': {'TP': 0, 'S': 0, 'G': 0},
            '外侵部位_部位': {'TP': 0, 'S': 0, 'G': 0},
            '同义词_部位': {'TP': 0, 'S': 0, 'G': 0},
            '预后状况_预后': {'TP': 0, 'S': 0, 'G': 0},
            '预后生存率_预后': {'TP': 0, 'S': 0, 'G': 0},
        }
        for true in true_list:
            label_type = true['label_type']
            entities[label_type]['G'] += 1
        for pred in pred_list:
            label_type = pred['label_type']
            label_name = pred['name']
            label_start = pred['start_pos']
            label_end = pred['end_pos']
            entities[label_type]['S'] += 1
            for true in true_list:
                if label_type == true['label_type'] and label_name == true['name'] and label_start == true[
                    'start_pos'] and label_end == true['end_pos']:
                    entities[label_type]['TP'] += 1
        # self.record_pred_info(sentence=sentence, true_list=true_list, pred_list=pred_list,
        #                       path='./result/classification_report/{}/pred_info.txt'.format(config.experiment_name))
        return entities

    def _build_list_dict(self, _len, _list, sentence):
        build_list = []
        tag_dict = {'疾病': '疾病',
                    '其他': '其他',
                    '检查': '检查',
                    '部位': '部位',
                    '其他治疗': '其他治疗',
                    '手术治疗': '手术治疗',
                    '流行病学': '流行病学',
                    '症状': '症状',
                    '社会学': '社会学',
                    '药物': '药物',
                    '预防_其他': '预防_其他',
                    '阶段_其他': '阶段_其他',
                    '就诊科室_其他': '就诊科室_其他',
                    '同义词_其他': '同义词_其他',
                    '辅助治疗_其他治疗': '辅助治疗_其他治疗',
                    '化疗_其他治疗': '化疗_其他治疗',
                    '放射治疗_其他治疗': '放射治疗_其他治疗',
                    '同义词_其他治疗': '同义词_其他治疗',
                    '手术治疗_手术治疗': '手术治疗_手术治疗',
                    '同义词_手术治疗': '同义词_手术治疗',
                    '实验室检查_检查': '实验室检查_检查',
                    '影像学检查_检查': '影像学检查_检查',
                    '辅助检查_检查': '辅助检查_检查',
                    '组织学检查_检查': '组织学检查_检查',
                    '同义词_检查': '同义词_检查',
                    '内窥镜检查_检查': '内窥镜检查_检查',
                    '筛查_检查': '筛查_检查',
                    '多发群体_流行病学': '多发群体_流行病学',
                    '发病率_流行病学': '发病率_流行病学',
                    '发病年龄_流行病学': '发病年龄_流行病学',
                    '多发地区_流行病学': '多发地区_流行病学',
                    '发病性别倾向_流行病学': '发病性别倾向_流行病学',
                    '死亡率_流行病学': '死亡率_流行病学',
                    '多发季节_流行病学': '多发季节_流行病学',
                    '传播途径_流行病学': '传播途径_流行病学',
                    '同义词_流行病学': '同义词_流行病学',
                    '同义词_疾病': '同义词_疾病',
                    '并发症_疾病': '并发症_疾病',
                    '病理分型_疾病': '病理分型_疾病',
                    '相关（导致）_疾病': '相关（导致）_疾病',
                    '鉴别诊断_疾病': '鉴别诊断_疾病',
                    '相关（转化）_疾病': '相关（转化）_疾病',
                    '相关（症状）_疾病': '相关（症状）_疾病',
                    '临床表现_症状': '临床表现_症状',
                    '治疗后症状_症状': '治疗后症状_症状',
                    '侵及周围组织转移的症状_症状': '侵及周围组织转移的症状_症状',
                    '同义词_症状': '同义词_症状',
                    '病因_社会学': '病因_社会学',
                    '高危因素_社会学': '高危因素_社会学',
                    '风险评估因素_社会学': '风险评估因素_社会学',
                    '病史_社会学': '病史_社会学',
                    '遗传因素_社会学': '遗传因素_社会学',
                    '同义词_社会学': '同义词_社会学',
                    '发病机制_社会学': '发病机制_社会学',
                    '病理生理_社会学': '病理生理_社会学',
                    '药物治疗_药物': '药物治疗_药物',
                    '同义词_药物': '同义词_药物',
                    '发病部位_部位': '发病部位_部位',
                    '转移部位_部位': '转移部位_部位',
                    '外侵部位_部位': '外侵部位_部位',
                    '同义词_部位': '同义词_部位',
                    '预后状况_预后': '预后状况_预后',
                    '预后生存率_预后': '预后生存率_预后',
                    }
        i = 0

        while i < _len:
            if _list[i][0] == 'B':
                label_type = _list[i][2:]
                start_pos = i
                end_pos = start_pos
                if end_pos != _len - 1:
                    if end_pos + 1 < _len and _list[end_pos + 1][0] != 'I':
                        end_pos += 1
                    else:
                        while end_pos + 1 < _len and _list[end_pos + 1][0] == 'I' and _list[end_pos + 1][
                                                                                      2:] == label_type:
                            end_pos += 1
                build_list.append(
                    {'name': ''.join(sentence[start_pos:end_pos + 1]), 'start_pos': start_pos, 'end_pos': end_pos,
                     'label_type': tag_dict[label_type]})
                i = end_pos + 1
            else:
                i += 1
        result = []
        for dict1 in build_list:
            if dict1 not in result:
                result.append(dict1)
        return result

    # def predict(self, path=None, model_name=None, save_path=None):
    #     logger.info('Start predict data...')
    #     if path is None:
    #         path = config.test_path
    #         model_name = self.config.save_model_path + 'model_{}.pkl'.format(self.config.experiment_name)
    #         save_path = self.config.result_path + 'result_all_{}.txt'.format(self.config.experiment_name)
    #     train_data = tool.load_data(config.train_path, config.is_bioes)
    #     dev_data = tool.load_data(config.dev_path, config.is_bioes)
    #     logger.info('Finished load data...')
    #     word_vocab = tool.get_text_vocab(train_data, dev_data)
    #     tag_vocab = tool.get_tag_vocab(train_data, dev_data)
    #     logger.info('Finished build vocab...')
    #
    #     model = self.init_model(self.config, word_vocab, len(word_vocab), len(tag_vocab), config.vector_path)
    #     model.load_state_dict(torch.load(model_name))
    #     all_subject_type = []
    #     all_predicate = []
    #     all_shcemas = []
    #     with open(self.config.shcemas_path, 'r', encoding='utf-8') as fs:
    #         for jsonstr in fs.readlines():
    #             jsonstr = json.loads(jsonstr)
    #             all_shcemas.append(jsonstr)
    #             all_subject_type.append(jsonstr['subject_type'])
    #             all_predicate.append(jsonstr['predicate'])
    #         fs.close()
    #     all_predicate = set(all_predicate)
    #     with open(save_path, 'w', encoding='utf-8') as fw:
    #         with open(path, 'r', encoding='utf-8') as f:
    #             lines = f.readlines()
    #             for jsonstr in tqdm(lines):
    #                 line = json.loads(jsonstr)
    #                 text = torch.tensor(
    #                     numpy.array([word_vocab.stoi[word] for word in line['text']], dtype='int64')).unsqueeze(
    #                     1).expand(len(line['text']), self.config.batch_size).to(DEVICE)
    #                 text_len = torch.tensor(numpy.array([len(line['text'])], dtype='int64')).expand(
    #                     self.config.batch_size).to(DEVICE)
    #                 results = model(text, text_len)
    #                 results_len = []
    #                 for result in results:
    #                     n = 0
    #                     for i in result:
    #                         if i != 1:
    #                             n += 1
    #                     results_len.append(n)
    #                 index = results_len.index(max(results_len))
    #                 result = results[index]
    #                 tag_pred = [tag_vocab.itos[k] for k in result]
    #                 sentence = line['text'].replace('\n', '')
    #                 sentencelist = [x for x in sentence]
    #                 sentence, dict_list = _get_single(sentence, tag_pred, all_predicate)
    #                 new = []
    #                 for list in dict_list:
    #                     for shcemas in all_shcemas:
    #                         if list['subject_type'] == shcemas['subject_type'] and list['predicate'] == shcemas[
    #                             'predicate']:
    #                             result_dict = {
    #                                 'predicate': list['predicate'],
    #                                 "subject": "".join(sentence[list['subject_start']:list['subject_end'] + 1]),
    #                                 'subject_type': list['subject_type'],
    #                                 "object": {
    #                                     "@value": "".join(sentence[list['object_start']:list['object_end'] + 1])},
    #                                 'object_type': {"@value": shcemas['object_type']}
    #                             }
    #                             if result_dict not in new:
    #                                 new.append(result_dict)
    #                 if sum([item.count('。') for item in sentence]) >= 2:
    #                     for item in new:
    #                         item['Combined'] = True
    #                 else:
    #                     for item in new:
    #                         item['Combined'] = False
    #
    #                 # new_l = []
    #                 # set1 = set()  # 定义一个新列表和集合
    #                 # for d in new:
    #                 #     if tuple(d.items()) not in set1:  # 将字典的items转成元组，判断是否在集合中
    #                 #         set1.add(tuple(d.items()))
    #                 #         new_l.append(d)
    #                 if len(new) == 0:
    #                     new = [{
    #                         "Combined": '',
    #                         'predicate': '',
    #                         "subject": '',
    #                         'subject_type': '',
    #                         "object": {"@value": ""},
    #                         'object_type': {"@value": ""},
    #                     }]
    #                     pred_dict = {
    #                         "text": ''.join(sentence),
    #                         "spo_list": new,
    #                     }
    #                 else:
    #
    #                     pred_dict = {
    #                         "text": ''.join(sentence),
    #                         "spo_list": new,
    #                     }
    #                 fw.write(json.dumps(pred_dict, ensure_ascii=False) + '\n')
    #     f.close()
    #     fw.close()

    # def _get_single(sentence, tag_pred, all_predicate):
    #     sentence = sentence
    #     result_list = []
    #     for index, tag in zip(range(len(tag_pred)), tag_pred):
    #         if tag[0] == 'B':
    #             start = index
    #             end = index
    #             label_type = tag[2:]
    #             if end != len(tag_pred) - 1:
    #                 while tag_pred[end + 1][0] == 'I' and tag_pred[end + 1][2:] == label_type:
    #                     # while tag_pred[end + 1][0] == 'M' or tag_pred[end + 1][0] == 'E' and tag_pred[end + 1][2:] == label_type:
    #                     end += 1
    #             result_list.append({'start': start,
    #                                 'end': end,
    #                                 'lable_type': label_type
    #                                 })
    #     predicate = []
    #     subject_type = []
    #     for i, item in enumerate(result_list):
    #         if item['lable_type'] in all_predicate or item['lable_type'] == '手术治疗_p':
    #             if item['lable_type'] == '手术治疗_p':
    #                 item['lable_type'] = item['lable_type'].replace('手术治疗_p', '手术治疗')
    #                 predicate.append(item)
    #             else:
    #                 predicate.append(item)
    #         else:
    #             subject_type.append(item)
    #     dict_list = []
    #     for i in subject_type:
    #         for j in predicate:
    #             dict_list.append({'subject_type': i['lable_type'],
    #                               'predicate': j['lable_type'],
    #                               'subject_start': i['start'],
    #                               'subject_end': i['end'],
    #                               'object_start': j['start'],
    #                               'object_end': j['end']
    #                               })
    #     return sentence, dict_list
    def predict(self, path=None, model_name=None, save_path=None):
        logger.info('Start predict data...')
        if path is None:
            path = config.test_path
            model_name = self.config.save_model_path + 'model_{}.pkl'.format(self.config.experiment_name)
            save_path = self.config.result_path + 'result_all_{}.txt'.format(self.config.experiment_name)
        train_data = tool.load_data(config.train_path, config.is_bioes)
        dev_data = tool.load_data(config.dev_path, config.is_bioes)
        logger.info('Finished load data...')
        word_vocab = tool.get_text_vocab(train_data, dev_data)
        tag_vocab = tool.get_tag_vocab(train_data, dev_data)
        logger.info('Finished build vocab...')

        model = self.init_model(self.config, word_vocab, len(word_vocab), len(tag_vocab), config.vector_path)
        model.load_state_dict(torch.load(model_name))
        all_subject_type = []
        all_predicate = []
        all_shcemas = []
        with open(self.config.shcemas_path, 'r', encoding='utf-8') as fs:
            for jsonstr in fs.readlines():
                jsonstr = json.loads(jsonstr)
                all_shcemas.append(jsonstr)
                all_subject_type.append(jsonstr['subject_type'])
                all_predicate.append(jsonstr['predicate'])
            fs.close()
        all_predicate = set(all_predicate)
        with open(save_path, 'w', encoding='utf-8') as fw:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for jsonstr in tqdm(lines):
                    line = json.loads(jsonstr)
                    text = torch.tensor(
                        numpy.array([word_vocab.stoi[word] for word in line['text']], dtype='int64')).unsqueeze(
                        1).expand(len(line['text']), self.config.batch_size).to(DEVICE)
                    text_len = torch.tensor(numpy.array([len(line['text'])], dtype='int64')).expand(
                        self.config.batch_size).to(DEVICE)
                    results = model(text, text_len)
                    results_len = []
                    for result in results:
                        n = 0
                        for i in result:
                            if i != 1:
                                n += 1
                        results_len.append(n)
                    index = results_len.index(max(results_len))
                    result = results[index]
                    tag_pred = [tag_vocab.itos[k] for k in result]
                    sentence = line['text'].replace('\n', '')
                    sentencelist = [x for x in sentence]
                    sentence, dict_list = _get_single(sentence, tag_pred, all_predicate)
                    new = []
                    for list in dict_list:
                        for shcemas in all_shcemas:
                            if list['subject_type'] == shcemas['subject_type'] and list['predicate'] == shcemas[
                                'predicate']:
                                result_dict = {
                                    'predicate': list['predicate'],
                                    "subject": "".join(sentence[list['subject_start']:list['subject_end'] + 1]),
                                    'subject_type': list['subject_type'],
                                    "object": {
                                        "@value": "".join(sentence[list['object_start']:list['object_end'] + 1])},
                                    'object_type': {"@value": shcemas['object_type']}
                                }
                                if result_dict not in new:
                                    new.append(result_dict)
                    if sum([item.count('。') for item in sentence]) >= 2:
                        for item in new:
                            item['Combined'] = True
                    else:
                        for item in new:
                            item['Combined'] = False

                    # new_l = []
                    # set1 = set()  # 定义一个新列表和集合
                    # for d in new:
                    #     if tuple(d.items()) not in set1:  # 将字典的items转成元组，判断是否在集合中
                    #         set1.add(tuple(d.items()))
                    #         new_l.append(d)
                    if len(new) == 0:
                        new = [{
                            "Combined": '',
                            'predicate': '',
                            "subject": '',
                            'subject_type': '',
                            "object": {"@value": ""},
                            'object_type': {"@value": ""},
                        }]
                        pred_dict = {
                            "text": ''.join(sentence),
                            "spo_list": new,
                        }
                    else:

                        pred_dict = {
                            "text": ''.join(sentence),
                            "spo_list": new,
                        }
                    fw.write(json.dumps(pred_dict, ensure_ascii=False) + '\n')
        f.close()
        fw.close()


def _get_single(sentence, tag_pred, all_predicate):
    sentence = sentence
    result_list = []
    for index, tag in zip(range(len(tag_pred)), tag_pred):
        if tag[0] == 'B':
            start = index
            end = index
            label_type = tag[2:]
            if end != len(tag_pred) - 1:
                while tag_pred[end + 1][0] == 'I' and tag_pred[end + 1][2:] == label_type:
                    # while tag_pred[end + 1][0] == 'M' or tag_pred[end + 1][0] == 'E' and tag_pred[end + 1][2:] == label_type:
                    end += 1
            result_list.append({'start': start,
                                'end': end,
                                'lable_type': label_type
                                })
    predicate = []
    subject_type = []
    for i, item in enumerate(result_list):
        if item['lable_type'] in all_predicate or item['lable_type'] == '手术治疗_p':
            if item['lable_type'] == '手术治疗_p':
                item['lable_type'] = item['lable_type'].replace('手术治疗_p', '手术治疗')
                predicate.append(item)
            else:
                predicate.append(item)
        else:
            subject_type.append(item)
    dict_list = []
    for i in subject_type:
        for j in predicate:
            dict_list.append({'subject_type': i['lable_type'],
                              'predicate': j['lable_type'],
                              'subject_start': i['start'],
                              'subject_end': i['end'],
                              'object_start': j['start'],
                              'object_end': j['end']
                              })
    return sentence, dict_list


if __name__ == '__main__':
    CHIP2020_NER = CHIP2020_RE()
    CHIP2020_NER.train()
    CHIP2020_NER.predict()
