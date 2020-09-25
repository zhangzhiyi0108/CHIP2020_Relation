# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/9/16 16:21

import torch
from openpyxl import Workbook
from abc import ABC, abstractmethod
from fastNLP import AccuracyMetric, SpanFPreRecMetric, Vocabulary
from base.evaluation.base_sequence_evaluator import BaseSeqEvaluator


class CommonSeqEvaluator(BaseSeqEvaluator):

    def __init__(self, tag_vocab, config):
        super(CommonSeqEvaluator, self).__init__()
        self._config = config
        self._vocab = Vocabulary()
        self._vocab.add_word_lst(tag_vocab.stoi.keys())
        self._evaluator = SpanFPreRecMetric(self._vocab,  only_gross=False, f_type=config.evaluation.type)
        self._pad_index = tag_vocab.stoi['<pad>']

    def _change_type(self, pred, target):
        seq_len = torch.tensor([len(text) for text in pred])
        max_len = max(seq_len)
        for text in pred:
            if len(text) < max_len:
                text.extend([self._pad_index for i in range(max_len - len(text))])
        pred = torch.tensor(pred).to(self._config.device)
        return pred, target, seq_len

    def evaluate(self, pred, target):
        # 送入batch数据
        pred, target, seq_len = self._change_type(pred, target)
        self._evaluator.evaluate(pred, target, seq_len)

    def _get_eval_result(self):
        # 统计所有batch数据的结果
        eval_dict = self._evaluator.get_metric()
        if self._config.data.chip_relation.use_chip_relation:
            names = list(set([label[2:] for label in self._vocab.word2idx.keys()][3:]))
            if '其他' in names:
                names.remove('其他')
        else:
            names = list(set([label[2:] for label in self._vocab.word2idx.keys()][3:]))
        head = ['label', '   precision', '   recall', '  F1_score']
        table = []
        table.append(head)
        for i in range(len(names)):
            ps = str(round(eval_dict['pre-' + names[i].lower()], 3))
            rs = str(round(eval_dict['rec-' + names[i].lower()], 3))
            f1s = str(round(eval_dict['f-' + names[i].lower()], 3))
            table.append([names[i], ps, rs, f1s])
        ps = str(round(eval_dict['pre'], 3))
        rs = str(round(eval_dict['rec'], 3))
        f1s = str(round(eval_dict['f'], 3))
        table.append(['{}_average'.format(self._config.evaluation.type), ps, rs, f1s])
        return eval_dict, table

    def get_eval_output(self):
        # 外部获取结果接口,并且可以配置是否打印（eval结果保存暂时默认保存）
        result, table = self._get_eval_result()
        if self._config.evaluation.is_display:
            self._print_table(table)
        self._write_csv(table)
        return result

    def _print_table(self, List):
        # 展示
        k = len(List)
        v = len(List[0])
        for i in range(k):
            for j in range(v):
                print(List[i][j].rjust(14), end=' ')
            print()

    def _write_csv(self, table):
        wb = Workbook()
        ws = wb['Sheet']
        for line in range(1,len(table)+1):
            for column in range(1, 5):
                ws.cell(line, column, table[line-1][column-1])
        save_path = self._config.learn.dir.saved + '/eval_result.xlsx'
        wb.save(save_path)