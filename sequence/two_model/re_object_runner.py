# -*- coding: utf-8 -*-
import json
import random

import torch
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
from common.evaluation.common_sequence_evaluator import CommonSeqEvaluator
from common.runner.common_runner import CommonRunner
from sequence.two_model.re_object_data import  SequenceDataLoader
from sequence.two_model.re_loss import SequenceCRFLoss
from sequence.two_model.re_model import BiLSTMCRF,TransformerEncoderModel

RANDOM_SEED = 2020

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class SubjectRunner(CommonRunner):

    def __init__(self, seq_config):
        super(SubjectRunner, self).__init__(seq_config)

        self._dict_result = {'max_F': 0.0}
        self._valid_log_fields = ['alpha', 'beta', 'episode', 'P', 'R', 'F']
        # init the metric
        self._metric = CommonSeqEvaluator(self.tag_vocab, seq_config)
        self._max_f1 = -1
        pass

    def _build_data(self):
        self._dataloader = SequenceDataLoader(self._config)

        self.word_vocab = self._dataloader.word_vocab
        self.tag_vocab = self._dataloader.tag_vocab

        self._config.data.num_vocab = len(self.word_vocab)
        self._config.data.num_tag = len(self.tag_vocab)

        self._train_dataloader = self._dataloader.load_train()
        self._valid_dataloader = self._dataloader.load_valid()
        self._test_dataloader = self._dataloader.load_test()
        pass

    def _build_model(self):
        self._model = TransformerEncoderModel(self._config).to(self._config.device)
        pass

    def _build_loss(self):
        self._loss = SequenceCRFLoss(self._config)
        pass

    def valid(self):
        model = self._load_checkpoint()
        self._model.eval()
        for dict_input in tqdm(self._valid_dataloader):
            dict_output = self._model(dict_input)
            self._display_output(dict_output)
            # send batch pred and target
            self._metric.evaluate(dict_output['outputs'], dict_output['target_sequence'].T)
        # get the result
        result = self._metric.get_eval_output()
        pass

    def _valid(self, episode, valid_log_writer, summary_writer):
        print("begin validating epoch {}...".format(episode + 1))
        # switch to evaluate mode
        self._model.eval()
        for dict_input in tqdm(self._valid_dataloader):
            dict_output = self._model(dict_input)
            # self._display_output(dict_output)
            # send batch pred and target
            self._metric.evaluate(dict_output['outputs'], dict_output['target_sequence'].T)
        # get the result
        result = self._metric.get_eval_output()
        if self._max_f1 < result['f']:
            self._max_f1 = result['f']
            self._save_checkpoint(episode)
            pass
        pass

    def _display_output(self, dict_outputs):
        batch_data = dict_outputs['batch_data']

        word_vocab = batch_data.dataset.fields['text'].vocab
        tag_vocab = batch_data.dataset.fields['tag'].vocab

        batch_input_sequence = dict_outputs['input_sequence'].T
        batch_output_sequence = np.asarray(dict_outputs['outputs']).T
        batch_target_sequence = dict_outputs['target_sequence'].T

        result_format = "{}\t{}\t{}\n"
        for input_sequence, output_sequence, target_sequence in zip(
                batch_input_sequence, batch_output_sequence, batch_target_sequence):
            this_result = ""
            for word, tag, target in zip(input_sequence, output_sequence, target_sequence):
                if word != "<pad>":
                    this_result += result_format.format(
                        word_vocab.itos[word], tag_vocab.itos[tag], tag_vocab.itos[target]
                    )
            print(this_result + '\n')
        pass

    def test(self):
        model = self._load_checkpoint()
        self._model.eval()
        for dict_input in tqdm(self._test_dataloader):
            dict_output = self._model(dict_input)
            self._display_output(dict_output)
            # send batch pred and target
            self._metric.evaluate(dict_output['outputs'], dict_output['target_sequence'].T)
        # get the result
        result = self._metric.get_eval_output()

    def predict(self):
        model = self._load_checkpoint()
        self._model.eval()
        all_object_type=[]
        all_predicate = []
        all_shcemas = []
        with open(self._config.data.chip_relation.result_path, 'w', encoding='utf-8') as fw:
            with open(self._config.data.chip_relation.shcemas_path, 'r', encoding='utf-8') as f:
                for jsonstr in f.readlines():
                    jsonstr = json.loads(jsonstr)
                    all_shcemas.append(jsonstr)
                    all_object_type.append(jsonstr['object_type'])
                    all_predicate.append(jsonstr['predicate'])
                all_predicate = set(all_predicate)
                all_object_type = set(all_object_type)
                for dict_input in tqdm(self._test_dataloader):
                    dict_output = self._model(dict_input)
                    results = dict_output['outputs']
                    sentences = np.asarray(dict_output['input_sequence'].T.cpu())
                    # batch 句子
                    batch_sentences = []
                    for sentence in sentences:
                        sentence = [self.word_vocab.itos[k] for k in sentence]
                        while '<pad>' in sentence:
                            sentence.remove('<pad>')
                        batch_sentences.append(sentence)
                    # batch 预测
                    tag_preds = []
                    for result in results:
                        tag_pred = [self.tag_vocab.itos[k] for k in result]
                        tag_preds.append(tag_pred)
                    for i, tag_preds_single in enumerate(tag_preds):
                        sentence = batch_sentences[i]
                        sentence,dict_list = self._get_single(sentence,tag_preds_single,all_predicate)
                        new = []
                        for list in dict_list:
                            for predicate in all_predicate:
                                if list['object_type'] == all_object_type :
                                    result_dict = {

                                            "object": {"@value":"".join(sentence[list['object_type_start']:list['object_type_end']+1])},
                                            'object_type': list['object_type'],
                                        }
                                    new.append(result_dict)
                        if len(new) == 0:
                            new =[{
                                        "Combined": '',
                                        "object": {"@value":""},
                                        'predicate':'',
                                    }]
                            pred_dict = {
                                "text": ''.join(sentence),
                                "spo_list": new,
                            }
                        else:

                            pred_dict = {
                                "text" : ''.join(sentence),
                                "spo_list" : new,
                            }
                        fw.write(json.dumps(pred_dict,ensure_ascii=False) + '\n')
            f.close()
        fw.close()
    def _get_single(self,batch_sentences,tag_pred,all_predicate):
        sentence = batch_sentences
        result_list = []
        for index, tag in zip(range(len(tag_pred)), tag_pred):
            if tag[0] == 'B':
                start = index
                end = index
                label_type = tag[2:]
                if end != len(tag_pred) - 1:
                    while tag_pred[end + 1][0] == 'I' and tag_pred[end + 1][2:] == label_type:
                        end += 1
                result_list.append({'start': start,
                                    'end': end,
                                    'lable_type': label_type
                                    })

        object_type = []
        for i, item in enumerate(result_list):
                object_type.append(item)
        dict_list=[]
        for i in object_type:
                dict_list.append({'object_type': i['lable_type'],
                                  'object_type_start' : i['start'],
                                  'object_type_end': i['end'],
                                  })
        return sentence, dict_list

        pass

    def _display_result(self, episode):
        pass

if __name__ == '__main__':
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_file = 're_object_config.yml'
    import dynamic_yaml

    with open(config_file, mode='r', encoding='UTF-8') as f:
        config = dynamic_yaml.load(f)
    config.device = device

    runner = SubjectRunner(config)
    runner.train()
    runner.predict()
    pass
