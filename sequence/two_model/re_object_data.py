import json
# -*- coding: utf-8 -*-
import torch
from torchtext.data import Field, BucketIterator, Dataset, Example
from torchtext.datasets import SequenceTaggingDataset

from common.data.common_data_loader import CommonDataLoader
from common.util.utils import timeit


def tokenizer(token):
    return [k for k in token]


START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD_TAG = "<PAD>"
UNK_TAG = "<PAD>"


class SequenceDataLoader(CommonDataLoader):

    def __init__(self, data_config):
        super(SequenceDataLoader, self).__init__(data_config)
        self.__build_field()
        self._load_data()

        pass

    def __build_field(self):
        self.TEXT = Field(sequential=True, use_vocab=True, tokenize=tokenizer, include_lengths=True)
        self.TAG = Field(sequential=True, use_vocab=True, tokenize=tokenizer, is_target=True)
        self._fields = [
            ('text', self.TEXT), ('tag', self.TAG)
        ]
        self._fields_test = [('text', self.TEXT)]
        pass

    @timeit
    def _load_data(self):
        self.train_data = REDataset(path=self._config.data.chip_relation.train_path, fields=self._fields)
        self.valid_data = REDataset(path=self._config.data.chip_relation.valid_path, fields=self._fields)
        self.test_data = REDataset(path=self._config.data.chip_relation.test_path, fields=self._fields_test)
        self.__build_vocab(self.train_data, self.valid_data, self.test_data)
        self.__build_iterator(self.train_data, self.valid_data, self.test_data)
        pass

    def __build_vocab(self, *dataset):
        """
        :param dataset: train_data, valid_data, test_data
        :return: text_vocab, tag_vocab
        """
        self.TEXT.build_vocab(*dataset)
        self.TAG.build_vocab(*dataset[:-1])
        self.word_vocab = self.TEXT.vocab
        self.tag_vocab = self.TAG.vocab
        pass

    def __build_iterator(self, *dataset):
        self._train_iter = BucketIterator(
            dataset[0], batch_size=self._config.data.train_batch_size, shuffle=True,
            sort_key=lambda x: len(x.text), sort_within_batch=True, device=self._config.device)

        self._valid_iter = BucketIterator(
            dataset[1], batch_size=self._config.data.train_batch_size, shuffle=False,
            sort_key=lambda x: len(x.text), sort_within_batch=True, device=self._config.device)

        self._test_iter = BucketIterator(
            dataset[2], batch_size=self._config.data.train_batch_size, shuffle=False,
            sort_key=lambda x: len(x.text), sort_within_batch=True, device=self._config.device)

    def load_train(self):
        return self._train_iter
        pass

    def load_test(self):
        return self._test_iter
        pass

    def load_valid(self):
        return self._valid_iter
        pass


class REDataset(Dataset):
    def __init__(self, path=None, fields=None, encoding="utf-8", **kwargs):
        config_file = 're_object_config.yml'
        import dynamic_yaml

        with open(config_file, mode='r', encoding='UTF-8') as f:
            config = dynamic_yaml.load(f)
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for jsonstr in f.readlines():
                jsonstr = json.loads(jsonstr)
                if path == config.data.chip_relation.test_path:
                    text = list(jsonstr['text'])
                    examples.append(Example.fromlist((text, ''), fields))
                else:
                    text_list, tag_list = self._get_list(jsonstr)
                    examples.append(Example.fromlist((text_list, tag_list), fields))
        f.close()
        super(REDataset, self).__init__(examples, fields, **kwargs)

    def _get_list(self, jsonstr):
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

            # subject_start_poses = self._find_all_index(text, subject)
            # subject_end_poses = [start + len(subject) - 1 for start in subject_start_poses]
            # for j in range(len(subject_start_poses)):
            #     x = subject_start_poses[j]
            #     tag_list[x] = 'B_' + subject_type
            #     x += 1
            #     while x <= subject_end_poses[j]:
            #         tag_list[x] = 'I_' + subject_type
            #         x += 1
            object_start_poses = self._find_all_index(text, object)
            object_end_poses = [start + len(object) - 1 for start in object_start_poses]
            for j in range(len(object_start_poses)):
                x = object_start_poses[j]
                tag_list[x] = 'B_' + object_type
                x += 1
                while x <= object_end_poses[j]:
                    tag_list[x] = 'I_' + object_type
                    x += 1
        text_list = list(text)
        return text_list, tag_list

    def _find_all_index(self, str1, str2):
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


if __name__ == '__main__':
    config_file = 'seq_config.yml'
    import dynamic_yaml

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(config_file, mode='r', encoding='UTF-8') as f:
        config = dynamic_yaml.load(f)
    data_loader = SequenceDataLoader(config)
    for batch, batch_data in enumerate(data_loader.load_train(), 0):
        text = batch_data.text
        print("batch = {}".format(batch))
        for idx, txt in enumerate(text, 0):
            print("idx={},text ={} ".format(idx, txt))

    pass
