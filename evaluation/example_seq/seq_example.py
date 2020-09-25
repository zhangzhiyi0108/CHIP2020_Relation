# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/3/29 14:15

from evaluation.eval_seq import Evaluator

golden_path = './gloden.txt'
predicted_path = './pred.txt'
index = 1
is_report = 1
entities = ['LOCATION', 'INTEGER', 'ORDINAL']
labels = ['B-INTEGER', 'I-INTEGER', 'B-ORDINAL', 'I-ORDINAL', 'B-LOCATION', 'I-LOCATION']

eval_seq = Evaluator(golden_path, predicted_path, entities, index, labels)
entities_json = eval_seq.evaluate_entities(is_report)
labels_json = eval_seq.evaluate_labels(is_report)

print('\n********************* Entities json *********************')
print(entities_json)
print('\n********************* Labels json *********************')
print(labels_json)
