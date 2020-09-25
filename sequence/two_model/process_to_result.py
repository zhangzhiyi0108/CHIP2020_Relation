#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : process_to_result
# @Author   : ZhiYi Zhang
# @Time     : 2020/9/22 11:36
import json

import torch

def get_result(subject_path, object_path, save_path, schemas_path):
    all_schemas = []
    with open(schemas_path, 'r', encoding='utf-8') as fm:
        for schame in fm.readlines():
            schame = json.loads(schame)
            all_schemas.append(schame)
        fm.close()
        all_subject = []
        all_predicate = []
        with open(subject_path, 'r', encoding='utf-8') as fs:
            with open(object_path, 'r', encoding='utf-8') as fo:
                with open(save_path, 'w', encoding='utf-8') as fw:
                        for subjects in fs.readlines():
                            subject_line = json.loads(subjects)
                            # subjects = subject_line['spo_list']
                            all_subject.append(subject_line)
                        for predicate in fo.readlines():
                            predicate_line= json.loads(predicate)
                            # objects = object_line['spo_list']
                            all_predicate.append(predicate_line)
                        for i,item in enumerate(all_subject):
                            subjects = all_subject[i]['spo_list']
                            predicates = all_predicate[i]['spo_list']
                            new = []
                            for subject in subjects:
                                for ob in predicates:
                                    for schema in all_schemas:
                                        if subject['subject_type'] == schema['subject_type'] and ob['predicate'] == schema['predicate']:
                                            predicate =  schema['predicate']
                                            pred = {
                                                "predicate": predicate,
                                                "subject":subject['subject'],
                                                'subject_type': subject['subject_type'],
                                                "object":{'@value':ob['object']['@value']} ,
                                                'object_type':{'@value':schema['object_type']},
                                            }
                                            new.append(pred)
                            if sum([item.count('。') for item in all_subject[i]['text']]) >= 2:
                                for item in new:
                                    item['Combined'] = True
                            else:
                                for item in new:
                                    item['Combined'] = False
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
                                    "text": all_subject[i]['text'],
                                    "spo_list": new,
                                }
                            else:

                                pred_dict = {
                                    "text": all_subject[i]['text'],
                                    "spo_list": new,
                                }
                            fw.write(json.dumps(pred_dict, ensure_ascii=False) + '\n')
                fo.close()
            fs.close()
        fw.close()
def get_result_subject_replace(subject_path, object_path, save_path, schemas_path):
    all_schemas = []
    with open(schemas_path, 'r', encoding='utf-8') as fm:
        for schame in fm.readlines():
            schame = json.loads(schame)
            all_schemas.append(schame)
        fm.close()
        all_subject = []
        all_object = []
        with open(subject_path, 'r', encoding='utf-8') as fs:
            with open(object_path, 'r', encoding='utf-8') as fo:
                with open(save_path, 'w', encoding='utf-8') as fw:
                        for subjects in fs.readlines():
                            subject_line = json.loads(subjects)
                            # subjects = subject_line['spo_list']
                            all_subject.append(subject_line)
                        for predicate in fo.readlines():
                            predicate_line= json.loads(predicate)
                            # objects = object_line['spo_list']
                            all_object.append(predicate_line)
                        for i,item in enumerate(all_subject):
                            subjects = all_subject[i]['spo_list']
                            object = all_object[i]['spo_list']
                            new = []
                            for subject in subjects:
                                for ob in object:
                                    for schema in all_schemas:
                                        if subject['predicate'] == schema['predicate'] and ob['object_type'] == schema['object_type']:
                                            predicate =  schema['predicate']
                                            pred = {
                                                "predicate": predicate,
                                                "subject":subject['subject'],
                                                'subject_type': schema['subject_type'],
                                                "object":{'@value':ob['object']['@value']} ,
                                                'object_type':{'@value':ob['object_type']},
                                            }
                                            new.append(pred)
                            if sum([item.count('。') for item in all_subject[i]['text']]) >= 2:
                                for item in new:
                                    item['Combined'] = True
                            else:
                                for item in new:
                                    item['Combined'] = False
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
                                    "text": all_subject[i]['text'],
                                    "spo_list": new,
                                }
                            else:

                                pred_dict = {
                                    "text": all_subject[i]['text'],
                                    "spo_list": new,
                                }
                            fw.write(json.dumps(pred_dict, ensure_ascii=False) + '\n')
                fo.close()
            fs.close()
        fw.close()



if __name__ == '__main__':
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    object_config_file = 're_object_config.yml'
    subject_config_file = 're_subject_config.yml'
    import dynamic_yaml

    with open(object_config_file, mode='r', encoding='UTF-8') as f:
        object_config = dynamic_yaml.load(f)
    object_config.device = device
    with open(subject_config_file, mode='r', encoding='UTF-8') as f:
        subject_config = dynamic_yaml.load(f)
    get_result(subject_config.data.chip_relation.result_path,object_config.data.chip_relation.result_path,subject_config.data.chip_relation.save_path,subject_config.data.chip_relation.shcemas_path)
