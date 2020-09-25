# zutnlp_research

## eval_seq
Strict: exact boundary surface string match and entity type;
Exact: exact boundary match over the surface string, regardless of the type;
Partial: partial boundary match over the surface string, regardless of the type;
Type: some overlap between the system tagged entity and the gold annotation is required;

### Metrics
+ eval_seq = Evaluator(golden_path, predicted_path, index, entities, labels)
+ evaluate_entities(is_report)
+ evaluate_labels(is_report)

### Example
Running
```
ffrom evaluation.eval_seq import Evaluator

golden_path = './gloden.txt'
predicted_path = './pred.txt'
index = 1
is_report = 1
entities = ['LOCATION','INTEGER','ORDINAL']
labels= ['B-INTEGER', 'I-INTEGER', 'B-ORDINAL', 'I-ORDINAL','B-LOCATION','I-LOCATION']
eval_seq = Evaluator(golden_path, predicted_path, index, entities, labels)
entities_json = eval_seq.evaluate_entities(is_report)
labels_json = eval_seq.evaluate_labels(is_report)
print('\n********************* Entities json *********************')
print(entities_json)
print('\n********************* Labels json *********************')
print(labels_json)

# 通过声明Evaluator()对象，初始化参数几个调用方法
# 参数分别代表:
# golden_path: 测试数据路径
# predicted_path：预测结果数据路径
# index: 数据中测试lables的所在列数的索引
# is_report：是否打印可视化报告
# entities: 语料中所有实体类型(不带B,I标签的实体类型)
# labels: 语料中所有实体类型(带B,I标签的实体类型)
## 补充：Evaluator()也可以直接传入列表 如：eval_seq = Evaluator(y_test, y_pred, entities)，当传入列表时则不需要传index参数
```

results_labels:
```
********************* Entities Report *********************
        strict      precision         recall       F1_score 
      LOCATION          0.667          0.667          0.667 
       INTEGER            1.0            1.0            1.0 
       ORDINAL            1.0            1.0            1.0 
       
{'ent_type': {'correct': 2, 'incorrect': 0, 'partial': 0, 'missed': 2, 'spurious': 2, 'possible': 4, 'actual': 4, 'precision': 0.5, 'recall': 0.5, 'F1_score': 0.5}, 'partial': {'correct': 2, 'incorrect': 0, 'partial': 0, 'missed': 2, 'spurious': 2, 'possible': 4, 'actual': 4, 'precision': 0.5, 'recall': 0.5, 'F1_score': 0.5}, 'strict': {'correct': 2, 'incorrect': 0, 'partial': 0, 'missed': 2, 'spurious': 2, 'possible': 4, 'actual': 4, 'precision': 0.5, 'recall': 0.5, 'F1_score': 0.5}, 'exact': {'correct': 2, 'incorrect': 0, 'partial': 0, 'missed': 2, 'spurious': 2, 'possible': 4, 'actual': 4, 'precision': 0.5, 'recall': 0.5, 'F1_score': 0.5}}
```

results_entities:
```
********************* Labels Report *********************
              precision    recall  f1-score   support

   B-INTEGER      1.000     1.000     1.000         1
   I-INTEGER      1.000     1.000     1.000         1
   B-ORDINAL      1.000     1.000     1.000         1
   I-ORDINAL      1.000     1.000     1.000         1
  B-LOCATION      1.000     1.000     1.000         3
  I-LOCATION      1.000     0.500     0.667         2

   micro avg      1.000     0.889     0.941         9
   macro avg      1.000     0.917     0.944         9
weighted avg      1.000     0.889     0.926         9

{'LOCATION': {'ent_type': {'correct': 2, 'incorrect': 0, 'partial': 0, 'missed': 1, 'spurious': 2, 'possible': 3, 'actual': 4, 'precision': 0.5, 'recall': 0.6666666666666666, 'F1_score': 0.5714285714285715}, 'partial': {'correct': 2, 'incorrect': 0, 'partial': 0, 'missed': 1, 'spurious': 2, 'possible': 3, 'actual': 4, 'precision': 0.5, 'recall': 0.6666666666666666, 'F1_score': 0.5714285714285715}, 'strict': {'correct': 2, 'incorrect': 0, 'partial': 0, 'missed': 1, 'spurious': 2, 'possible': 3, 'actual': 4, 'precision': 0.5, 'recall': 0.6666666666666666, 'F1_score': 0.5714285714285715}, 'exact': {'correct': 2, 'incorrect': 0, 'partial': 0, 'missed': 1, 'spurious': 2, 'possible': 3, 'actual': 4, 'precision': 0.5, 'recall': 0.6666666666666666, 'F1_score': 0.5714285714285715}}, 'DATE': {'ent_type': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 2, 'possible': 0, 'actual': 2, 'precision': 0.0, 'recall': 0, 'F1_score': 0}, 'partial': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 2, 'possible': 0, 'actual': 2, 'precision': 0.0, 'recall': 0, 'F1_score': 0}, 'strict': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 2, 'possible': 0, 'actual': 2, 'precision': 0.0, 'recall': 0, 'F1_score': 0}, 'exact': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 2, 'possible': 0, 'actual': 2, 'precision': 0.0, 'recall': 0, 'F1_score': 0}}, 'ORGANIZATION': {'ent_type': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 2, 'possible': 0, 'actual': 2, 'precision': 0.0, 'recall': 0, 'F1_score': 0}, 'partial': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 2, 'possible': 0, 'actual': 2, 'precision': 0.0, 'recall': 0, 'F1_score': 0}, 'strict': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 2, 'possible': 0, 'actual': 2, 'precision': 0.0, 'recall': 0, 'F1_score': 0}, 'exact': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 2, 'possible': 0, 'actual': 2, 'precision': 0.0, 'recall': 0, 'F1_score': 0}}, 'DURATION': {'ent_type': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 2, 'possible': 0, 'actual': 2, 'precision': 0.0, 'recall': 0, 'F1_score': 0}, 'partial': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 2, 'possible': 0, 'actual': 2, 'precision': 0.0, 'recall': 0, 'F1_score': 0}, 'strict': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 2, 'possible': 0, 'actual': 2, 'precision': 0.0, 'recall': 0, 'F1_score': 0}, 'exact': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 2, 'possible': 0, 'actual': 2, 'precision': 0.0, 'recall': 0, 'F1_score': 0}}, 'PERSON': {'ent_type': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 2, 'possible': 0, 'actual': 2, 'precision': 0.0, 'recall': 0, 'F1_score': 0}, 'partial': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 2, 'possible': 0, 'actual': 2, 'precision': 0.0, 'recall': 0, 'F1_score': 0}, 'strict': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 2, 'possible': 0, 'actual': 2, 'precision': 0.0, 'recall': 0, 'F1_score': 0}, 'exact': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 2, 'possible': 0, 'actual': 2, 'precision': 0.0, 'recall': 0, 'F1_score': 0}}, 'ORDINAL': {'ent_type': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 1, 'spurious': 2, 'possible': 1, 'actual': 2, 'precision': 0.0, 'recall': 0.0, 'F1_score': 0}, 'partial': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 1, 'spurious': 2, 'possible': 1, 'actual': 2, 'precision': 0.0, 'recall': 0.0, 'F1_score': 0}, 'strict': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 1, 'spurious': 2, 'possible': 1, 'actual': 2, 'precision': 0.0, 'recall': 0.0, 'F1_score': 0}, 'exact': {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 1, 'spurious': 2, 'possible': 1, 'actual': 2, 'precision': 0.0, 'recall': 0.0, 'F1_score': 0}}}
```

## eval_gen
Evaluation code for various unsupervised automated metrics for NLG (Natural Language Generation).
It takes as input a hypothesis file, and one or more references files and outputs values of metrics.
Rows across these files should correspond to the same example.

### Metrics
+ get_bleu()
+ get_cider()
+ get_rouge()
+ get_embedding_average_cosine_similarity()
+ get_vector_extrema_cosine_similarity()
+ get_greedy_matching()

### Example
Running
```chameleon
path = 'data/'
hyp_path = path + 'hyp.txt'
ref_path = path + 'ref1.txt'
ref_paths = [path + 'ref1.txt', path + 'ref2.txt']
q_path = path + 'ques.txt'
a_path = path + 'ans.txt'
vector_path = path + 'glove.840B.300d.txt'

evaluation = Evaluation_Gen(vector_path, hyp_path, ref_path, ref_paths, q_path, a_path, n_gram=4, sigma=6.0, is_vec=False)
evaluation.get_bleu()
# evaluation.get_multi_bleu()   #此方法需在linux环境下运行
evaluation.get_rouge()
evaluation.get_cider()
# 以下三个方法需要预加载vector_path, is_vec=True
# evaluation.get_embedding_average_cosine_similarity()
# evaluation.get_vector_extrema_cosine_similarity()
# evaluation.get_greedy_matching()

# 通过声明Evaluation_Gen()对象，初始化参数几个调用各种方法
# 参数分别代表:
# vector_path: 词向量路径
# hyp_path: 候选路径
# ref_path: 参考路径
# q_path: 问题路径
# a_path: 答案路径
# is_vec: 词向量是否加载
```
gives
```chameleon
加载时间: 24.674013137817383
转换vec时间: 301.1023564338684
BLEU_1:	0.5499999999725
BLEU_2:	0.4281744192662396
BLEU_3:	0.28404282012721094
BLEU_4:	0.20114343061802883
Multi_BLEU: 20.11
ROUGE_L:0.5221037854154413
CIDEr:	1.2421919556558583
Embedding Averagr Consine Similarity: 0.7514761926251049
Vector Extrema Consine Similarity: 0.7793922011370746
Greedy Matching: 0.6404752053311417
```

### Problem
+ is_vec=True 需要加载词向量（此处使用Glove）





