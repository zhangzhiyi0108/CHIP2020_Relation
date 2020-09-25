from __future__ import unicode_literals, print_function, division

import logging
import time

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig()


def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(12, 12)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm_perc, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    from matplotlib import pyplot as plt
    sns.set()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', vmin=0, vmax=100, ax=ax, cmap='YlOrRd', linewidths=.5, cbar=True)
    plt.savefig(filename)


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        print('......begin {0:8s}......'.format(f.__name__))
        result = f(*args, **kw)
        te = time.time()
        print('......finish {0:8s}, took:{1:.4f} sec   ......'.format(f.__name__, te - ts))
        return result

    return timed


def idx(list_a, list_b):
    """
    ind
    :param list_a: 
    :param list_b: 
    :return: 
    """
    result_idx = []
    result_values = []
    for tmp in list_a:
        index = np.where(tmp == list_b)
        result_idx.extend(index[0])
        result_values.extend(list_b[index])

    result_idx = np.asarray(result_idx).astype(int).squeeze()
    result_values = np.asarray(result_values).astype(int).squeeze()

    return result_idx, result_values


def compute_accuracy(label_true, label_predict):
    """
    :param label_true:
    :param label_predict:
    :return:
    """
    acc = 0.0
    unique_labels = np.unique(label_true)
    for l in unique_labels:
        indics = np.nonzero(label_true == l)[0]
        acc += accuracy_score(label_true[indics], label_predict[indics])
    acc = acc / unique_labels.shape[0]
    return acc


def compute_whole_accuracy(label_true, label_predict):
    """
    :param label_true:
    :param label_predict:
    :return:
    """
    acc = accuracy_score(label_true, label_predict)
    return acc


def computer_class_acc(model, device, test_batch_size, test_feature, test_label, attribute_class, class_id):
    feature_dataset = TensorDataset(
        torch.from_numpy(test_feature), torch.from_numpy(test_label)
    )
    test_loader = DataLoader(feature_dataset, batch_size=test_batch_size, shuffle=False)
    # get all predicted features and ground true labels
    re_batch_labels_total = list()
    mapped_features = list()
    for batch_feature, batch_labels in test_loader:
        # batch_size = batch_labels.shape[0]
        batch_feature = Variable(batch_feature).to(device).float()  # 32*1024

        # get predicted features
        feature_encoded, feature_decoded, _, _, output_labels = model(
            batch_feature, None, batch_labels
        )
        mapped_features.extend(feature_encoded)
        re_batch_labels_total.extend(output_labels.detach().cpu().numpy())
    re_batch_labels_total = np.asarray(re_batch_labels_total)

    # get all predicted attributes

    attribute_class_dataset = TensorDataset(
        torch.from_numpy(attribute_class)
    )
    attribute_class_dataloader = DataLoader(
        attribute_class_dataset, batch_size=test_batch_size, shuffle=False)
    mapped_attribute_classes = list()
    for batch_att in attribute_class_dataloader:
        batch_att = Variable(batch_att).to(device).float()  # 32*1024
        _, _, attribute_encoded, attribute_decoded, _ = model(None, batch_att, None)
        mapped_attribute_classes.extend(attribute_encoded)

    pass
