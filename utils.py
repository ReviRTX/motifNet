import torch
import numpy as np
import time,os
from sklearn.metrics import f1_score
from torch import nn
import scipy.stats as stats
from sklearn.metrics import accuracy_score, precision_score, recall_score


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def r2(x,y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return r_value**2

def cal_acc(x,y):
    cp = 1-np.abs(x.values-y)
    acc = len(np.where(cp==1)[0]) / len(x)
    return acc


def cal_accuracy_precision_recall(y_true, y_pred, pos_label=1):
    return {
            "accuracy": float("%.5f" % accuracy_score(y_true=y_true, y_pred=y_pred)),
            "precision": float("%.5f" % precision_score(y_true=y_true, y_pred=y_pred, pos_label=pos_label)),
            "recall": float("%.5f" % recall_score(y_true=y_true, y_pred=y_pred, pos_label=pos_label))
            }

#计算F1score
def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return f1_score(y_true, y_pre)

#打印时间
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


# 调整学习率
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

#多标签使用类别权重
class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return (loss * self.weights).mean()

class binary_BCE(nn.Module):
    def __init__(self):
        super(binary_BCE, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return loss.mean()
