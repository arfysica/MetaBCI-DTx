import torch
import numpy as np
import random

def norm_length(x, len_norm):
    len_x = x.shape[2]
    if len_x > len_norm:
        index = np.random.randint(len_x - len_norm)
    return x[:, :, index:index+1000]

# 判断模型结构是否一致
def isSame(m1, m2):
    assert type(m1) == type(m2)
    assert len(list(m1.children())) == len(list(m2.children()))
    for c1, c2 in zip(m1.children(), m2.children()):
        isSame(c1, c2)

def support_query_split(X, y, k_shot=5):
    unique_labels = np.unique(y)
    index_dict = {i: [] for i in unique_labels}
    for i, label in enumerate(y):
        index_dict[label].append(i)
    sampled_index = np.array([random.sample(index_dict[i], k=k_shot) for i in unique_labels]).reshape(-1)
    random.shuffle(sampled_index)
    return X[sampled_index], X, y[sampled_index], y