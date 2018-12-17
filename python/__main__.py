import random
import numpy as np
import pandas as pd


def split_dataset(dataset: list, ratio):
    train_size = len(dataset) * ratio

    train_data = []
    test_data = list(dataset)

    while len(train_data) < train_size:
        rand_index = random.randrange(len(test_data))
        obj = test_data.pop(rand_index)
        train_data.append(obj)

    return train_data, test_data

predictor = NaiveBayes('../Zoo.arff', 'type')
table = predictor.prob_table()
for feature_name in table.keys():
    print(table[feature_name])



