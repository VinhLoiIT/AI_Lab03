import random
import naive_bayes as nb
import numpy as np
import pandas as pd
import arff


def split_dataset(dataset: list, ratio):
    train_size = len(dataset) * ratio

    train_data = []
    test_data = list(dataset)

    while len(train_data) < train_size:
        rand_index = random.randrange(len(test_data))
        obj = test_data.pop(rand_index)
        train_data.append(obj)

    return train_data, test_data

def parse_arff_file(arff_file_path):
    data_file = open(arff_file_path, 'r')
    data = arff.load(data_file)

    attributes = {}
    for attribute in data['attributes']:
        attributes[attribute[0]] = attribute[1]
    rows = data['data']

    df = pd.DataFrame(rows, columns=attributes.keys())

    for feature_name in df.drop(columns=['animal_name', 'type']):
        df[feature_name] = df[feature_name].astype(np.int8)
    return df

def classify():
    dataset = parse_arff_file('../Zoo.arff')
    testset = parse_arff_file('../Zoo_test.arff')

    classifier = nb.NaiveBayes(dataset)
    result = classifier.predict(testset)
    print(result)


if __name__ == '__main__':
    classify()
