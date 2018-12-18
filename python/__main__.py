import random
import naive_bayes as nb
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

def classify():
    classifier = nb.NaiveBayes('../Zoo.arff', 'type')

    columns = pd.Index(['animal_name', 'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed',
              'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize', 'type'])
    rows = [
        ['NameIsSecret', 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 4, 1, 0, 1, np.nan],
        ['NameIsSecret', 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 2, 1, 1, 0, np.nan],
        ['NameIsSecret', 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, np.nan],
        ['NameIsSecret', 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, np.nan],
        ['NameIsSecret', 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 4, 1, 0, 0, np.nan]
    ]

    df_test = pd.DataFrame(rows, columns=columns)
    # print(df_test.head())
    result = classifier.predict(df_test)
    print(result)

if __name__ == '__main__':
    classify()




