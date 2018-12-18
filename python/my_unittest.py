import naive_bayes as nb
import pandas as pd
import numpy as np


filepath = '../Zoo.arff'
alg = nb.NaiveBayes(filepath, 'type')


def test_print_dataset():
    print('Dataset')
    print(alg.df)
#
# def test_classifier():
#     print(alg.get_classifier())

def test_get_prior_probability_distribution():
    prior_prob_dist = alg.get_prior_probability_distribution()
    print(prior_prob_dist)


def test_predict():
    print('Predict')

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
    result = alg.predict(df_test)
    print(result)

def test():
    test_print_dataset()
    # test_classifier()
    test_get_prior_probability_distribution()
    test_predict()

test()
