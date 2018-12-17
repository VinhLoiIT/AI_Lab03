import naive_bayes as nb

filepath = '../Zoo.arff'
alg = nb.NaiveBayes(filepath, 'type')


def test_print_dataset():
    print('Dataset')
    print(alg.df)


def test_count_table():
    print('Counting table')
    count_table = alg.count_table()
    for feature_name in count_table.keys():
        print('------------------------------')
        print('Feature: ', feature_name)
        print('------------------------------')
        print(count_table[feature_name])


def test_prob_table():
    print('Probability table')
    prob_table = alg.prob_table()
    for feature_name in prob_table.keys():
        print('------------------------------')
        print('Feature: ', feature_name)
        print('------------------------------')
        print(prob_table[feature_name])


def test():
    test_print_dataset()
    test_count_table()
    test_prob_table()


test()
