import pandas as pd
import numpy as np
import arff
import math

class NaiveBayes:

    def __init__(self, arff_file_path, target_name):
        data_file = open(arff_file_path, 'r')
        data = arff.load(data_file)

        attributes = {}
        for attribute in data['attributes']:
            attributes[attribute[0]] = attribute[1]
        rows = data['data']

        self.df = pd.DataFrame(rows, columns=attributes.keys())

        for feature_name in self.df.drop(columns='animal_name'):
            self.df[feature_name] = self.df[feature_name].astype(np.int8)

        self.target_name = target_name
        self.target_column = self.df[target_name]
        self.target_unique_values = pd.unique(self.target_column)

    def get_prior_probability_distribution(self):
        prior_prob_dist = {}
        for feature_name in self.df:
            description = self.df[feature_name].value_counts()
            total = description.sum()
            description = description.apply(lambda x: x / total)
            prior_prob_dist[feature_name] = description
        return prior_prob_dist

    def calc_bernoulli_prob(self, x, prob_dist):
        return math.pow(prob_dist[x], x) * math.pow(1 - prob_dist[x], 1 - x)

    def predict(self, test_set: pd.DataFrame):
        assert isinstance(test_set, pd.DataFrame)

        prior_prob_dist = self.get_prior_probability_distribution()

        for index, row in test_set.drop(columns=['animal_name', 'type']).iterrows():
            row = row.astype(np.int8)
            likelihood = []
            for target_unique_value in self.target_unique_values:
                target_unique_value = int(target_unique_value)
                prob_target = prior_prob_dist[self.target_name][target_unique_value]
                for feature_name in self.df.drop(columns=['animal_name', 'type']):
                    prob = self.calc_bernoulli_prob(row[feature_name], prior_prob_dist[feature_name])
                    prob_target = prob_target * prob
                likelihood.append(prob_target)
            # print("Prob: ", likelihood)
            # print("Predict target {} has max prob is {}".format(
            #     self.target_unique_values[likelihood.index(max(likelihood))],
            #     max(likelihood)
            # ))
            test_set.at[index, 'type'] = self.target_unique_values[likelihood.index(max(likelihood))]
        return test_set
