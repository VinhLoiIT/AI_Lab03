import pandas as pd
import numpy as np

class NaiveBayes:

    def __init__(self, dataset: pd.DataFrame):
        self.df = dataset
        self.target_name = 'type'
        self.target_column = self.df['type']
        self.target_unique_values = pd.unique(self.target_column)

    def get_prior_probability_distribution(self):
        prob_dist = {}
        for feature_name in self.df.drop(columns='type'):
            prior_prob_dist = self.df.groupby(['type'])[feature_name].value_counts().unstack(fill_value=0).stack()
            prior_prob_dist = prior_prob_dist.apply(lambda x: x + 1)
            prior_prob_dist = prior_prob_dist / prior_prob_dist.sum(level=0)
            prob_dist[feature_name] = prior_prob_dist

        target_prob_dist = self.df['type'].value_counts()
        target_prob_dist = target_prob_dist.apply(lambda x: (x + 1) / (sum(target_prob_dist) + len(pd.unique(target_prob_dist))))
        prob_dist['type'] = target_prob_dist

        return prob_dist

    def calc_bernoulli_prob(self, feature_name, feature_value, target, prob_dist):
        event_prob = prob_dist[feature_name][target][feature_value]
        return event_prob

    def predict(self, test_set: pd.DataFrame):
        assert isinstance(test_set, pd.DataFrame)

        prob_dist = self.get_prior_probability_distribution()
        test_set = test_set.drop(columns='animal_name')
        test_set['type'] = test_set['type'].astype(str)

        prob_list = []

        for index, row in test_set.drop(columns=['type']).iterrows():
            row = row.astype(np.int8)
            likelihood = []
            for target_unique_value in self.target_unique_values:
                prob_target = prob_dist[self.target_name][target_unique_value]
                for feature_name in test_set.drop(columns=['type']):
                    prob = self.calc_bernoulli_prob(feature_name, row[feature_name], target_unique_value, prob_dist)
                    prob_target = prob_target * prob
                likelihood.append(prob_target)
            total = sum(likelihood)
            likelihood = [x/total * 100 for x in likelihood]
            classified = self.target_unique_values[likelihood.index(max(likelihood))]

            test_set.at[index, 'type'] = classified
            prob_list.append(max(likelihood))

        predicted_series = test_set['type']
        prob_series = pd.Series(prob_list)
        result = pd.DataFrame({'Predicted': predicted_series, 'Probability': prob_series})

        return result
