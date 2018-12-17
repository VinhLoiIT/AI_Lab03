import pandas as pd
import arff


class NaiveBayes:

    def __init__(self, arff_file_path, target_name):
        data_file = open(arff_file_path, 'r')
        data = arff.load(data_file)

        attributes = {}
        for attribute in data['attributes']:
            attributes[attribute[0]] = attribute[1]
        rows = data['data']

        self.df = pd.DataFrame(rows, columns=attributes.keys())
        self.target = self.df[target_name]
        self.target_unique_values = pd.unique(self.target)

    def count_table(self):

        result = {}

        for feature_name in self.df.drop(columns='type'):
            frame = self.df[[feature_name, 'type']]

            rows = pd.Index(pd.unique(frame[feature_name]), name="rows")
            columns = pd.Index(pd.unique(frame['type']), name="columns")
            table = pd.DataFrame(data=None, index=rows, columns=columns)

            for feature_unique_value in pd.unique(frame[feature_name]):
                for target_unique_value in pd.unique(frame['type']):
                    filtered = frame[
                        (frame[feature_name] == feature_unique_value) & (frame['type'] == target_unique_value)]
                    count = len(filtered.index)
                    table.at[feature_unique_value, target_unique_value] = count

            result[feature_name] = table
        return result

    def prob_table(self):
        count_table_all = self.count_table()

        for feature_name in count_table_all.keys():
            count_table = count_table_all[feature_name]
            # for target_value in count_table:
            #     count_table.iloc[:, target_value].apply(lambda x: x / x.sum())
            prob_table = count_table.apply(lambda x: x / x.sum(), axis=1)
            count_table_all[feature_name] = prob_table

        return count_table_all

    def predict(self, features):
        pass
