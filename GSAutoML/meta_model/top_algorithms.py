

class TopPerformers:
    def __init__(self, performance_df):
        self.performance_df = performance_df

    def get_top_performers(self, closest_dataset_name: str, datasets_names_column: str):
        """
        Select algorithms with their configurations of a specific datasets
        :param closest_dataset_name: name of the dataset to select algorithms trained on
        :param datasets_names_column: name of column that contain dataset name in the algorithms performance dataframe

        return top_algorithms: dataframe contains the top performing algorithms with
                               their hyperparameters and evaluation scores
        """

        top_algorithms = self.performance_df[self.performance_df[datasets_names_column] == closest_dataset_name]
        print(top_algorithms)
        top_algorithms = top_algorithms.drop_duplicates()
        return top_algorithms
