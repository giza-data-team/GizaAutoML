import pandas as pd
from scipy.spatial.distance import cityblock


class DistanceCalculator:
    def __init__(self, meta_features_df):
        """
        Initialize the DistanceCalculator with a DataFrame of normalized meta features.

        Parameters:
            meta_features_df (pd.DataFrame): A DataFrame containing normalized meta features.
        """
        self.meta_features_df = meta_features_df

    def calculate_distances(self, query_dataset_id, dataset_id_column='dataset_id',
                            columns_to_exclude=[], top=1):
        """
        Calculate the L1 distances between a specific dataset ID and all other rows in the DataFrame.

        Parameters:
            query_dataset_id (str): The specific dataset ID for which distances will be calculated.
            dataset_id_column (str, optional): The name of the column containing dataset IDs.
            columns_to_exclude (list, optional): List of column names to exclude from distance calculation. Default is ['id'].
            top (int, optional): The number of top closest datasets to return. Default is None (return all).

        Returns:
            pd.DataFrame: A DataFrame with columns with top N 'dataset_id' and 'distance', sorted by distance.
        """
        # Select the row corresponding to the given query_dataset_id
        #todo: convert int to str
        target_row = self.meta_features_df[self.meta_features_df[dataset_id_column] == query_dataset_id].drop(columns=[dataset_id_column] + columns_to_exclude)

        # Calculate L1 distances between the target row and all other rows using cityblock (L1) distance
        distances = self.meta_features_df.drop(columns=[dataset_id_column] + columns_to_exclude).apply(
            lambda row: cityblock(target_row.iloc[0], row), axis=1
        )

        # Create a DataFrame with dataset IDs and their corresponding distances
        result_df = pd.DataFrame({dataset_id_column: self.meta_features_df[dataset_id_column], 'distance': distances})

        # Exclude the query dataset from the result
        result_df = result_df[result_df[dataset_id_column] != query_dataset_id]

        # Sort the DataFrame by distance in ascending order
        result_df = result_df.sort_values(by='distance')
        print(result_df)

        # Optionally return only the top N closest datasets
        if top is not None:
            result_df = result_df.head(top)

        return result_df
