

from library.dataset import Dataset

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

class FeatureEngineering:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __calculate_vif(self):
        """
        Calculates the VIF of the features.

        Returns
        -------
        pd.DataFrame
            A dataframe with the features and their VIF.
        """
        vif_data = pd.DataFrame()
        vif_data["Feature"] = self.dataset.X_train.select_dtypes(include=["number"]).columns
        vif_data["VIF"] = [variance_inflation_factor(self.dataset.X_train.values, i) for i in range(len(self.dataset.X_train.columns))]
        return vif_data
    
    def start_vif_elimination(self, threshold=10):
        """
        Starts the VIF elimination process. Eliminates in all sets.
        Note: this is computationally expensive for high-feature datasets.

        Parameters
        ----------
        threshold : float
            The threshold for the VIF.

        Returns
        -------
        None
        """
        number_of_iterations = 0
        while True:
            number_of_iterations += 1
            vif_data = self.__calculate_vif()
            print(f"VIF computed for iteration {number_of_iterations}:")
            max_vif = vif_data["VIF"].max()
            if max_vif < threshold:
                  break
            feature_to_drop = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
            self.dataset.X_train.drop(columns=[feature_to_drop], inplace=True)
            self.dataset.X_val.drop(columns=[feature_to_drop], inplace=True)
            self.dataset.X_test.drop(columns=[feature_to_drop], inplace=True)
            print(f"Dropped: {feature_to_drop}")
      
      
      
