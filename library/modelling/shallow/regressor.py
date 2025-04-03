from datetime import datetime

from library.dataset import Dataset
from library.modelling.shallow.base import ModelAssesment

import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class RegressorAssesment(ModelAssesment):
  """
  This class is used to assess the performance of a regressor.
  """
  def __init__(self, dataset: Dataset, results_path: str, results_columns: list, columns_to_check_duplicates: list) -> None:
    super().__init__(dataset, results_path, results_columns, columns_to_check_duplicates)

  def __set_assesment__(self, y_actual: pd.Series, y_pred: pd.Series, plot: bool = True):
    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    if plot:
      print(f"\t\t => MAE: {mae}")
      print(f"\t\t => MSE: {mse}")
      print(f"\t\t => R2: {r2}")
    return mae, mse, r2

  def evaluate_regressor(self, modelsToExclude: list = [], saveModelResults: bool = False, dataToWrite: dict = {}):
    if saveModelResults:
      assert dataToWrite is not None, "dataToWrite must be provided if saveModelResults is True"

    for modelName, model in self.models.items():
      if modelName in modelsToExclude or model["test_predictions"] is None or model["val_predictions"] is None:
        continue
      print(f"Evaluating {modelName}")
      print(f"\t => VALIDATION ASSESMENT:")
      y_actual_val = self.dataset.y_val_encoded if hasattr(self.dataset, 'y_train_encoded') else self.dataset.y_validation
      y_pred_val = model["val_predictions"]
      mae_val, mse_val, r2_val = self.__set_assesment__(y_actual_val, y_pred_val)
      print(f"\t => TEST ASSESMENT:")
      y_actual_test = self.dataset.y_test_encoded if hasattr(self.dataset, 'y_train_encoded') else self.dataset.y_test
      y_pred_test = model["test_predictions"]
      mae_test, mse_test, r2_test = self.__set_assesment__(y_actual_test, y_pred_test)
      self.models[modelName]["metrics"] = {
        "mae_val": mae_val,
        "mse_val": mse_val,
        "r2_val": r2_val,
        "mae_test": mae_test,
        "mse_test": mse_test,
        "r2_test": r2_test
      }
      if saveModelResults:
        dataToWrite["modelName"] = modelName
        dataToWrite["mse"] = mse_test
        dataToWrite["mae"] = mae_test
        dataToWrite["r2"] = r2_test
        dataToWrite["timeStamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.get_model_results_saved(dataToWrite=dataToWrite, featuresUsed=self.dataset.X_train.columns.tolist())