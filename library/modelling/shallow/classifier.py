
from datetime import datetime

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import concurrent.futures



from library.dataset import Dataset
from library.modelling.shallow.base import ModelAssesment

class ClassifierAssesment(ModelAssesment):
  """
  This class is used to assess the performance of a classifier.
  """
  def __init__(self, dataset: Dataset, results_path: str, results_columns: list, columns_to_check_duplicates: list) -> None:
    super().__init__(dataset, results_path, results_columns, columns_to_check_duplicates)
  
  def __set_assesment__(self, 
                        y_actual: pd.Series,
                        y_pred: pd.Series, 
                        plot: bool = True):
    """
    Assesment of the model in a given set 

    Parameters
    ----------
      y_actual : pd.Series
        The actual labels
      y_pred : pd.Series
        The predicted labels
      plot : bool
        Whether to plot the results

    Returns
    -------
      tuple
      The classification report and the confusion matrix
    """
    class_report = classification_report(y_actual, y_pred, output_dict=True) # F1 score, precision, recall for each class
    conf_matrix = confusion_matrix(y_actual, y_pred)
    if plot:
      print(f"Validation Classification Report: \n{class_report}")
      sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
      plt.xlabel("Predicted Label")
      plt.ylabel("True Label")
      plt.title("Confusion Matrix")
      plt.show()
    return class_report, conf_matrix
  
  def __model_assesment__(self, 
                          modelName: str, 
                          model: dict, 
                          plot: bool = True, 
                          save_results: bool = True, 
                          dataToWrite: dict = {}):
    """
    Evaluate the classifier and save the results
    """
    print(f"Evaluating {modelName}")
    print(f"\t => VALIDATION ASSESMENT:")
    y_actual_val = self.dataset.y_val_encoded
    y_pred_val = model["val_predictions"]
    assert y_actual_val is not None and y_pred_val is not None, f"y_actual_val or y_pred_val is None for {modelName}. Model name is {modelName}, model object is {model}"
    self.__set_assesment__(y_actual_val, y_pred_val, plot)
    print(f"\t => TEST ASSESMENT:")
    y_actual_test = self.dataset.y_test_encoded
    y_pred_test = model["test_predictions"]
    assert y_actual_test is not None and y_pred_test is not None, f"y_actual_test or y_pred_test is None for {modelName}. Model name is {modelName}, model object is {model}"
    class_report, confusion_matrix = self.__set_assesment__(y_actual_test, y_pred_test, plot)
    self.models[modelName]["metrics"] = {
        "class_report": class_report,
        "confusion_matrix": confusion_matrix
      }
    if save_results:
      dataToWrite["hyperParameters"] = model["hyperParameters"]
      dataToWrite["modelName"] = modelName
      dataToWrite["accuracy"] = class_report["accuracy"]
      dataToWrite["f1"] = class_report["weighted avg"]["f1-score"]
      dataToWrite["precision"] = class_report["weighted avg"]["precision"]
      dataToWrite["recall"] = class_report["weighted avg"]["recall"]
      dataToWrite["timeStamp"] =  datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      self.get_model_results_saved(dataToWrite=dataToWrite, featuresUsed=self.dataset.X_train_encoded.columns.tolist())
    return modelName, model

  def evaluate_classifiers(self, 
                           plot: bool = True, 
                           modelsToExclude: list = [],
                           save_results: bool = True, 
                           dataToWrite: dict = {}):
    """
    Evaluate the classifier and save the results

    Parameters
    ----------
      plot : bool
        Whether to plot the results
      modelsToExclude : list
        The models to exclude from the evaluation
      save_results : bool
        Whether to save the results
      dataToWrite : dict
        The data to write to the results file

    Returns
    -------
      None
    """
    
    if save_results:
      assert dataToWrite is not None, "dataToWrite must be provided if save_results is True"
    for modelName, model in self.models.items():
      if modelName in modelsToExclude or model["test_predictions"] is None or model["val_predictions"] is None:
        modelsToExclude.append(modelName)
    with concurrent.futures.ProcessPoolExecutor() as executor:
      # Submit all model fitting tasks to the executor
        future_to_assesment = {executor.submit(self.__model_assesment__, modelName, model, plot, save_results, dataToWrite): modelName for modelName, model in self.models.items() if modelName not in modelsToExclude}
        
        for future in concurrent.futures.as_completed(future_to_assesment):
            modelName, model = future.result() 
            self.models[modelName] = model # update results
        print("All models have been assesed and results saved.")
    return self.models
     
