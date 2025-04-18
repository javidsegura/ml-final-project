from datetime import datetime

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import concurrent.futures


from library.phases.model_selection.model.model import Model
from library.phases.dataset.dataset import Dataset

class Classifier(Model):
      def __init__(self,  modelName: str, model_sklearn: object, results_header: list[str], dataset: Dataset):
            self.dataset = dataset   
            super().__init__(modelName, model_sklearn, results_header, dataset)
      

      def __set_assesment__(self, 
                        y_actual: pd.Series,
                        y_pred: pd.Series,
                        modelName: str):
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

          return class_report
      
      def evaluate(self, modelName: str, current_phase: str):
            print(f"Evaluating {modelName} in {current_phase} phase")
            assert current_phase in ["pre", "in", "post"], "Current phase must be one of the tuning states"
            if current_phase == "pre" or current_phase == "in":
                  y_actual = self.dataset.y_val
                  y_pred = self.tuning_states[current_phase].assesment["predictions_val"]
            elif current_phase == "post":
                  y_actual = self.dataset.y_test
                  y_pred = self.tuning_states[current_phase].assesment["predictions_test"]
            else:
                  raise ValueError("Invalid phase")
            
            assert y_actual is not None, "y_actual is None"
            assert y_pred is not None, "y_pred is None"

            class_report = self.__set_assesment__(y_actual, y_pred, modelName)

            accuracy = class_report["accuracy"]
            f1_score = class_report["weighted avg"]["f1-score"]
            precision = class_report["weighted avg"]["precision"]
            recall = class_report["weighted avg"]["recall"]
            results = {
                  "f1-score": f1_score,
                  "precision": precision,
                  "recall": recall,
                  "accuracy": accuracy
            }
            print(f"METRIC RESULTS FOR {modelName} => F1: {f1_score}, Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}")
            # Storing results to assesment attribute
            for metric, value in results.items():
                  self.tuning_states[current_phase].assesment[metric] = value
            
      def evaluate_training(self, modelName: str):
            raise NotImplementedError("Training evaluation not implemented for classifier")
