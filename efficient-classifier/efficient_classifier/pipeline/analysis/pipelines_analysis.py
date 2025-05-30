from efficient_classifier.pipeline.pipeline import Pipeline
from efficient_classifier.pipeline.analysis.neuralNets.neuralNetsPlots import NeuralNetsPlots


from efficient_classifier.utils.miscellaneous.save_or_store_plot import save_or_store_plot
from efficient_classifier.utils.miscellaneous.eliminate_unsued_plots import eliminate_unused_plots

import yaml


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
import numpy as np
class PipelinesAnalysis:
      def __init__(self, pipelines: dict[str, dict[str, Pipeline]]):
            self.pipelines = pipelines
            self.encoded_map = None
            self.phase = None
            self.best_performing_model = None
            self.neural_nets_plots = None
            self.variables = yaml.load(open("efficient-classifier/efficient_classifier/configurations.yaml"), Loader=yaml.FullLoader)

            # Below you can find two attributes that are used to store the results of the analysis.
            self.merged_report_per_phase = {
                   "pre": None,
                   "in": None,
                   "post": None
            }
            self.results_per_phase = {
                   "pre": {
                          "classification_report": None,
                          "classification_report_train": None,
                          "metrics_df": None
                   },
                   "in": {
                          "classification_report": None,
                          "classification_report_train": None,
                          "metrics_df": None
                   },
                   "post": {
                          "classification_report": None,
                          "classification_report_train": None,
                          "metrics_df": None
                   }
            }


      def _create_report_dataframe(self, report: dict, modelName: str, include_training: bool = False):
            """
            Adds accuracy to the report as its own column instead of as an index (as it is by default)
            """
            accuracy = report.pop('accuracy')
            report['modelName'] = modelName + ("_train" if include_training else "")
            df = pd.DataFrame(report)
            df.loc['accuracy'] = accuracy
            df.loc['accuracy', 'modelName'] = modelName + ("_train" if include_training else "")

            return df
      
      def _add_additional_metrics_to_report(self, df: pd.DataFrame, modelName: str, additional_metrics: dict, include_training: bool = False):
            """
            Adds metrics to the report as its own columns instead of as an index (as it is by default)
            """
            if not include_training:
                  for key, value in additional_metrics["not_train"].items():
                        df.loc[key] = value
                        df.loc[key, "modelName"] = modelName
            else:
                  for key, value in additional_metrics["train"].items():
                        key = key.split("_")[0] # remove the postfix
                        df.loc[key] = value
                        df.loc[key, "modelName"] = modelName + "_train"
            
            
            return df
      
      def _compute_classification_report(self, include_training: bool = False):
            """
            Plots the classification report of a given model

            Note: adding to the class report kappa score 
            """
            assert self.phase in ["pre", "in", "post"], "Phase must be either pre, in or post"
            classification_reports = []
            for category in self.pipelines:
                  if self.phase == "in" and category == "baseline": # We do not evaluate the baseline models while tuning (cause they are not tuned)
                        continue                                                            
                  for pipeline in self.pipelines[category]:
                              for modelName in self.pipelines[category][pipeline].modelling.list_of_models: 
                                    if self.phase == "post" and category == "not_baseline" and self.best_performing_model["modelName"] != modelName:  # Only select the model that is the best if pipeline is in post mode
                                          continue
                                    if modelName not in self.pipelines[category][pipeline].modelling.models_to_exclude: # Exclude models that are not to be included
                                          additional_metrics = self.pipelines[category][pipeline].modelling.list_of_models[modelName].tuning_states[self.phase].assesment["metrics"]["additional_metrics"]
                                          if self.phase != "post":
                                                      y_pred = self.pipelines[category][pipeline].modelling.list_of_models[modelName].tuning_states[self.phase].assesment["predictions_val"]
                                                      y_true = self.pipelines[category][pipeline].modelling.dataset.y_val
                                                      assert y_pred is not None, f"Predictions are None for model: {modelName}. Phase: {self.phase}, Category: {category}, Pipeline: {pipeline}"
                                                      assert y_true is not None, f"Actual is None for model: {modelName}"
                                                      not_train_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                                                      df_not_train = self._create_report_dataframe(not_train_report, modelName)
                                                      df_not_train = self._add_additional_metrics_to_report(df_not_train, modelName, additional_metrics)
                                                      
                                                      if include_training: # inter-model evaluation (meaning u compare the overftting)
                                                            y_pred_train = self.pipelines[category][pipeline].modelling.list_of_models[modelName].tuning_states[self.phase].assesment["predictions_train"]
                                                            y_true_train = self.pipelines[category][pipeline].modelling.dataset.y_train
                                                            training_report = classification_report(y_true_train, y_pred_train, output_dict=True, zero_division=0)
                                                            df_training_report = self._create_report_dataframe(training_report, modelName, include_training=True)
                                                            df_training_report = self._add_additional_metrics_to_report(df_training_report, modelName, additional_metrics, include_training=True)
                                                            
                                          else:
                                                      y_pred = self.pipelines[category][pipeline].modelling.list_of_models[modelName].tuning_states[self.phase].assesment["predictions_test"]
                                                      y_true = self.pipelines[category][pipeline].modelling.dataset.y_test
                                                      not_train_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                                                      df_not_train = self._create_report_dataframe(not_train_report, modelName)
                                                      df_not_train = self._add_additional_metrics_to_report(df_not_train, modelName, additional_metrics)

                                                      if include_training:
                                                            y_pred_train = self.pipelines[category][pipeline].modelling.list_of_models[modelName].tuning_states[self.phase].assesment["predictions_train"]
                                                            train = self.pipelines[category][pipeline].modelling.dataset.y_train
                                                            val = self.pipelines[category][pipeline].modelling.dataset.y_val
                                                            y_true_train = np.concatenate([train, val])
                                                            training_report = classification_report(y_true_train, y_pred_train, output_dict=True, zero_division=0)
                                                            df_training_report = self._create_report_dataframe(training_report, modelName, include_training=True)
                                                            df_training_report = self._add_additional_metrics_to_report(df_training_report, modelName, additional_metrics, include_training=True)
                                          classification_reports.append(df_not_train)
                                          if include_training:
                                                classification_reports.append(df_training_report)
            self.merged_report_per_phase[self.phase] = pd.concat(classification_reports).T # Get all the reports for the models in all the pipelines together
            
            # This is given the encoded map (the numbers in target variable to the actual class names)
            if self.encoded_map is not None:
                  reverse_map = {str(v): k for k, v in self.encoded_map.items()} #{number:name}
                  index = self.merged_report_per_phase[self.phase].index.tolist()
                  new_index = []
                  for idx in index:
                        if idx in reverse_map:  
                              new_index.append(reverse_map[idx])
                        else:  
                              new_index.append(idx)
                  self.merged_report_per_phase[self.phase].index = new_index
            
            return self.merged_report_per_phase[self.phase]
      
      def plot_cross_model_comparison(self, metrics: list[str] = None, cols: int = 2, save_plots: bool = False, save_path: str = None):
                  """
                  Plots the classification report of a given model
                  """
                  assert self.phase in ["pre", "in", "post"], "Phase must be either pre, in or post"
                  if not metrics:
                        metrics = self.variables["phase_runners"]["dataset_runner"]["metrics_to_evaluate"]["classification"]

                  print(f"PLOTTING CROSS MODEL COMPARISON FOR {self.phase} PHASE")
                  
                  # Compute the classification report DataFrame.
                  class_report_df = self._compute_classification_report()
                  self.results_per_phase[self.phase]["classification_report"] = class_report_df
                  num_metrics = len(metrics)
                  rows = math.ceil(num_metrics / cols)

                  fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 7))
                  axes = axes.flatten()  

                  for i, metric_key in enumerate(metrics):
                        print(f"Plotting: {metric_key}")
                        class_report_cols = class_report_df.columns
                        assert metric_key in class_report_cols, f"Metric not present in {class_report_cols}"
                        ax = axes[i]
                        
                        metric_df = class_report_df[metric_key]

                        df_numeric = metric_df.iloc[:-1].astype(float)
                        model_names = metric_df.loc["modelName"]
                        # Check if df_numeric is a Series or DataFrame
                        if isinstance(df_numeric, pd.Series):
                            isConstantMetric = len(set(df_numeric)) == 1
                        else:
                            isConstantMetric = len(set(df_numeric.iloc[:, 0])) == 1

                        if isinstance(model_names, str): # single model
                              model_names = [model_names]
                              ax.plot(df_numeric.index, df_numeric.iloc[:], marker='o', label=model_names[0])
                        else:
                              model_names = model_names.values
                              if isConstantMetric:
                                    bars = ax.bar(model_names, df_numeric.iloc[0, :])
                                    ax.bar_label(bars, fmt='%.4f')
                              else:
                                    for i, model_name in enumerate(model_names):
                                          ax.plot(df_numeric.index, df_numeric.iloc[:, i], marker='o', label=model_name)
                              
                        ax.set_title(f'{metric_key} by Model')
                        ax.set_xlabel('Class Index')
                        ax.set_ylabel(metric_key)
                        ax.set_ylim(0, 1)
                        ax.tick_params(axis='x', rotation=45)
                        ax.legend()
                        ax.grid(True)
                  
                  eliminate_unused_plots(fig, axes, i)

                  plt.tight_layout()
                  plt.suptitle(f"Cross-model Performance Comparison - {self.phase} phase")
                  plt.tight_layout(rect=[0, 0, 1, 0.96])
                  save_or_store_plot(fig, save_plots, directory_path=save_path + f"/{self.phase}/model_performance", filename=f"cross_model_comparison_{self.phase}.png")
      
      def plot_intra_model_comparison(self, metrics: list[str] = None, save_plots: bool = False, save_path: str = None):
            """
            3 cols each with two trends. As many rows as unique models
            """
            print(f"METRICS IS {metrics}")
            if not metrics:
                  metrics = self.variables["phase_runners"]["dataset_runner"]["metrics_to_evaluate"]["classification"]
            class_report_df = self._compute_classification_report(include_training=True)
            self.results_per_phase[self.phase]["classification_report_train"] = class_report_df
            models = class_report_df.T["modelName"].unique()
            models = {model.split("_")[0] for model in models}
            
            num_metrics = len(metrics)
            cols = num_metrics
            rows = len(models)

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
            if rows == 1:
                  axes = axes.reshape(1, -1)

            colors = ["red", "blue", "green", "purple", "orange", "brown", "pink", "gray", "cyan", "magenta"]
            colors_length = len(colors)
            
            for i, model in enumerate(models):
                color_train = colors[i % colors_length]
                color_no_train = colors[(i + 1) % colors_length]
                for j, metric in enumerate(metrics):
                    class_report_cols = class_report_df.columns
                    assert metric in class_report_cols, f"Metric not present in {class_report_cols}"
                    model_filter = class_report_df.T["modelName"].str.startswith(model)
                    model_df = class_report_df.T[model_filter]

                    ax = axes[i, j]
                    metric_df = model_df.T[metric]

                    df_numeric = metric_df.iloc[:-1].astype(float)
                    model_names = metric_df.loc["modelName"].values
                    isConstantMetric = len(set(df_numeric.iloc[:, 0])) == 1


                    if isConstantMetric:
                        bars = ax.bar(model_names, df_numeric.iloc[0, :])
                        ax.bar_label(bars, fmt='%.4f')
                    else:     
                        ax.plot(df_numeric.index, df_numeric.iloc[:, 0], marker="o", label=model_names[0], color=color_train)
                        ax.plot(df_numeric.index, df_numeric.iloc[:, 1], marker="s", label=model_names[1], color=color_no_train)

                    ax.set_title(f'{metric} - {model}')
                    ax.set_xlabel('Class Index')
                    ax.set_ylabel(metric)
                    ax.set_ylim(0, 1)
                    ax.tick_params(axis='x', rotation=45)
                    if metric != "accuracy":
                        ax.legend()
                    ax.grid(True)


            plt.tight_layout()
            plt.tight_layout(rect=[0, 0, 1, 0.96])  
            plt.suptitle(f"Intra-model Perfomance Comparison - {self.phase} phase")
            plt.show()
            save_or_store_plot(fig, save_plots, directory_path=save_path + f"/{self.phase}/model_performance", filename=f"intra_model_comparison_{self.phase}.png")

      def plot_results_df(self, metrics: list[str], save_plots: bool = False, save_path: str = None):
            """
            Results df is the dataframe with some general performance metrics + time-based metrics (time to fit, time to predict)
            """
            assert self.phase in ["pre", "in", "post"], "Phase must be either pre, in or post"
            dataframes = []
            for category in self.pipelines:
                  for pipeline in self.pipelines[category]:
                        df = self.pipelines[category][pipeline].modelling.results_analysis[self.phase].phase_results_df
                        dataframes.append(df)
            metrics_df = pd.concat(dataframes)
            self.results_per_phase[self.phase]["metrics_df"] = metrics_df

            num_metrics = len(metrics)
            cols = 2
            rows = math.ceil(num_metrics / cols)

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
            axes = axes.flatten()  # Flatten to iterate easily, even if 1 row

            for i, metric in enumerate(metrics):
                  ax = axes[i]
                  sns.barplot(data=metrics_df, x='modelName', y=metric, ax=ax, palette="viridis")
                  ax.set_title(f'{metric} by Model')
                  ax.set_xlabel('Model Name')
                  ax.set_ylabel(metric)
                  ax.tick_params(axis='x', rotation=45)

                  # Annotate values
                  for container in ax.containers:
                        ax.bar_label(container, fmt='%.4f', label_type='edge')

            # Hide any unused subplots
            for j in range(i + 1, len(axes)):
                  fig.delaxes(axes[j])

            plt.tight_layout()
            plt.suptitle(f"Model Performance - {self.phase} phase")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            save_or_store_plot(fig, save_plots, directory_path=save_path + f"/{self.phase}/model_performance", filename=f"time_based_model_performance_{self.phase}.png")
            return metrics_df
      
      def plot_feature_importance(self, save_plots: bool = False, save_path: str = None):
            
            assert self.phase in ["pre", "in", "post"], "Phase must be either pre, in or post"
            importances_dfs = {}
            for pipeline in self.pipelines["not_baseline"]:
                  if pipeline not in ["ensembled", "tree_based"]:
                        continue
                  for modelName in self.pipelines["not_baseline"][pipeline].modelling.list_of_models:
                        if self.phase == "post" and modelName != self.best_performing_model["modelName"]:
                                          continue
                        if modelName not in self.pipelines["not_baseline"][pipeline].modelling.models_to_exclude:
                              importances = self.pipelines["not_baseline"][pipeline].modelling.list_of_models[modelName].tuning_states[self.phase].assesment["model_sklearn"].feature_importances_
                              feature_importance_df = pd.DataFrame({
                                                                            'Feature': self.pipelines["not_baseline"][pipeline].dataset.X_train.columns,
                                                                            'Importance': importances
                                                                            }).sort_values(by='Importance', ascending=False)
                              importances_dfs[pipeline] = feature_importance_df
            for pipeline in importances_dfs:
                  fig, ax = plt.subplots(figsize=(10, 10))
                  sns.barplot(
                        x="Importance",
                        y="Feature",
                        data=importances_dfs[pipeline],
                        ax=ax
                        )
                  ax.set_title(f"Feature Importances for {pipeline} model")
                  plt.tight_layout()
                  plt.tight_layout(rect=[0, 0, 1, 0.96])
                  save_or_store_plot(fig, save_plots, directory_path=save_path + f"/{self.phase}/feature_importance", filename=f"feature_importance_{self.phase}.png")
            return importances_dfs

      def plot_confusion_matrix(self, save_plots: bool = False, save_path: str = None):
            """
            Plots the confusion matrix of a given model
            """
            assert self.phase in ["pre", "in", "post"], "Phase must be either pre, in or post"
            confusion_matrices = {}
            residuals = {}
            for category in self.pipelines:
                  for pipeline in self.pipelines[category]:
                        for modelName in self.pipelines[category][pipeline].modelling.list_of_models:
                              if modelName not in self.pipelines[category][pipeline].modelling.models_to_exclude:
                                    if category == "not_baseline" and self.phase == "post" and modelName != self.best_performing_model["modelName"]:
                                          continue
                                    if self.phase == "in" and category == "baseline":
                                          continue
                                    if self.phase != "post":
                                          pred = self.pipelines[category][pipeline].modelling.list_of_models[modelName].tuning_states[self.phase].assesment["predictions_val"]
                                          actual = self.pipelines[category][pipeline].modelling.dataset.y_val
                                          residuals[pipeline] = self.pipelines[category][pipeline].modelling.dataset.y_val[pred != actual]
                                    else:
                                          pred = self.pipelines[category][pipeline].modelling.list_of_models[modelName].tuning_states[self.phase].assesment["predictions_test"]
                                          actual = self.pipelines[category][pipeline].modelling.dataset.y_test
                                          residuals[pipeline] = self.pipelines[category][pipeline].modelling.dataset.y_test[pred != actual]

                                    assert pred is not None, "Predictions are None"
                                    assert actual is not None, "Actual is None"
                                    assert len(pred) == len(actual), "Predictions and actual must be of the same length"
                                    cm = confusion_matrix(actual, pred)
                                    confusion_matrices[modelName] = {
                                                                    "absolute": cm,
                                                                    "relative": cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                                                                    }
            
            fig, axes = plt.subplots(len(confusion_matrices), 2, figsize=(15, 5* len(confusion_matrices)))
            # Convert axes to 2D array if there's only one model
            if len(confusion_matrices) == 1:
                  axes = np.array([axes])
                  
            # Get category labels if encoded_map exists
            labels = None
            if self.encoded_map is not None:
                  # Sort by encoded value to ensure correct order
                  labels = [label for label in self.encoded_map]
            assert labels is not None, "Labels are None"
            
            for i, (modelName, cm_data) in enumerate(confusion_matrices.items()):
                  print(f"Plotting: {modelName}")
                  # Absolute Confusion Matrix (meaning it does not have the percentage of class predictionsm)
                  sns.heatmap(cm_data["absolute"], 
                        annot=True, 
                        fmt='d',  
                        cmap='Blues',
                        ax=axes[i, 0],
                        xticklabels=labels,
                        yticklabels=labels)
                  axes[i, 0].set_title(f"Absolute Confusion Matrix for model: {modelName}")
                  axes[i, 0].set_xlabel("Predicted")
                  axes[i, 0].set_ylabel("Actual")

                  # Relative Confusion Matrix
                  sns.heatmap(cm_data["relative"], 
                        annot=True, 
                        fmt='.1f',  
                        cmap='Blues',
                        ax=axes[i, 1],
                        xticklabels=labels,
                        yticklabels=labels)
                  axes[i, 1].set_title(f"Relative Confusion Matrix for model: {modelName}")
                  axes[i, 1].set_xlabel("Predicted")
                  axes[i, 1].set_ylabel("Actual")
            
            plt.tight_layout()
            plt.suptitle(f"Confusion Matrix - {self.phase} phase")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            save_or_store_plot(fig, save_plots, directory_path=save_path + f"/{self.phase}/model_performance", filename=f"confusion_matrices_{self.phase}.png")

            return residuals, confusion_matrices
      
      def plot_results_summary(self, training_metric: str, performance_metric: str, save_plots: bool = False, save_path: str = None):
            """
            Scatterplot: x-axis is either timeToFit or timeToPredict and y-axis is a performance metric
            """
            assert training_metric in ["timeToFit", "timeToPredict"], "training_metric must be either timeToFit or timeToPredict"
            assert performance_metric in self.variables["phase_runners"]["dataset_runner"]["metrics_to_evaluate"]["classification"], "performance_metric must be a classification metric"
            
            if self.phase == "pre" or self.phase == "in":
                  performance_metric += "_val"
            else:
                  performance_metric += "_test"
            
            metrics_df = self.results_per_phase[self.phase]["metrics_df"]
            
            fig, ax = plt.subplots(figsize=(15, 8))

            # draw the scatterplot without legend
            sns.scatterplot(
                  data=metrics_df,
                  x=training_metric,
                  y=performance_metric,
                  hue="modelName",
                  legend=False,        
                  s=150,                
                  alpha=0.7,
                  ax=ax            
            )

            for _, row in metrics_df.iterrows():
                  plt.annotate(
                        f"{row['modelName']}\n{row[performance_metric]:.2f}",                   
                        (row[training_metric], row[performance_metric]),  
                        textcoords="offset points",         
                        xytext=(5, 5),                      
                        ha='left',                         
                        va='bottom',                        
                        fontsize=9                          
                  )

            plt.xlabel(f"{training_metric} (log scale)")
            plt.ylabel(performance_metric)
            plt.title(f"Model Performance: {training_metric} vs. {performance_metric}")
            plt.tight_layout()
            plt.ylim(0, 1)
            plt.grid(True)
            plt.xscale("log")
            save_or_store_plot(fig, save_plots, directory_path=save_path + f"/{self.phase}/model_performance", filename=f"results_summary_{self.phase}.png")

      def plot_per_epoch_progress(self, metrics: list[str], save_plots: bool = False, save_path: str = None):
            self.neural_nets_plots = NeuralNetsPlots(self.pipelines["not_baseline"]["feed_forward_neural_network"].modelling.list_of_models["Feed Forward Neural Network"].tuning_states[self.phase].assesment["model_sklearn"])
            self.neural_nets_plots.plot_per_epoch_progress(metrics, phase=self.phase, save_plots=save_plots, save_path=save_path)

                        

             
                        
      
                
            


