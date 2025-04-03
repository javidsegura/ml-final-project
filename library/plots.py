import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from library.dataset import Dataset




class Plots:
  """ 
  We will be using 'composition' desing pattern to create plots from the dataframe object that is an instance of the Dataset class
  This design pattern allows for two classes to be able to share data (e.g: dataset object)
  """
  def __init__(self, dataset: Dataset) -> None:
    self.dataset = dataset

  def plot_correlation_matrix(self, size: str = "small"):
    """
    Plots the correlation matrix of the dataframe

    Parameters
    ----------
      size : str
        The size of the plot. Taken on ["s", "m", "l", "auto"]
    """
    only_numerical_df = self.dataset.df.select_dtypes(include=["number"])
    corr = only_numerical_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool)) # avoid redundancy
    if size == "s":
      f, ax = plt.subplots(figsize=(5, 3))
    elif size == "m":
      f, ax = plt.subplots(figsize=(10, 6))
    elif size == "l":
      f, ax = plt.subplots(figsize=(20, 15))
    elif size == "auto":
      f, ax = plt.subplots()
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    vmin, vmax = corr.min().min(), corr.max().max()
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
              square=True, linewidths=.5, cbar_kws={"shrink": .8}, vmin=vmin, vmax=vmax)
  def say_hello(self):
    print("Hello, world!")

  def get_set_distributions(self, figsize=(15, 20)):
    """
    Plots distributions of all features across train, validation and test sets.
    
    Parameters
    ----------
    figsize : tuple, optional
        Figure size (width, height)
    """
    # Get all features
    features = self.dataset.X_train.columns
    n_features = len(features)
    
    # Calculate rows needed (features + 1 for target variable)
    n_rows = (n_features // 3) + (1 if n_features % 3 > 0 else 0) + 1
    
    # Create figure
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Plot feature distributions
    for i, feature in enumerate(features):
        ax = axes[i]
        
        # Check if feature is numerical or categorical
        if np.issubdtype(self.dataset.X_train[feature].dtype, np.number):
            # Numerical feature
            sns.histplot(self.dataset.X_train[feature], kde=True, label='Train', ax=ax, alpha=0.5)
            
            if hasattr(self.dataset, 'X_val'):
                sns.histplot(self.dataset.X_val[feature], kde=True, label='Validation', ax=ax, alpha=0.5)
                
            if hasattr(self.dataset, 'X_test'):
                sns.histplot(self.dataset.X_test[feature], kde=True, label='Test', ax=ax, alpha=0.5)
        else:
            # Categorical feature
            train_counts = self.dataset.X_train[feature].value_counts(normalize=True)
            
            if hasattr(self.dataset, 'X_val'):
                val_counts = self.dataset.X_val[feature].value_counts(normalize=True)
            
            if hasattr(self.dataset, 'X_test'):
                test_counts = self.dataset.X_test[feature].value_counts(normalize=True)
            
            # Plot categorical counts
            train_counts.plot(kind='bar', label='Train', ax=ax, alpha=0.5, position=0, width=0.3)
            
            if hasattr(self.dataset, 'X_val'):
                val_counts.plot(kind='bar', label='Validation', ax=ax, alpha=0.5, position=1, width=0.3)
                
            if hasattr(self.dataset, 'X_test'):
                test_counts.plot(kind='bar', label='Test', ax=ax, alpha=0.5, position=2, width=0.3)
        
        ax.set_title(f'{feature}')
        ax.legend()
    
    # Plot target variable distributions
    if n_features < len(axes) - 1:  # Make sure we have space for target plot
        # Get last row for target plots
        for i in range(3):
            ax = axes[n_features + i]
            
            if i == 0 and hasattr(self.dataset, 'y_train'):
                if np.issubdtype(self.dataset.y_train.dtype, np.number):
                    sns.histplot(self.dataset.y_train, kde=True, ax=ax)
                else:
                    self.dataset.y_train.value_counts(normalize=True).plot(kind='bar', ax=ax)
                ax.set_title('Target - Train Set')
            
            elif i == 1 and hasattr(self.dataset, 'y_val'):
                if np.issubdtype(self.dataset.y_val.dtype, np.number):
                    sns.histplot(self.dataset.y_val, kde=True, ax=ax)
                else:
                    self.dataset.y_val.value_counts(normalize=True).plot(kind='bar', ax=ax)
                ax.set_title('Target - Validation Set')
            
            elif i == 2 and hasattr(self.dataset, 'y_test'):
                if np.issubdtype(self.dataset.y_test.dtype, np.number):
                    sns.histplot(self.dataset.y_test, kde=True, ax=ax)
                else:
                    self.dataset.y_test.value_counts(normalize=True).plot(kind='bar', ax=ax)
                ax.set_title('Target - Test Set')
    
    # Remove empty subplots if any
    for i in range(n_features + 3, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()