import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from library.dataset import Dataset

class Preprocessing:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def analyze_duplicates(self, drop: bool = False):
        """
        Analiza y elimina duplicados en el dataframe.
    
        Parameters
        ----------
        drop : bool, default=False
            Si es True, elimina los duplicados del dataframe.
    
        Returns
        -------
        int
            Número de filas duplicadas encontradas.
        """
        num_duplicates = self.df.duplicated().sum()
        print(f"Duplicados encontrados: {num_duplicates}")
    
        if num_duplicates > 0:
            print("Ejemplo de duplicados:")
            print(self.df[self.df.duplicated()].head())
    
            if drop:
                self.df = self.df.drop_duplicates().reset_index(drop=True)
                print("Duplicados eliminados.")
    
        return num_duplicates

    def analyze_missing_data(self, drop_threshold: float = 0.5, fill_method: str = None):
        """
        Analiza y trata los valores faltantes en el dataframe.
    
        Parameters
        ----------
        drop_threshold : float, default=0.5
            Si el porcentaje de valores nulos en una columna es mayor que este umbral, se elimina la columna.
    
        fill_method : str, default=None
            Método para imputar valores nulos. Opciones: 'mean', 'median', 'mode'. 
            Si es None, no imputa nada.
    
        Returns
        -------
        pd.DataFrame
            DataFrame con el análisis de valores faltantes.
        """
        missing_counts = self.df.isnull().sum()
        missing_percent = (missing_counts / len(self.df)) * 100
        missing_data = pd.DataFrame({'Missing Count': missing_counts, 'Missing %': missing_percent})
        
        print("Valores faltantes por columna:")
        print(missing_data[missing_data['Missing Count'] > 0])
    
        # Eliminar columnas con muchos nulos
        cols_to_drop = missing_data[missing_data['Missing %'] > drop_threshold * 100].index.tolist()
        if cols_to_drop:
            print(f"Eliminando columnas con más del {drop_threshold*100}% de nulos: {cols_to_drop}")
            self.df.drop(columns=cols_to_drop, inplace=True)
    
        # Imputar valores si se especifica un método
        if fill_method:
            if fill_method == 'mean':
                self.df.fillna(self.df.mean(), inplace=True)
            elif fill_method == 'median':
                self.df.fillna(self.df.median(), inplace=True)
            elif fill_method == 'mode':
                self.df.fillna(self.df.mode().iloc[0], inplace=True)
            else:
                print("Método no reconocido. Usa 'mean', 'median' o 'mode'.")
        
        return missing_data

    def analyze_class_balance(self, target_column: str, apply_smote: bool = False):
        """
        Analiza el balance de clases y aplica SMOTE si es necesario.
    
        Parameters
        ----------
        target_column : str
            Nombre de la columna objetivo.
    
        apply_smote : bool, default=False
            Si es True, aplica SMOTE para balancear las clases.
    
        Returns
        -------
        pd.Series
            Distribución de clases antes del balanceo.
        """
        class_counts = self.df[target_column].value_counts()
        print("Distribución de clases:")
        print(class_counts)
    
        # Verificar si el dataset está desbalanceado
        min_class = class_counts.min()
        max_class = class_counts.max()
        imbalance_ratio = max_class / min_class
        print(f"Ratio de desbalanceo: {imbalance_ratio:.2f}")
    
        if imbalance_ratio > 2:  # Umbral arbitrario, cambiar si es necesario
            print("⚠️ Se detecta un desbalanceo de clases.")
        else:
            print("✅ No hay un desbalanceo significativo.")
    
        # Aplicar SMOTE si es necesario
        if apply_smote:
            print("Aplicando SMOTE para balancear las clases...")
            X, y = self.__get_X_y__(target_column)
            categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
            smote_nc = SMOTENC(categorical_features=[X.columns.get_loc(col) for col in categorical_features], random_state=RANDOM_STATE)
            X_resampled, y_resampled = smote_nc.fit_resample(X, y)
            
            # Crear nuevo dataframe balanceado
            self.df = pd.DataFrame(X_resampled, columns=X.columns)
            self.df[target_column] = y_resampled
            print("✔️ SMOTE aplicado con éxito.")
    
        return class_counts
    
    def remove_reboot_column(self) -> None:
        self.X_train_without_reboot = self.dataset.X_train.drop(columns="Reboot")
    
    # Function to determine if a feature needs standardization or normalization
    def determine_scaling_method(self, feature_series) -> str:
        """
        Determine whether a feature should be standardized or normalized based on its characteristics.
        
        Args:
            feature_series: Pandas Series containing the feature values
            
        Returns:
            str: 'robust', 'normalize', or 'none'
        """
        # self.remove_reboot_column()
        
        # Skip if all values are the same (zero variance)
        if feature_series.std() == 0:
            return 'none'
        
        # Check for outliers using IQR method
        Q1 = feature_series.quantile(0.25)
        Q3 = feature_series.quantile(0.75)
        IQR = Q3 - Q1
        outlier_threshold = 1.5 * IQR
        has_outliers = ((feature_series < (Q1 - outlier_threshold)) | 
                        (feature_series > (Q3 + outlier_threshold))).any()
        
        # Check for normality using skewness
        skewness = abs(stats.skew(feature_series.dropna()))
        is_skewed = skewness > 1.0  # Threshold for significant skewness
        
        # Decision logic
        if has_outliers:
            return 'robust'  # Standardization handles outliers better
        elif is_skewed:
            return 'normalize'    # Normalization is better for non-normal distributions
        else:
            return 'robust'  # Default to standardization for most ML algorithms
        
    # Visualize the effect of scaling on a few features
    def plot_before_after_scaling(self, original_df, scaled_df, features, n_features=4) -> None:
        """
        Plot histograms before and after scaling for selected features.
        
        Args:
            original_df: DataFrame containing the original data
            scaled_df: DataFrame containing the scaled data
            features: List of feature names to plot
            
        Returns:
            None
        """
        # Select a subset of features if there are too many
        if len(features) > n_features:
            features = np.random.choice(features, n_features, replace=False)
        e
        fig, axes = plt.subplots(len(features), 2, figsize=(12, 3*len(features)))
        
        for i, feature in enumerate(features):
            # Original distribution
            sns.histplot(original_df[feature], kde=True, ax=axes[i, 0])
            axes[i, 0].set_title(f'Original: {feature}')
            
            # Scaled distribution
            sns.histplot(scaled_df[feature], kde=True, ax=axes[i, 1])
            axes[i, 1].set_title(f'Scaled: {feature}')
        
        plt.tight_layout()
        plt.show()
        
    def prepare_scaling(self) -> None:
        """
        Prepare the data for scaling by determining the scaling method for each feature.
        
        Returns:
            None
        """
        # Get numerical columns (excluding Reboot_before)
        numerical_cols = [col for col in self.X_train_without_reboot.columns]
        
        # Determine scaling method for each feature
        scaling_methods = {}
        for col in numerical_cols:
            scaling_methods[col] = self.determine_scaling_method(self.X_train_without_reboot[col])
            
        # Create lists for each scaling method
        self.robust_cols = [col for col, method in scaling_methods.items() if method == 'robust']
        self.normalize_cols = [col for col, method in scaling_methods.items() if method == 'normalize']
        self.no_scaling_cols = [col for col, method in scaling_methods.items() if method == 'none']
        
    def apply_scaling(self):
        """
        Apply scaling to the training data based on the determined scaling methods.
        
        Returns:
            DataFrame: The scaled training data
        """
        self.prepare_scaling()
        
        scaler_robust = RobustScaler()
        scaler_minmax = MinMaxScaler()
        
        # Create a copy of the training data to avoid warnings
        self.X_train_scaled = self.dataset.X_train.copy()
        
        # Apply standardization
        if self.robust_cols:
            self.X_train_scaled[self.robust_cols] = scaler_robust.fit_transform(self.X_train_without_reboot[self.robust_cols])

        # Apply normalization
        if self.normalize_cols:
            self.X_train_scaled[self.normalize_cols] = scaler_minmax.fit_transform(self.X_train_without_reboot[self.normalize_cols])

        # Visualize scaling effects using utility function
        if self.robust_cols:
            print("\nExamples of standardized features:")
            self.plot_before_after_scaling(self.X_train_without_reboot, self.X_train_scaled, self.robust_cols)

        if self.normalize_cols:
            print("\nExamples of normalized features:")
            self.plot_before_after_scaling(self.X_train_without_reboot, self.X_train_scaled, self.normalize_cols)
            
        return self.X_train_scaled
