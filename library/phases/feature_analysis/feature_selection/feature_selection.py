
from library.phases.dataset.dataset import Dataset
from library.phases.feature_analysis.feature_selection.automatic import AutomaticFeatureSelection
from library.phases.feature_analysis.feature_selection.manual import ManualFeatureSelection

class FeatureSelection:
      """
      """
      def __init__(self, dataset: Dataset) -> None:
            self.dataset = dataset
            self.automatic_feature_selection = AutomaticFeatureSelection(self.dataset)
            self.manual_feature_selection = ManualFeatureSelection(self.dataset)
      


    
        