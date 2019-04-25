
from unittest import TestCase

import pandas as pd
from sklearn.datasets import load_wine

from auto_explore.eda import AutopilotExploratoryAnalysis

def get_wine_data():
    '''Fetches wine data from sklearn.datasets'''
    wine = load_wine()
    wine_df = pd.DataFrame(wine['data'], columns=wine['feature_names'])
    wine_df['target'] = wine['target']
    return

class TestAutopilotEDA(TestCase):
    def __init__(self):
        self.ax = AutopilotExploratoryAnalysis()

    def test_split_data(self):
        pass

    def test_characterize_missing_values(self):
        pass

    def test_fill_missing_values(self):
        pass

    def test_high_level_profile(self):
        pass

    def test_identify_reject_columns(self):
        pass

    def test_characterize_distributions(self):
        pass

    def test_scale_numeric_columns(self):
        pass
