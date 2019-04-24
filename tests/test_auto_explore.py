
from unittest import TestCase

from sklearn.datasets import load_wine

from auto_explore.eda import AutopilotExploratoryAnalysis

def get_wine_data():
    wine = load_wine()
    wine_df = pd.DataFrame(wine['data'], columns=wine['feature_names'])
    wine_df['target'] = wine['target']
    return

class TestAutopilotEDA(TestCase):
    def __init__(self):
        ax = AutopilotExploratoryAnalysis()
