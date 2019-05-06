
from unittest import TestCase

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.datasets import load_iris

from auto_explore.eda import AutopilotExploratoryAnalysis
from auto_explore.viz import *
from auto_explore.apis import fetch_fred_data
from auto_explore.datetime import make_calendars
from auto_explore.stats import best_theoretical_distribution


# Iris data gathering
def fetch_iris():
    data = load_iris()
    iris_df = pd.DataFrame(data['data'], columns=data['feature_names'])
    iris_df['target'] = data['target']
    return iris_df

iris_df = fetch_iris()


# Wine data gathering
def get_wine_data():
    '''Fetches wine data from sklearn.datasets'''
    wine = load_wine()
    wine_df = pd.DataFrame(wine['data'], columns=wine['feature_names'])
    wine_df['target'] = wine['target']
    return wine_df

wine_df = get_wine_data()


# Economic data gathering
def fetch_econ_data():
    series_list = ['SP500', 'NASDAQCOM', 'DJIA', 'RU2000PR']
    econ_df = fetch_fred_data(series_list)
    year_list = np.arange(2011, 2020)
    cal_df = make_calendars(year_list, drop_index=False)
    cal_df.index.name = 'DATE'
    df = cal_df.join(econ_df).dropna()
    df['date'] = df.index.values
    return df

econ_df = fetch_econ_data()

cat_cols = ['month', 'year', 'weekday']
bin_cols = ['is_weekday', 'is_holiday', 'is_holiday_week']
num_cols = ['NASDAQCOM', 'DJIA', 'RU2000PR']
text_cols = []
target_col = ['SP500']

args = (econ_df, bin_cols, cat_cols, num_cols, text_cols)
kwargs = dict(target_col=target_col)
econ_ax = AutopilotExploratoryAnalysis(*args, **kwargs)



class TestAutopilotEDA(TestCase):

    def test_split_data(self):
        self.assertEqual(len(econ_ax.split_data), 2)

    def test_characterize_missing_values(self):
        self.assertEqual(econ_ax.characterize_missing_values().sum(), 0.)

    def test_fill_missing_values(self):
        econ_ax.df.loc[econ_ax.df.date == '2011-01-03', 'DJIA'] = np.nan
        econ_ax.fill_missing_values()
        self.assertTrue(econ_ax.has_missing_values == False)

    def test_characterize_distributions(self):
        self.assertEqual(econ_ax.characterize_distributions().shape, (7, 3))

    def test_scale_numeric_columns(self):
        self.assertTrue(isinstance(econ_ax.scale_numeric_columns(), tuple))

    def test_cluster_and_plot(self):
        cluster_and_plot_pca(wine_df, show=False)

    def test_rf_feature_importances(self):
        rf_feature_importances(wine_df.drop(columns='target'), wine_df.target, show=False)

    def test_derive_optimal_clusters(self):
        num_cols = ['alcohol', 'malic_acid', 'ash',
                    'alcalinity_of_ash', 'magnesium',
                    'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                    'proanthocyanins', 'color_intensity', 'hue',
                    'od280/od315_of_diluted_wines', 'proline']
        derive_optimal_clusters(wine_df, num_cols=num_cols, show=False)
