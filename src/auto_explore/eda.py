'''
Extends work from:
 - https://github.com/pandas-profiling/pandas-profiling
 - https://github.com/abhayspawar/featexp
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport

class AutopilotExploratoryAnalysis:
    '''
    Questions that need answering:
        - How the data should exist when feeding it in
        - Output to console or notebook vs. HTML
    '''
    def __init__(self, df, bin_cols, cat_cols, num_cols, text_cols,
                target_col=None, hue=None, time_dim=None,
                dask=True, dask_kwargs=None):
        '''
        ARGS:
            bin_cols <list>: Binary columns.  Vectors must adhere to
                arrays of int, float or bool.  Transformed to int.
            cat_cols <list>: Categorical columns.
            num_cols <list>: Numerical (float or int) columns.
            text_cols <list>: String columns of free text.
        KWARGS:
            target_col <str>: Target column that will be modeled.
            hue <str>: Optional -- for visualization.  Group to viz over.
            dask <bool>: Whether to use Dask or not (use with larget datasets)
        '''
        self.df = df
        self.bin_cols = bin_cols
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.text_cols = text_cols
        for kwarg, value in kwargs.items():
            setattr(self, f'{kwarg}', value)

    @property
    def split_data(self):
        pass

    @staticmethod
    def modify_notebook_configuration(self):
        pass

    @staticmethod
    def characterize_missing_values(self):
        pass

    def fill_missing_values(self, groupby_cols=None):
        '''Several methods to impute missing data

        Depends on characterization of nulls.
        '''
        pass

    def high_level_profile(self):
        '''Run Pandas_Profiler'''
        pass

    def identify_reject_columns(self):
        '''Relies on self.high_level_profile method'''
        pass

    def characterize_distributions(self):
        '''Determine what each numeric column's distribution is
        and return recommended scaling techniques'''
        pass

    def scale_numeric_columns(self):
        pass

    def viz_target_over_groups(self):
        pass

    def convert_categoricals(self):
        pass

    def generate_correlation_heatmap(self):
        pass

    def generate_univarite_plots(self):
        pass

    def derive_features_from_data(self, feature_derivation_func):
        pass

    def derive_top_text_features(self):
        pass

    def extract_text_features(self, vectorizer_func=CountVectorizer, **vectorizer_kwargs):
        pass

    def cluster_and_plot(self, cluster_cols=None, cluster_func=None):
        pass

    def derive_optimal_clusters(self):
        pass

    def full_suite_report(self):
        pass

    def rf_feature_importance(self):
        pass
