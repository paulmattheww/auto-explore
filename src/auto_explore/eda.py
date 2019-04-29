'''
Class definitions for automated EDA.
'''
import pandas as pd
from sklearn.model_selection import train_test_split

from .diligence import get_df_columns_dtypes
from .diligence import get_numeric_columns
from .diligence import get_str_or_object_columns

class AutopilotExploratoryAnalysis:
    def __init__(self, df, bin_cols, cat_cols, num_cols, text_cols,
                target_col=None, hue=None, na_tolerance=.10, time_dim=None,
                dask=True):
        '''It is best to pass in the df in a format that is ready for analysis.
        Ideally, missing values have already been filled in to extract a
        reasonable first-pass analysis.

        ARGS:
            bin_cols <list>: Binary columns.  Vectors must adhere to
                arrays of int, float or bool.  Transformed to int.
            cat_cols <list>: Categorical columns.
            num_cols <list>: Numeric columns.
            text_cols <list>: Text columns.
        KWARGS:
            target_col=None, hue=None, na_tolerance=.10, time_dim=None, dask=True
        '''
        self.df = df
        self.bin_cols = bin_cols
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.text_cols = text_cols
        for k, v in kwargs.items():
            setattr(self, f'{k}', v)

    @property
    def split_data(self):
        pass

    def characterize_missing_values(self):
        '''Returns a pd.Series with the column name as the key and the percent
        of the column that is missing as the value.
        '''
        return self.df.isna().sum() / self.df.shape[0]

    def fill_missing_values(self, groupby_cols=None):
        '''Several methods to impute missing data.  Depends on characterization
        of missing values.
        '''
        pass

    @property
    def profile_report(self):
        '''Runs pandas_profiler.ProfileReport on self.df
        '''
        return ProfileReport(self.df)

    def identify_reject_columns(self):
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
