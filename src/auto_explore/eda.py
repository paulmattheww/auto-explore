'''
Class definitions for automated EDA.
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .featexp import get_trend_stats, get_univariate_plots
from .stats import best_theoretical_distribution
from .diligence import get_df_columns_dtypes
from .diligence import get_numeric_columns
from .diligence import get_str_or_object_columns
from .viz import *

class AutopilotExploratoryAnalysis:
    def __init__(self, df, bin_cols, cat_cols, num_cols,
                text_cols=None, target_col=None, time_dim=None, dask=False):
        '''It is best to pass in the df in a format that is ready for analysis.
        Ideally, missing values have already been filled in or dropped in order
        to extract a reasonable first-pass analysis, though this object type
        can help you with the process of identifying where attention should be
        applied when preprocessing your data.

        ARGS:
            bin_cols <list>: Binary columns.  Vectors must adhere to
                arrays of int, float or bool.  Transformed to int.
            cat_cols <list>: Categorical columns.
            num_cols <list>: Numeric columns.
            text_cols <list>: Text columns.
        KWARGS:
            target_col=None, hue=None, na_tolerance=.10, time_dim=None, dask=True
        '''
        # set attributes
        self.df = df
        self.bin_cols = bin_cols
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.text_cols = text_cols
        self.target_col = target_col
        self.time_dim = time_dim
        self.dask = dask

        # Set derivative attributes
        self._X = df[bin_cols + cat_cols + num_cols] # raw X
        self._y = df[target_col]                     # raw y
        self.X = None                                # processed X
        self.y = None                                # processed y

    @property
    def split_data(self):
        '''Splits self.df into a dict of evenly split data for use in data
        exploration tasks, such as univariate plots.
        '''
        splt_kwargs = dict(train_size=.5, random_state=777)
        return train_test_split(self.df, **splt_kwargs)

    def characterize_missing_values(self):
        '''Returns a pd.Series with the column name as the key and the percent
        of the column that is missing as the value.
        '''
        return self.df.isna().sum() / self.df.shape[0]

    @property
    def has_missing_values(self):
        '''Quick check for missing values for input df.'''
        if self.characterize_missing_values().sum() > 0:
            return True
        return False

    def fill_missing_values(self, interpolation_func='median'):
        '''Several methods to impute missing data.  Depends on characterization
        of missing values.

        KWARGS:
            groupby_cols <list> or None: Use if it is desired to perform some
                sort of interpolation based on group identity.
            interpolation_func <object>: Function object that will accept a
                pd.DataFrame.groupby object and use heuristics to interpolate
        RETURNS:
        '''
        num_cols, cat_cols, bin_cols = self.num_cols, self.cat_cols, self.bin_cols
        _df = self.df.copy()
        if self.characterize_missing_values().sum() > 0:
            if interpolation_func == 'median':
                _df[num_cols] = _df[num_cols].fillna(_df[num_cols].median())
            else:
                _df[num_cols] = _df[num_cols].fillna(_df[num_cols].mean())
            _df[cat_cols] = _df[cat_cols].fillna('MISSING')
            _df[bin_cols] = _df[bin_cols].fillna(0)
            setattr(self, 'df', _df)
            print("Set attribute `df` with non-missing data version of original")

    @property
    def profile_report(self):
        '''Runs pandas_profiler.ProfileReport on self.df to return an HTML
        rendering of the data's summary.  Also a property.

        RETURNS:
            pandas_profiler.ProfileReport of self.df
        '''
        return ProfileReport(self.df)

    def identify_reject_columns(self, threshold=.90):
        '''Leverages pandas_profiler.ProfileReport object of self to use the
        method get_rejected_variables.  Returns all variables that are stable
        insofar as they are not highly correlated with other variables.

        KWARGS:
        RETURNS:
        '''
        return self.profile_report.get_rejected_variables(threshold=threshold)

    def characterize_distributions(self):
        '''Determine what each numeric column's distribution is and return
        recommended a DataFrame of the best distributions.

        RETURNS:
        '''
        dist_dict = dict()
        for num_col in self.num_cols:
            dist_dict[num_col] = best_theoretical_distribution(self.df[num_col])
        return pd.DataFrame(dist_dict)

    def scale_numeric_columns(self, Scaler=StandardScaler, scaler_kwargs={}):
        '''User passes a Scaler function to transform their dataset into a
        machine learning friendly format.

        KWARGS:
        RETURNS:
        '''
        scaler = Scaler().fit(self.df[self.num_cols])
        return (scaler.transform(self.df[self.num_cols]), scaler)

    def target_distribution_over_binary_groups(self, override_bin_cols=None, **kwargs):
        '''

        ARGS:
        KWARGS:
        RETURNS:
        '''
        if not override_bin_cols:
            args = (self.df, self.bin_cols, self.target_col)
            target_distribution_over_binary_groups(*args, **kwargs)
        else:
            args = (self.df, override_bin_cols, self.target_col)
            target_distribution_over_binary_groups(*args, **kwargs)

    def convert_categoricals(self):
        '''

        ARGS:
        KWARGS:
        RETURNS:
        '''
        pass

    def generate_correlation_heatmap(self, X=None):
        '''

        ARGS:
        KWARGS:
        RETURNS:
        '''
        if not X:
            correlation_heatmap(self.df[self.num_cols + self.bin_cols])
        else:
            correlation_heatmap(X)

    def generate_univarite_plots(self, features_list=None):
        '''Leverages featexp (with modifications made inside of this repo) to
        plot univariate plots for both a train and test dataset for comparison

        KWARGS:
        RETURNS:
        '''
        if self.target_col is None:
            raise ValueError("No target_col specified.")
        if features_list is None:
            features_list = self.num_cols
        kwargs = dict(data=self.split_data[0],
                      target_col=self.target_col,
                      data_test=self.split_data[1],
                      features_list=features_list)
        return get_univariate_plots(**kwargs)

    def derive_features_from_data(self, feature_derivation_func, **feature_kwargs):
        '''

        ARGS:
        KWARGS:
        RETURNS:
        '''
        X = feature_derivation_func(self._X, **feature_kwargs)
        setattr(self, 'X', X)
        print(f'Attribute X set on {self}')

    def derive_top_text_features(self):
        '''

        ARGS:
        KWARGS:
        RETURNS:
        '''
        pass

    def extract_text_features(self, vectorizer_func=CountVectorizer, **vectorizer_kwargs):
        '''

        ARGS:
        KWARGS:
        RETURNS:
        '''
        pass

    def cluster_and_plot(self, cluster_cols=None, cluster_func=None):
        '''

        ARGS:
        KWARGS:
        RETURNS:
        '''
        pass

    def derive_optimal_clusters(self):
        '''

        ARGS:
        KWARGS:
        RETURNS:
        '''
        pass

    def pairplot_matrix(self, numeric_cols=None):
        '''Plots seaborn's version of a PairGrid using a 2-D KDE plot
        on the lower half, a 1-D KDE plot on the diagonal, and a scatterplot
        on the upper half.

        You may want to sample your data if you have a lot of data, especially
        if you have many dimensions.  Works best on smaller datasets.

        ARGS:
        KWARGS:
        RETURNS:
        '''
        if not numeric_cols:
            numeric_cols = self.num_cols + self.bin_cols
        scatterplot_matrix_kde(self.df[numeric_cols])

    def full_suite_report(self):
        '''

        ARGS:
        KWARGS:
        RETURNS:
        '''
        pass

    def rf_feature_importances(self, X=None):
        '''

        ARGS:
        KWARGS:
        RETURNS:
        '''
        if X:
            rf_feature_importances(X, wine_df.target)
        else:
            rf_feature_importances(self.X, wine_df.target)



# END
