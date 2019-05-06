'''
Class definitions for automated EDA.

TODO:
- Lasso & Ridge feature selection
- VarianceThreshold drop of features
- SelectKBest feature selection
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from pandas_profiling import ProfileReport

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
            interpolation_func <str>: One of 'mean', 'median', 'bfill', 'ffill'
        RETURNS:
            None, sets df attr
        '''
        num_cols, cat_cols, bin_cols = self.num_cols, self.cat_cols, self.bin_cols
        _df = self.df.copy()
        if self.characterize_missing_values().sum() > 0:
            if interpolation_func == 'median':
                _df[num_cols] = _df[num_cols].fillna(_df[num_cols].median())
            elif interpolation_func == 'ffill':
                _df[num_cols] = _df[num_cols].fillna(method='ffill')
            elif interpolation_func == 'bfill':
                _df[num_cols] = _df[num_cols].fillna(method='bfill')
            else:
                _df[num_cols] = _df[num_cols].fillna(_df[num_cols].mean())
            _df[cat_cols] = _df[cat_cols].fillna('MISSING')
            _df[bin_cols] = _df[bin_cols].fillna(0)
            setattr(self, 'df', _df)
            print("Set attribute `df` with non-missing data version of original")

    def characterize_distributions(self):
        '''Determine what each numeric column's distribution is and return
        recommended a DataFrame of the best distributions.

        RETURNS:
            <pd.DataFrame>: which theoretical distribution fits best the data
                by taking by looking at the output statistics
        '''
        dist_dict = dict()
        for num_col in self.num_cols:
            dist_dict[num_col] = best_theoretical_distribution(self.df[num_col])
        return pd.DataFrame(dist_dict)

    def scale_numeric_columns(self, Scaler=StandardScaler,
                             scaler_kwargs=dict(random_state=777)):
        '''User passes a Scaler function to transform their dataset into a
        machine learning friendly format.

        KWARGS:
            Scaler <sklearn.preprocessing>: Transformer for scaling data
            scaler_kwargs <dict>: kwargs to Transformer passed
        RETURNS:
            scaled data for num_cols only
        '''
        scaler = Scaler().fit(self.df[self.num_cols])
        return (scaler.transform(self.df[self.num_cols]), scaler)

    def target_distribution_over_binary_groups(self, override_bin_cols=None, **kwargs):
        '''Characterizes the distribution of a numeric target vector on the
        y-axis as either a box, violin, or boxen plot.  Takes a list of
        categorical columns and displays one plot per unique value in that
        list, putting the unique values of the category on the x-axis.

        KWARGS:
            override_bin_cols <list>: Specific list of binary columns to
                override the original specified
            kwargs <dict>: kwargs to target_distribution_over_binary_groups()
        RETURNS:
        '''
        if not override_bin_cols:
            args = (self.df, self.bin_cols, self.target_col)
            target_distribution_over_binary_groups(*args, **kwargs)
        else:
            args = (self.df, override_bin_cols, self.target_col)
            target_distribution_over_binary_groups(*args, **kwargs)

    def convert_categoricals(self):
        '''Converts all cat_cols in self.df to one-hot variables, preserving
        one column for each value.  Drops Nulls.
        '''
        kwargs = dict(columns=self.cat_cols, drop_first=False)
        _df = pd.get_dummies(self.df, **kwargs)
        setattr(self, 'df', _df)
        print("Setting attribute `df` to new frame with numerical categories.")

    def generate_correlation_heatmap(self, heatmap_cols=None):
        '''Generates a correlation heatmap.

        ARGS:
            heatmap_cols <list>: Numeric columns to perform correlations
        RETURNS:
            None, plots to console
        '''
        if not heatmap_cols:
            correlation_heatmap(self.df[self.num_cols + self.bin_cols])
        else:
            correlation_heatmap(self.df[heatmap_cols])

    def generate_univarite_plots(self, features_list=None):
        '''Leverages featexp (with modifications made inside of this repo) to
        plot univariate plots for both a train and test dataset for comparison

        KWARGS:
            features_list <list>: Which numeric features to use in univariate
                plotting
        RETURNS:
            None, outputs to console univariate plots and statistics
        '''
        if self.target_col is None:
            raise ValueError("No target_col specified.")
        if features_list is None:
            features_list = self.num_cols + self.bin_cols
        kwargs = dict(data=self.split_data[0],
                      target_col=self.target_col[0],
                      data_test=self.split_data[1],
                      features_list=features_list)
        get_univariate_plots(**kwargs)

    def get_univariate_trend_stats(self, **kwargs):
        '''Calculates trend changes and correlation between train/test for
        each feature in features_list (if supplied in kwargs).

        KWARGS:
            kwargs <dict>: See get_trend_stats in featexp.py
        RETURNS:
            stats <pd.DataFrame>: Reverse sorted by trend correlation
        '''
        if self.target_col is None:
            raise ValueError("No target_col specified.")
        stats = get_trend_stats(data=self.split_data[0],
                                target_col=self.target_col[0],
                                data_test=self.split_data[1],
                                **kwargs)
        return stats

    def derive_features_from_data(self, feature_derivation_func, **feature_kwargs):
        '''User-specified feature_derivation_func to transform the dataset
        into a machine learning format.

        ARGS:
            feature_derivation_func <object>: User-defined
        KWARGS:
            feature_kwargs <dict>: kwargs for feature_derivation_func
        RETURNS:
            Sets X on self
        '''
        X = feature_derivation_func(self._X, **feature_kwargs)
        setattr(self, 'X', X)
        print(f'Attribute X set on {self}')

    def cluster_and_plot(self, cluster_cols=None, **cluster_kwargs):
        '''Performs KMeans (or another specified clustering method that
        accepts n_clusters as an arg) on either the binary and numeric columns
        in the original dataset, or the cluster_cols specified as a list.

        ARGS:
            cluster_cols <list>: Columns to cluster on self.df
        KWARGS:
            cluster_kwargs <dict>: num_cols, Scaler, ClusterAlgorithm
        RETURNS:
            None, plots to the console
        '''
        if cluster_cols:
            cluster_and_plot_pca(self.df[cluster_cols], **cluster_kwargs)
        else:
            cluster_and_plot_pca(self.df[self.num_cols + self.bin_cols])

    def derive_optimal_clusters(self, **kwargs):
        '''Without changing kwargs this function scales the data with a
        StandardScaler for numeric columns, then performs KMeans clustering
        over a range of clusters.

        KWARGS:
            kwargs <dict>: dict(cluster_range=np.arange(2, 11),
                                        num_cols=None, Scaler=StandardScaler,
                                        ClusterAlgorithm=KMeans)
        RETURNS:
            None, plots to console
        '''
        derive_optimal_clusters(self.df, num_cols=self.num_cols, **kwargs)

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

    def lm_group_plot(self, x, y, grp, **kwargs):
        '''Plots y over x and performs a linear regression for each unique
        value of the grp column.

        ARGS:
            df <pd.DataFrame>: Data
            x <str>: X-axis column name
            y <str>: y-axis column name
            grp <str>: Factor over which to fit
        KWARGS:
            size <int>: see mpl
            aspect <float>: see mpl
            title <str>: Title of the plot
        RETURNS:
            None, plots to the console.
        '''
        lm_group_plot(self.df, x, y, grp, **kwargs)

    def rf_feature_importances(self, features_list=None, **kwargs):
        '''Derives feature importances from an sklearn.ensemble.RandomForestClassifier
        or sklearn.ensemble.RandomForestRegressor model and plots them in descending
        order the feature importances of that model.

        KWARGS:
            features_list <list>: List of features in self.df to use for modeling
            kwargs <dict>:
                RandomForestModel <sklearn.ensemble.RandomForest*>:
                    hasattr(feature_importances_)
                forest_kwargs <dict>: Keyword args for RandomForestModel training
                pretrained <bool>: If True model will not be trained
                title <str>: Title of plot output
                outpath <path-like>: Output file name of image if want to save
                use_top_n <int>: Number of feaures to use in plotting
                figsize <tuple>: Size of the figure
        '''
        y = self.df[self.target_col[0]]
        if features_list:
            X = self.df[features_list]
            rf_feature_importances(X, y, **kwargs)
        else:
            X = self.df[self.bin_cols + self.num_cols]
            rf_feature_importances(X, y, **kwargs)

    # TODO:
    # @property
    # def profile_report(self):
    #     '''Runs pandas_profiler.ProfileReport on self.df to return an HTML
    #     rendering of the data's summary.  Also a property.
    #
    #     RETURNS:
    #         pandas_profiler.ProfileReport of self.df
    #     '''
    #     return ProfileReport(self.df)
    #
    # def identify_reject_columns(self, threshold=.90):
    #     '''Leverages pandas_profiler.ProfileReport object of self to use the
    #     method get_rejected_variables.  Returns all variables that are stable
    #     insofar as they are not highly correlated with other variables.
    #
    #     KWARGS:
    #     RETURNS:
    #     '''
    #     return self.profile_report.get_rejected_variables(threshold=threshold)



# END
