# Auto Explore

[![Build Status](https://travis-ci.com/paulmattheww/auto-explore.svg?branch=master)](https://travis-ci.com/paulmattheww/auto-explore)

The goal of this Python library is to create a reliable tool for performing a first-pass exploratory data analysis.  The hope is that ML developers & data analysts will shorten their iteration cycle time by using this tool.  

The earliest stages of a machine learning project require exploratory analysis to uncover raw features and insights that can be exploited in the modeling process.  Exploratory analysis typically follows a somewhat tree-like (and at times recursive) process where task-patterns emerge across projects.  By specifying certain parameters *a priori* about the data in question, a process that adheres to these task-patterns can be designed using open source tools to automate the majority of the "first pass" data analysis work -- freeing up time for deep-dive analyses, modeling, and deployment.

The open source projects that will be relied upon most for this project include:

- [pandas-profiling](https://github.com/pandas-profiling/pandas-profiling)
- [featexp](https://github.com/abhayspawar/featexp)
- [speedml](https://speedml.com/automate-exploratory-data-analysis/)
- [pandas](https://pandas.pydata.org/)
- [dask](https://github.com/dask/dask)

The delivery mechanism will be two-fold: a command line interface and a Python package usable in an interactive runtime environment.  Functionality will range from the bare minimum of feature identification to a more detailed reporting on the data.  A full suite will be available for patient users.  

While the term "automated" data analysis sounds difficult, the heavy lifting has been done by these library authors.  This project will simply be extending good work that already exists, meaning I will not need to spend considerable time re-inventing the wheel on already established techniques.  

A high-level design of the code is below.

```python
class AutopilotExploratoryAnalysis:
    '''
    Questions that need answering:
        - How the data should exist when feeding it in
        - Output to console or notebook vs. HTML
    '''
    def __init__(self, df, drop_cols, bin_cols, cat_cols, num_cols, text_cols,
                target_col=None, hue=None, na_tolerance=.10, time_dim=None, dask=True):
        '''
        ARGS:
            drop_cols, bin_cols, cat_cols, num_cols, text_cols
        KWARGS:
            target_col=None, hue=None, na_tolerance=.10, time_dim=None, dask=True
        '''
        pass

    @property
    def split_data(self):
        pass

    def modify_notebook_configuration(self):
        pass

    def characterize_missing_values(self):
        pass

    def fill_missing_values(self, groupby_cols=None):
        '''Several methods to impute missing data'''
        pass

    def high_level_profile(self):
        '''Run Pandas_Profiler'''
        pass

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

```
