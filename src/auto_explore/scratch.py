
#pipenv shell ipython

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

from auto_explore.viz import *
from auto_explore.apis import fetch_fred_data
from auto_explore.datetime import make_calendars

from auto_explore.eda import AutopilotExploratoryAnalysis

def get_wine_data():
    '''Fetches wine data from sklearn.datasets'''
    wine = load_wine()
    wine_df = pd.DataFrame(wine['data'], columns=wine['feature_names'])
    wine_df['target'] = wine['target']
    return wine_df

wine_df = get_wine_data()

#cluster_and_plot_pca(wine_df)
#correlation_heatmap(wine_df)

# datatype sensors
series_list = ['SP500', 'NASDAQCOM', 'DJIA', 'RU2000PR'] # cboe energy sector etf volatility
econ_df = fetch_fred_data(series_list)

year_list = np.arange(2011, 2020)
cal_df = make_calendars(year_list, drop_index=False)
cal_df.index.name = 'DATE'
df = cal_df.join(econ_df).dropna()
df['date'] = df.index.values

cat_cols = ['month', 'year', 'weekday']
bin_cols = ['is_weekday', 'is_holiday', 'is_holiday_week']
num_cols = ['NASDAQCOM', 'DJIA', 'RU2000PR']
text_cols = []
target_col = ['SP500']

stack_df = pd.DataFrame(df[['SP500', 'NASDAQCOM']].stack()).reset_index(drop=False)
stack_df.columns = ['date', 'ix', 'close']
#plot_tseries_over_group_with_histograms(stack_df, 'date', 'close', 'ix')


# text cluster
from sklearn.datasets import fetch_20newsgroups
txt = fetch_20newsgroups(subset='train').data
#text_cluster_tsne(txt[:1000], n_clusters=6)

# rf_feature_importance
#rf_feature_importances(wine_df.drop(columns='target'), wine_df.target)


# target_distribution_over_binary_groups
pct_chg_df = df[['SP500', 'DJIA']].pct_change()
pct_chg_df['is_holiday_week'] = pct_chg_df.index.values
hol_map = dict(zip(cal_df.index.values, cal_df.is_holiday_week))
pct_chg_df['is_holiday_week'] = pct_chg_df['is_holiday_week'].map(hol_map)
#target_distribution_over_binary_groups(pct_chg_df, ['is_holiday_week'], 'SP500')


#
from sklearn.datasets import load_iris
data = load_iris()
iris_df = pd.DataFrame(data['data'], columns=data['feature_names'])
iris_df['target'] = data['target']




from auto_explore.eda import AutopilotExploratoryAnalysis

args = (df, bin_cols, cat_cols, num_cols, text_cols)
kwargs = dict(target_col=target_col)
ax = AutopilotExploratoryAnalysis(*args, **kwargs)


# test_split_data
len(ax.split_data) == 2

# test_fill_missing_values
ax.characterize_missing_values()

#TODO
# test_fill_missing_values

# test_characterize_distributions
#ax.characterize_distributions().idxmin(axis=0)

# test_scale_numeric_columns
ax.scale_numeric_columns()

# test_has_missing_values
ax.has_missing_values

# force missing and fill in
#ax.df.loc[ax.df.date == '2011-01-03', 'DJIA'] = np.nan
ax.fill_missing_values()

#ax.profile_report
ax.generate_univarite_plots()








#END
