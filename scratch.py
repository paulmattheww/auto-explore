
#pipenv shell ipython

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

from auto_explore.viz import text_cluster_tsne
from auto_explore.viz import cluster_and_plot_pca
from auto_explore.viz import correlation_heatmap
from auto_explore.viz import categorical_frequency_distribution
from auto_explore.viz import plot_tseries_over_group_with_histograms
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

cluster_and_plot_pca(wine_df)
correlation_heatmap(wine_df)

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
plot_tseries_over_group_with_histograms(stack_df, 'date', 'close', 'ix')

# from auto_explore.eda import AutopilotExploratoryAnalysis
# args = (df, bin_cols, cat_cols, num_cols, text_cols)
# kwargs = dict(target_col=target_col)
# ax = AutopilotExploratoryAnalysis(*args, **kwargs)
