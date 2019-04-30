# Import statements
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import double
from numba.decorators import jit
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scikitplot.metrics import plot_silhouette
from yellowbrick.text import TSNEVisualizer
import matplotlib as mpl
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Settings preferred
mpl.rcParams['figure.figsize'] = (13, 9)
sns.set_style("whitegrid")

# Declaration of constants
cluster_kwargs = dict(random_state=777)
text_kwargs = dict(ngram_range=(1,3), min_df=3, max_features=1000)


def categorical_frequency_distribution(df, cat_cols, top_n=None):
    '''Characterizes the frequency distribution of either the top_n most
    populous distinct categorical values within a given cat_col.  Specify top_n
    if you see there are too many unique categorical values in your dataset.

    ARGS:
        df <pd.DataFrame>: Data containing cat_cols plus potentially more.
        cat_cols <list>: List of columns with discrete categories.
    KWARGS:
        top_n <int>: Top N categories to show if there are too manuy.
    RETURNS:
        None, outputs plots to console.  
    '''
    top_n = top_n or df.shape[0]
    for col in cat_cols:
        title = f"Frequency Distribution of {col}"
        (df[col].value_counts().head(top_n) / df.shape[0]).plot.barh(title=title)
        ax = plt.gca()
        ax.grid(alpha=.4)
        plt.show()


def plot_tseries_over_group_with_histograms(df, xcol, ycol, grpcol,
                                            title_prepend='{}',
                                            labs=None, x_angle=0, labelpad=60,
                                            window=15, ignore_cols=[]):
    '''
    Function for plotting time series df[ycol] over datetime range df[xcol]
    using the unique_grp_vals contained in df[grpcol].unique().

        Example:
            title_prepend = 'Time Series for {}'
            xcol = 'date'
            ycol = 'rolling_15_mean'
            grpcol = 'variable'
            labs = dict(xlab='', ylab='Value')

            plot_tseries_over_group_with_histograms(smooth_df,
                                                    xcol, ycol, grpcol,
                                                    title_prepend, labs,
                                                    x_angle=90,
                                                    ignore_cols=onehot_cols)

    ARGS:
        df <pd.DataFrame>: containing datetime and series to plot
        xcol <str>: of column name in df for datetime series
        ycol <str>: of column name in df for tseries
    KWARGS:
        grpcol <str>: str of column name in df of group over which to plot
        labs <dict>: of xlab, ylab
        title_prepend <str>: containing "{}" that prepends group names in title
    RETURNS:
        None, plots a time series vector over groups with histogram on y axis.
    '''
    years = mdates.YearLocator()    # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')

    unique_grp_vals = df[grpcol].unique()
    nrows = len(unique_grp_vals) - len(ignore_cols)
    figsize = (13, 6 * nrows)
    fig, axes = plt.subplots(nrows, 1, figsize=figsize)
    title_prepend_hist = 'Histogram of ' + str(title_prepend)
    j = 0
    for i, grp in enumerate(unique_grp_vals):
        _df = df.loc[df[grpcol] == grp]
        if grp not in ignore_cols:
            _df = df.loc[df[grpcol] == grp]
            ax = axes[j]
            ax.plot(_df[xcol], _df[ycol], alpha=.2, color='black')
            ax.plot(_df[xcol], _df[ycol].rolling(window=window, min_periods=min(5, window)).mean(),
                    alpha=.5, color='r', label='{} period rolling avg'.format(window),
                    linestyle='--')
            longer_window = int(window * 3)
            ax.plot(_df[xcol], _df[ycol].rolling(window=longer_window, min_periods=5).mean(),
                    alpha=.8, color='darkred', label='{} period rolling avg'.format(longer_window),
                    linewidth=2)
            mu, sigma = _df[ycol].mean(), _df[ycol].std()
            ax.axhline(mu, linestyle='--', color='r', alpha=.3)
            ax.axhline(mu - sigma, linestyle='-.', color='y', alpha=.3)
            ax.axhline(mu + sigma, linestyle='-.', color='y', alpha=.3)
            ax.set_title(title_prepend.format(grp))
            ax.legend(loc='best')
            bottom, top = mu - 3*sigma, mu + 3*sigma
            ax.set_ylim((bottom, top))
            if labs is not None:
                ax.set_xlabel(labs['xlab'])
                ax.set_ylabel(labs['ylab'])
            ax.xaxis.labelpad = labelpad
            ax.xaxis.set_minor_locator(months)
            ax.grid(alpha=.1)
            if x_angle != 0:
                for tick in ax.get_xticklabels():
                    tick.set_rotation(x_angle)

            divider = make_axes_locatable(ax)
            axHisty = divider.append_axes('right', 1.2, pad=0.1, sharey=ax)
            axHisty.grid(alpha=.1)
            axHisty.hist(_df[ycol].dropna(), orientation='horizontal', alpha=.5, color='lightgreen', bins=25)
            axHisty.axhline(mu, linestyle='--', color='r', label='mu', alpha=.3)
            axHisty.axhline(mu - sigma, linestyle='-.', color='y', label='+/- two sigma', alpha=.3)
            axHisty.axhline(mu + sigma, linestyle='-.', color='y', alpha=.3)
            axHisty.legend(loc='best')

            j += 1
        else:
            pass

    sns.set_style("whitegrid")
    sns.despine()
    plt.show()


def text_cluster_tsne(text_vector,
                      TextVectorizer=TfidfVectorizer,
                      text_kwargs=text_kwargs,
                      n_clusters=10,
                      labels=None):
    '''Uses a TextVectorizer to transform the text contained (at the sentence
    or paragraph level) in the text_vector arg to produce a TSNE visualization.
    The label for the final plot is clusters produced from KMeans if labels
    are not passed.

    ARGS:
        text_vector <np.array>: Vector of text units.  Must be type str.
    KWARGS:
        TextVectorizer <sklearn.feature_extraction.text>: Transformer.
        text_kwargs <dict>: kwargs to pass to TextVectorizer
        n_clusters <int>: If not using labels, number of clusters in KMeans
        labels <np.array>: True categorical labels.  Discrete.
    RETURNS:
        None, prints visualizations to the console.
    '''
    txt_vctzr = TextVectorizer(**text_kwargs)
    docs = txt_vctzr.fit_transform(text_vector)
    tsne = TSNEVisualizer()

    if labels is None:
        # derive clusters if labels not provided
        clusters = KMeans(n_clusters=n_clusters)
        clusters.fit(docs)
        tsne.fit(docs, ["cluster_{}".format(c) for c in clusters.labels_])
    else:
        # otherwise use labels
        tsne.fit(docs, labels)
    sns.despine()
    tsne.poof()


def cluster_and_plot_pca(df,
                         cluster_range=np.arange(2, 9),
                         ClusterAlgorithm=KMeans,
                         cluster_kwargs=cluster_kwargs):
    '''An unsupervised learning approach to visualizing the number of clusters
    that are appropriate for a given dataset `df`.  If using another
    ClusterAlgorithm than KMeans it must accept the kwarg n_clusters.

    Data should have no missing values and all columns should have already
    been transformed into numeric datatypes (i.e. converting categorical
    features into one-hot encoded vectors).

        Example:
            from sklearn.cluster import MiniBatchKMeans

            clust_kwargs = dict(random_state=77)
            cluster_and_plot_pca(wine_df,
                                ClusterAlgorithm=MiniBatchKMeans,
                                cluster_kwargs=clust_kwargs)

    ARGS:
        df <pd.DataFrame>: DataFrame with all numeric data types and no
            missing values.
    KWARGS:
        cluster_range <list> or <array>: List of integers signifying the number
            of clusters to test in sequence.
        ClusterAlgorithm <sklearn.cluster>: Currently supports KMeans
            and MiniBatchKMeans.
        cluster_kwargs <dict>: kwargs for ClusterAlgorithm, not including
            num_clusters.
    RETURNS:
        None.  Plots are printed to the console.
    '''
    pca = PCA(n_components=2, random_state=777)
    X_pca = pd.DataFrame(pca.fit(df).transform(df))

    for n_clusters in cluster_range:
        clust_col = "clusters_"+str(n_clusters)

        # perform kmeans or some other clustering algorithm
        clust = ClusterAlgorithm(n_clusters=n_clusters, **cluster_kwargs)
        clust.fit(df)
        X_pca[clust_col] = clust.labels_

        # plot PCA with segments
        fig, ax = plt.subplots()
        for _, x in X_pca.groupby(clust_col):
            ax.scatter(x[0], x[1], label=str(_), alpha=.6)
        ax.grid(alpha=.4)
        sns.despine()
        ax.set_title(f"Clusters={n_clusters} Projected on 2D Principal Components",
                    size=12)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="best")

        # plot silhouette plot
        plot_silhouette(X_pca, clust.labels_)
        ax = plt.gca()
        ax.grid(alpha=.4)
        sns.despine()
        ax.set_title(f"Silhouette Plot Clusters={n_clusters}", size=12)

    plt.show()

@jit
def correlation_heatmap(df, cutoff=None, title='', outpath=None, type='pearson'):
    '''Performs a correlation heatmap on a pd.DataFrame object.  Uses numba's
    `jit` decorator to eliminate memory expansion (in this instance at the cost
    of ~5% speed).

    ARGS:
        df <pd.DataFrame> or <dd.DataFrame>: Data of numeric vectors w/o
            missing values.
    KWARGS:
        cutoff <float>: Absolute value of cutoff to visualize a given
            correlation between two vectors.  0<x<1.
        title <str>: Title of the heatmap plot.
        outpath <str>: Out path with extension to save image file to.
        type <str>: 'pearson' or 'spearman'
    RETURNS:
        df_corr <pd.DataFrame>: Correlation matrix with variable names.
    '''
    if type not in ['pearson', 'spearman']:
        type = 'pearson'
    df_corr = df.corr(type)
    np.fill_diagonal(df_corr.values, 0)
    if cutoff is not None:
        for col in df_corr.columns:
            df_corr.loc[df_corr[col].abs() <= cutoff, col] = 0
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(df_corr, ax=ax, cmap='RdBu_r')
    plt.suptitle(title, size=18)
    if outpath is None:
        pass
    else:
        plt.savefig(outpath)
    plt.show()
    return df_corr
