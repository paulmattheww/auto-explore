'''
TODO:
make color palettes more thought out
look into PLOTNINE for ggplot2 type stuff.
'''

# Import statements
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import double
from numba.decorators import jit
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scikitplot.metrics import plot_silhouette
from yellowbrick.text import TSNEVisualizer
import matplotlib as mpl
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.ensemble import RandomForestClassifier


# Settings preferred
mpl.rcParams['figure.figsize'] = (13, 9)
sns.set_style("whitegrid")

# Declaration of constants
cluster_kwargs = dict(random_state=777)
text_kwargs = dict(ngram_range=(1,3), min_df=3, max_features=1000)

def derive_optimal_clusters(df, cluster_range=np.arange(2, 11),
                            num_cols=None, Scaler=StandardScaler,
                            ClusterAlgorithm=KMeans, show=False):
    '''Without changing kwargs this function scales the data with a
    StandardScaler for numeric columns, then performs KMeans clustering
    over a range of clusters.

    ARGS:
        df <pd.DataFrame>: Data in question to cluster.  Ensure there are
            not any missing values, and all categoricals are converted to
            numeric data types
    KWARGS:
        kwargs <dict>: dict(cluster_range=np.arange(2, 11),
                                    num_cols=None, Scaler=StandardScaler,
                                    ClusterAlgorithm=KMeans)
    RETURNS:
        None, plots to console
    '''
    # scale the data
    scl = Scaler()
    scl.fit(df[num_cols])
    df[num_cols] = scl.transform(df[num_cols])

    # iterate over
    sse = list()
    for k in cluster_range:
        clust = ClusterAlgorithm(n_clusters=k)
        clust.fit(df)
        sse.append(clust.inertia_)

    # plot
    fig, ax = plt.subplots()
    ax.plot(cluster_range, sse, 'bx-', linestyle=':')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Sum of Squared Distances')
    ax.set_title('Elbow Method for Optimal K Clusters', size=18)
    sns.despine()
    ax.grid(alpha=.4)
    plt.show()

def lm_group_plot(df, x, y, grp, size=5, aspect=1.5, title="", show=True):
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
    sns.lmplot(x=x, y=y, data=df, hue=grp, size=size, aspect=aspect)
    ax = plt.gca()
    sns.despine()
    ax.grid(alpha=.4)
    ax.set_title(title, size=24)
    if show:
        plt.show()

def scatterplot_matrix_kde(df):
    '''Plots seaborn's version of a PairGrid using a 2-D KDE plot
    on the lower half, a 1-D KDE plot on the diagonal, and a scatterplot
    on the upper half.

    You may want to sample your data if you have a lot of data, especially
    if you have many dimensions.  Works best on smaller datasets.

    ARGS:
    KWARGS:
    RETURNS:
    '''
    g = sns.PairGrid(df, diag_sharey=False)
    g.map_lower(sns.kdeplot)
    g.map_upper(sns.scatterplot)
    g.map_diag(sns.kdeplot, lw=3)
    plt.show()


def rf_feature_importances(X, y, RandomForestModel=RandomForestClassifier,
                           forest_kwargs=dict(random_state=777), pretrained=False,
                           title='', outpath=None, use_top_n=None, figsize=(13, 8)):
    '''Derives feature importances from an sklearn.ensemble.RandomForestClassifier
    or sklearn.ensemble.RandomForestRegressor model and plots them in descending
    order the feature importances of that model.

    ARGS:
        X <pd.DataFrame>:
        y <array>:
    KWARGS:
        RandomForestModel <sklearn.ensemble.RandomForest*>: hasattr(feature_importances_)
        forest_kwargs <dict>: Keyword args for RandomForestModel training
        pretrained <bool>: If True model will not be trained
        title <str>: Title of plot output
        outpath <path-like>: Output file name of image if want to save
        use_top_n <int>: Number of feaures to use in plotting
        figsize <tuple>: Size of the figure
    '''
    if not pretrained:
        forest = RandomForestModel(**forest_kwargs)
        forest.fit(X, y)
    else:
        forest = RandomForestModel
    forest.features_list_ = X.columns.tolist()

    feats = {}
    for feature, importance in zip(forest.features_list_, forest.feature_importances_):
        feats[feature] = importance

    fig, ax = plt.subplots(figsize=figsize)
    feats = pd.Series(feats).sort_values()
    if use_top_n:
        feats = feats[-use_top_n:]
    feats.plot(kind="barh", ax=ax)
    ax = plt.gca()
    sns.despine()
    ax.set_title(title)
    if outpath:
        plt.savefig(outpath)
    plt.show()



def target_distribution_over_binary_groups(df, binary_cols, target_col,
                                          plot_type='boxenplot',
                                          plot_kwargs={}):
    '''For use during feature engineering.  Pass a DataFrame with a list of
    `binary_cols` that represent the names of columns that are binary categories.
    The `target_col` str is the variable you are trying to model.  Requires
    seaborn >= 0.9.0.

    ARGS:
        df <pd.DataFrame>: DataFrame that contains all columns passed
        binary_cols <list>: List of binary columns as either int or bool
        target_col <str> or <list>: Name of target column. assert len(list) == 1
    KWARGS:
        plot_type <str>: Either 'boxenplot', 'boxplot', or 'violinplot'
        plot_kwargs <dict>: Keyword arguments for plot_type passed
    RETURNS:
        None, plots to the console.
    '''
    for col in binary_cols:
        if plot_type=='boxenplot':
            sns.boxenplot(y=df[target_col], x=df[col], **plot_kwargs)
        elif plot_type=='violinplot':
            sns.violinplot(y=df[target_col], x=df[col], **plot_kwargs)
        else:
            sns.boxplot(y=df[target_col], x=df[col], **plot_kwargs)
        ax = plt.gca()
        mu0, mu1 = df[target_col].groupby(df[col]).mean()
        sd0, sd1 = df[target_col].groupby(df[col]).std()
        ncol = df.loc[df[col]==1].shape[0]
        ax.axhline(mu0, label=f'mean = {round(mu0, 2)}|{col} = 0', color='blue', linestyle=':')
        ax.axhline(mu1, label=f'mean = {round(mu1, 2)}|{col} = 1 with {ncol} observations',
                   color='orange', linestyle='-.')
        ax.grid(alpha=.4)
        ax.set_title(col)
        sns.despine()
        ax.legend(loc='best')
        plt.show()


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
                                            vertical_predicate_col=None, vertical_value_col=None, title_prepend='{}',
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
            ax.plot(_df[xcol], _df[ycol], alpha=.6, linestyle=':', color='blue')
            ax.plot(_df[xcol], _df[ycol].rolling(window=window, min_periods=min(5, window)).mean(),
                    alpha=.5, color='r', label='{} period rolling avg'.format(window),
                    linestyle='--')
            longer_window = int(window * 3)
            ax.plot(_df[xcol], _df[ycol].rolling(window=longer_window, min_periods=5).mean(),
                    alpha=.8, color='darkred', label='{} period rolling avg'.format(longer_window),
                    linewidth=2)

            # get mean and std form subset
            mu, sigma = _df[ycol].mean(), _df[ycol].std()
            ax.axhline(mu, linestyle='--', color='r', alpha=.2, label=f"mean = {round(mu, 2)}")
            ax.axhline(mu - sigma, linestyle='-.', color='y', alpha=.2, label=f"mean - std = {round(mu - sigma, 2)}")
            ax.axhline(mu + sigma, linestyle='-.', color='y', alpha=.2, label=f"mean + std = {round(mu + sigma, 2)}")
            ax.set_title(title_prepend.format(grp))
            ax.legend(loc='best')
            bottom, top = mu - 4*sigma, mu + 4*sigma
            bottom = max(0, bottom)
            ax.set_ylim((bottom, top))

            # set labels on X and Y axes
            if labs:
                ax.set_xlabel(labs['xlab'])
                ax.set_ylabel(labs['ylab'])

            # configure axes labelpads and months
            ax.xaxis.labelpad = labelpad
            ax.xaxis.set_minor_locator(months)
            ax.grid(alpha=.3)

            # angle dates if gnarly
            if x_angle != 0:
                for tick in ax.get_xticklabels():
                    tick.set_rotation(x_angle)

            # acquire vertical lines on the X axis
            if vertical_predicate_col:
                for ix, row in df.iterrows():
                    if row[vertical_predicate_col] == True:
                        ax.axvline(row[vertical_value_col], alpha=.2, color='gray')

            # switch axes to the right side for histogram
            divider = make_axes_locatable(ax)
            axHisty = divider.append_axes('right', 1.2, pad=0.1, sharey=ax)
            axHisty.grid(alpha=.1)
            axHisty.hist(_df[ycol].dropna(), orientation='horizontal', alpha=.5, color='lightgreen', bins=25)
            axHisty.axhline(mu, linestyle='--', color='r', label='mu', alpha=.3)
            axHisty.axhline(mu - sigma, linestyle='-.', color='y', label='+/- 1 sigma', alpha=.3)
            axHisty.axhline(mu + sigma, linestyle='-.', color='y', alpha=.3)
            axHisty.legend(loc='best')

            j += 1

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

    # return the clustering algorithm for predictions
    return clust

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
