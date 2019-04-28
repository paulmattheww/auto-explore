# Import statements
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

# Declaration of constants
cluster_kwargs = dict(random_state=777)
text_kwargs = dict(ngram_range=(1,3), min_df=3, max_features=1000)


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
        tsne.fit(docs, ["c{}".format(c) for c in clusters.labels_])
    else:
        tsne.fit(docs, labels)

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
