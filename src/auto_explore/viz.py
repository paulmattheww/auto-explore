import matplotlib.pyplot as plt
import seaborn as sns
from numba import double
from numba.decorators import jit
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scikitplot.metrics import plot_silhouette

kmeans_kwargs = dict(n_clusters=n_clusters, random_state=777)

def cluster_and_plot_pca(df,
                         cluster_range=np.arange(2, 9),
                         ClusterAlgorithm=KMeans,
                         cluster_kwargs=kmeans_kwargs):
    '''An unsupervised learning approach to visualizing the number of clusters
    that are appropriate for a given dataset `df`.  
    '''
    pca = PCA(n_components=2, random_state=777)
    X_pca = pd.DataFrame(pca.fit(df.fillna(0)).transform(df.fillna(0)))

    for n_clusters in cluster_range:
        clust_col = "clusters_"+str(n_clusters)

        # perform kmeans
        clust = ClusterAlgorithm(**cluster_kwargs)
        clust.fit(df)
        X_pca[clust_col] = kmeans.labels_

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
        ax.set_title(f"Silhouette Plot Clusters={n_clusters}", size=18)

    plt.show()

@jit
def correlation_heatmap(df, cutoff=None, title='', outpath=None, type='pearson'):
    '''Performs a correlation heatmap on a pd.DataFrame object.  Uses numba's
    `jit` decorator to eliminate memory expansion (in this instance at the cost
    of ~5% speed).

    ARGS:

    KWARGS:

    RETURNS:
    '''
    df_corr = df.corr(type)
    np.fill_diagonal(df_corr.values, 0)
    if cutoff != None:
        for col in df_corr.columns:
            df_corr.loc[df_corr[col].abs() <= cutoff, col] = 0
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(df_corr, ax=ax, cmap='RdBu_r')
    plt.suptitle(title, size=18)
    if outpath == None:
        pass
    else:
        plt.savefig(outpath)
    plt.show()
    return df_corr
