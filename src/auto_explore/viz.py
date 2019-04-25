import matplotlib.pyplot as plt
import seaborn as sns

def correlation_heatmap(df, cutoff=None, title='', outpath=None, type='pearson'):
    '''Performs a correlation heatmap on a pd.DataFrame object.

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
