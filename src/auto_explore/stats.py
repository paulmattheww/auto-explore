
from scipy import stats
import numpy as np
import pandas as pd

DIST_NAMES = ['gamma', 'beta', 'rayleigh', 'norm', 'pareto', 'uniform', 'logistic']

def best_theoretical_distribution(data, bins=100, dist_names=DIST_NAMES):
    '''Uses log-likelihood to determine which distribution is a best fit
    for a given 1D array by characterizing the sum of squared errors of each
    model on the actual data.

    ARGS:
        data <np.array>: 1D Vector with no missing values to characterize
    KWARGS:
        bins <int>: Number of bins to use in np.histogram.
    RETURNS:
        param_dict {dict}: results of dist.fit(data) and the sum of squared
            errors for each of the distributions in dist_names
    '''
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    param_dict = dict()
    for dist_name in dist_names:
        # Isolate distribution by name
        dist = getattr(stats, dist_name)
        params = dist.fit(data)

        # Separate parts of parameters
        arg, loc, scale = params[:-2], params[-2], params[-1]

        # Calculate fitted PDF and error with fit in distribution
        pdf = dist.pdf(x, loc=loc, scale=scale, *arg)
        sse = np.sum(np.power(y - pdf, 2.0))

        # Persist
        param_dict[dist_name] = sse

    return  pd.Series(param_dict)


class Analyzers:
    def __init__(self):
        pass
