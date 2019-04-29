
import scipy import scipy.stats

def characterize_possible_distributions(y):
    '''Uses log-likelihood to determine which distribution is a best fit
    for a given vector y.
    '''
    param_dict = dict()
    dist_names = ['gamma', 'beta', 'rayleigh', 'norm', 'pareto']
    for dist_name in dist_names:
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(y)
        param_dict[dist_name] = param
    return param_dict


class Analyzers:
    def __init__(self):
        pass
