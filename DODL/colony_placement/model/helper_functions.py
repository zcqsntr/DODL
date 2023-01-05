from scipy.ndimage import laplace
import numpy as np
from scipy.ndimage import laplace as lp


def leaky_hill(s, K, lam, min, max):
    # get rid of the very small negative values
    s[s < 0] = 0

    h = (max - min) * s ** lam / (K ** lam + s ** lam) + min
    return h


def leaky_inverse_hill(s, K, lam, min, max):
    # get rid of the very small negative values
    s[s < 0] = 0

    h = (max - min) * K ** lam / (K ** lam + s ** lam) + min
    return h


def hill(s, K, lam):
    s[s < 0] = 0
    h = s**lam / (K**lam + s**lam)
    return(h)


def ficks(s, w, laplace = False):
    if not laplace: # if no custom laplace given then use default square boundary conds
        return(lp(s) / np.power(w, 2))
        #return(lp(s, mode = 'nearest') / np.power(w, 2))
    else:
        return (laplace(s) / np.power(w, 2))


