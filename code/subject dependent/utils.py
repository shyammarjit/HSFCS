# This code is created by Shyam Marjit. Some functions and codes are taken from 
# https://github.com/nikdon/pyEntropy and https://github.com/raphaelvallat/antropy.
# I give them credit for all these taken parts of the code.

# Import python library
from math import factorial, floor, log, sqrt
import numpy as np
from sklearn.neighbors import KDTree
from scipy.stats import kurtosis, mode, skew

def first_difference(input_data):
    # First order difference
    fd, temp = 0, 0
    for i in range(0, input_data.shape[0]-1):
        temp = abs(input_data[i+1]-input_data[i])
        fd += temp
    return fd/(input_data.shape[0]-1)

def second_difference(input_data):
    # second order difference
    sd, temp = 0, 0
    for i in range(0, input_data.shape[0] - 2):
        temp = abs(input_data[i+2]-input_data[i])
        sd += temp
    return sd/(input_data.shape[0]-2)

def avg_and_rms_power(input_data):
    # average power and root mean square of a signal
    mean_data, avg_power = np.mean(input_data), 0
    for i in range(input_data.shape[0]):
        temp = (mean_data - input_data[i])**2
        avg_power += temp
    return avg_power/(input_data.shape[0]), np.sqrt(avg_power/(input_data.shape[0]))

def num_zerocross(x, normalize=False, axis=-1):
    # Number of zero-crossings.
    x = np.asarray(x)
    nzc = np.diff(np.signbit(x), axis=axis).sum(axis=axis)
    if normalize:
        nzc = nzc / x.shape[axis]
    return nzc

def petrosian_fd(input_data):
    # Petrosian fractal dimension
    axis = -1
    x = np.asarray(input_data)
    N = x.shape[axis]
    # Number of sign changes in the first derivative of the signal
    nzc_deriv = num_zerocross(np.diff(x, axis=axis), axis=axis)
    pfd = np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * nzc_deriv)))
    return pfd

def shannon_entropy(time_series):
    if not isinstance(time_series, str): # Check if string
        time_series = list(time_series)
    data_set = list(set(time_series)) # Create a frequency data
    freq_list = []
    for entry in data_set:
        counter = 0.
        for i in time_series:
            if i == entry:
                counter += 1
        freq_list.append(float(counter) / len(time_series))
    # Shannon entropy
    ent = 0.0
    for freq in freq_list:
        ent += freq * np.log2(freq)
    return -ent

def _embed(x, order=3, delay=1):
    # Time-delay embedding
    x = np.asarray(x)
    N = x.shape[-1]
    if x.ndim == 1: # 1D array (n_times)
        Y = np.zeros((order, N - (order - 1) * delay))
        for i in range(order):
            Y[i] = x[(i * delay) : (i * delay + Y.shape[1])]
        return Y.T
    else: # 2D array (signal_indice, n_times)
        Y = []
        # pre-defiend an empty list to store numpy.array (concatenate with a list is faster)
        embed_signal_length = N - (order - 1) * delay
        # define the new signal length
        indice = [[(i * delay), (i * delay + embed_signal_length)] for i in range(order)]
        # generate a list of slice indice on input signal
        for i in range(order):
            # loop with the order
            temp = x[:, indice[i][0] : indice[i][1]].reshape(-1, embed_signal_length, 1)
            # slicing the signal with the indice of each order (vectorized operation)
            Y.append(temp)
            # append the sliced signal to list
        return np.concatenate(Y, axis=-1)

def app_entropy(x, order=2, metric="chebyshev", approximate=True):
    _all_metrics = KDTree.valid_metrics
    phi = np.zeros(2)
    r = 0.2 * np.std(x, ddof=0)
    _emb_data1 = _embed(x, order, 1)
    if approximate:
        emb_data1 = _emb_data1
    else:
        emb_data1 = _emb_data1[:-1]
    count1 = (KDTree(emb_data1, metric=metric).query_radius(emb_data1, r, count_only=True).astype(np.float64))
    emb_data2 = _embed(x, order + 1, 1)
    count2 = (KDTree(emb_data2, metric=metric).query_radius(emb_data2, r, count_only=True).astype(np.float64))
    if approximate:
        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
    return np.subtract(phi[0], phi[1])

def AntroPy_xlogx(x, base=2):
    """Returns x log_b x if x is positive, 0 if x == 0, and np.nan
    otherwise. This handles the case when the power spectrum density
    takes any zero value.
    """
    x = np.asarray(x)
    xlogx = np.zeros(x.shape)
    xlogx[x < 0] = np.nan
    valid = x > 0
    xlogx[valid] = x[valid] * np.log(x[valid]) / np.log(base)
    return xlogx

def perm_entropy(x, order=3, delay=1, normalize=False):
    # Permutation Entropy
    # If multiple delay are passed, return the average across all d
    if isinstance(delay, (list, np.ndarray, range)):
        return np.mean([perm_entropy(x, order=order, delay=d, normalize=normalize) for d in delay])
    x = np.array(x)
    ran_order = range(order)
    hashmult = np.power(order, ran_order)
    # Embed x and sort the order of permutations
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind="quicksort")
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = - AntroPy_xlogx(p).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    return pe

def pyentrp_embed(x, order=3, delay=1):
    # Time-delay embedding.
    N = len(x)
    Y = np.empty((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T

def weighted_permutation_entropy(time_series, order=2, delay=1, normalize=False):
    x = pyentrp_embed(time_series, order=order, delay=delay)
    weights = np.var(x, axis=1)
    sorted_idx = x.argsort(kind='quicksort', axis=1)
    motifs, c = np.unique(sorted_idx, return_counts=True, axis=0)
    pw = np.zeros(len(motifs))
    # TODO hashmap
    for i, j in zip(weights, sorted_idx):
        idx = int(np.where((j == motifs).sum(1) == order)[0])
        pw[idx] += i
    pw /= weights.sum()
    b = np.log2(pw)
    wpe = -np.dot(pw, b)
    if normalize:
        wpe /= np.log2(factorial(order))
    return wpe

def _linear_regression(x, y):
    # Fast linear regression using Numba.
    epsilon = 10e-9
    n_times = x.size
    sx2, sx, sy, sxy = 0, 0, 0, 0
    for j in range(n_times):
        sx2 += x[j] ** 2
        sx += x[j]
        sxy += x[j] * y[j]
        sy += y[j]
    den, num = n_times * sx2 - (sx**2), n_times * sxy - sx * sy
    slope = num / (den + epsilon)
    intercept = np.mean(y) - slope * np.mean(x)
    return slope, intercept

def higuchi_fd(x, kmax=10):
    # Higuchi Fractal Dimension.
    x = np.asarray(x, dtype=np.float64)
    kmax = int(kmax)
    n_times = x.size
    lk = np.empty(kmax)
    x_reg = np.empty(kmax)
    y_reg = np.empty(kmax)
    for k in range(1, kmax + 1):
        lm = np.empty((k,))
        for m in range(k):
            ll = 0
            n_max = floor((n_times - m - 1) / k)
            n_max = int(n_max)
            for j in range(1, n_max):
                ll += abs(x[m + j * k] - x[m + (j - 1) * k])
            ll /= k
            ll *= (n_times - 1) / (k * n_max)
            lm[m] = ll
        # Mean of lm
        m_lm = 0
        for m in range(k):
            m_lm += lm[m]
        m_lm /= k
        lk[k - 1] = m_lm
        x_reg[k - 1] = log(1.0 / k)
        y_reg[k - 1] = log(m_lm)
    higuchi, int_ = _linear_regression(x_reg, y_reg)
    return higuchi

def statistical_features(input_data, advanced = True):
    # Statistical features
    # Mean, Variance, Mode, Skew, Standard Deviation, Kurtosis
    mean, var, mode_ = np.mean(input_data), np.var(input_data), float(mode(input_data)[0]),
    median, skew_, std = np.median(input_data), skew(input_data), np.std(input_data)
    kurtosis_ = kurtosis(input_data)
    if(advanced == True):
        # First Difference, Second Difference, Normalized, First Difference, Normalized Second Difference
        first_diff = first_difference(input_data)
        norm_first_diff = first_diff/std
        sec_diff = second_difference(input_data)
        norm_sec_diff = sec_diff/std
        return [mean, var, mode_, median, skew_, std, kurtosis_, first_diff, norm_first_diff, sec_diff, norm_sec_diff]
    return [mean, var, mode_, median, skew_, std, kurtosis_]