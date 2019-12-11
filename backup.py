import glob, re
import numpy as np
import pandas as pd
from pandas import Timestamp
from sklearn import *
from datetime import datetime
from xgboost import XGBRegressor
import math
import statistics

from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plt

# import sklearn
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import gmean
from scipy.stats.mstats import gmean
# from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# ============ Step 05: Clustering using DBSCAN and Pairwise Similarity Evaluation ==============

# #############################################################################
# Generate sample data
# centers = [[1, 1], [-1, -1], [1, -1]]
# X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                             random_state=0)
# X = StandardScaler().fit_transform(X)

visitor_matrix = visitor_df.values
# print(type(visitor_array))
print('visitor_matrix:', visitor_matrix)

# Transpose the matrix for timeseries as row and timestamps as column ------ NOT WORKING HERE
visitor_matrix = visitor_matrix.transpose()
print('visitor_matrix after transposed:', visitor_matrix)

# Convert from int to float -  not working but scalar can do it for you
# visitor_matrix = float(visitor_matrix)

# Normalizing matrix in column
def scale(X, x_min, x_max):
    # Scale the matrix (column wise)
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/(denom * 1.0)

# def scale(X, x_min, x_max):
#     # Scale the entire matrix (not column wise)
#     nom = (X-X.min())*(x_max-x_min)
#     denom = X.max() - X.min()
#     denom = denom + (denom is 0)
#     return x_min + nom/(denom * 1.0)

print("visitor matrix:", visitor_matrix)

similarities = cosine_similarity(visitor_matrix)
print('pairwise dense output:\n {}\n'.format(similarities))

#also can output sparse matrices
similarities_sparse = cosine_similarity(visitor_matrix,dense_output=False)
print('pairwise sparse output:\n {}\n'.format(similarities_sparse))


visitor_matrix = scale(visitor_matrix, 0, 1)
print("visitor matrix after normalize:", visitor_matrix)
#Print second row
# print("visitor_matrix row:",visitor_matrix[1,:])
print("Maxima along the first axis:", np.amax(visitor_matrix, axis=0))
print("Maxima along the second axis:", np.amax(visitor_matrix, axis=1))

#  DBSCAN from scikit learn
def cluster_dbscan(matrix, distance_measure, eps):
    """
    Parameters
    ----------
    matrix: np.matrix
        The input matrix. If distance measure is sts, this should be the sts
        distance matrix. If other distance, this should be the time-series
        matrix of size ngenes x nsamples.
    distance_measure: str
        The distance measure, default is sts, short time-series distance.
        Any distance measure available in scikit-learn is available here.
        Note: multiple time-series is NOT supported for distances other than
        "sts".
    Returns
    -------
    cluster_labels: list of int
        A list of size ngenes that defines cluster membership.
    """
    # if (distance_measure == "sts"):
    #     dbs = DBSCAN(eps=eps, metric='precomputed', min_samples=2)
    # else:
    dbs = DBSCAN(eps=eps, metric=distance_measure, min_samples=1)
    cluster_labels = dbs.fit_predict(matrix)
    return cluster_labels

def zscore(x):
    """Computes the Z-score of a vector x. Removes the mean and divides by the
    standard deviation. Has a failback if std is 0 to return all zeroes.
    Parameters
    ----------
    x: list of int
        Input time-series
    Returns
    -------
    z: list of float
        Z-score normalized time-series
    """
    mean = np.mean(x)
    sd = np.std(x)
    if sd == 0:
        z = np.zeros_like(x)
    else:
        z = (x - mean)/sd
    return z

def normalize_simple(matrix):
    """Normalizes a matrix by columns, and then by rows. With multiple
    time-series, the data are normalized to the within-series total, not the
    entire data set total.
    Parameters
    ----------
    matrix: np.matrix
        Time-series matrix of abundance counts. Rows are sequences, columns
        are samples/time-points.
    mask: list or np.array
        List of objects with length matching the number of timepoints, where
        unique values delineate multiple time-series. If there is only one
        time-series in the data set, it's a list of identical objects.
    Returns
    -------
    normal_matrix: np.matrix
        Matrix where the columns (within-sample) have been converted to
        proportions, then the rows are normalized to sum to 1.
    """
    normal_matrix = matrix / matrix.sum(0)
    # normal_matrix[np.invert(np.isfinite(normal_matrix))] = 0
    # for mask_val in np.unique(mask):
    #     y = normal_matrix[:, np.where(mask == mask_val)[0]]
    #     y = np.apply_along_axis(zscore, 1, y)
    #     normal_matrix[:, np.where(mask == mask_val)[0]] = y
    #     del y
    return normal_matrix

def normalize_clr(matrix, delta = 0.65, threshold = 0.5):
    """Normalizes a matrix by centre log ratio transform with zeros imputed
    by the count zero multiplicative method from the zCompositions package
    by Javier Palarea-Albaladejo and Josep Antoni Martin-Fernandez. Uses two
    parameters, delta and threshold, identically to the zCompositions
    implementation. This scheme is the same as used by the CoDaSeq R package.
    Parameters
    ----------
    matrix: np.matrix
        Time-series matrix of abundance counts. Rows are sequences, columns
        are samples/time-points.
    delta: float
        Fraction of the upper threshold used to impute zeros (default=0.65)
    threshold: float
        For a vector of counts, factor applied to the quotient 1 over the
        number of trials (sum of the counts) used to produce an upper limit
        for replacing zero counts by the CZM method (default=0.5).
    Returns
    -------
    normal_matrix: np.matrix
        Matrix where the columns (within-sample) have been converted to centre
        log ratio transformed values to control for within-sample
        compositionality, and the rows are brought onto the same scale by
        computing the Z-score of each element in the time-series.
    """

    #Zero imputation with count zero multiplicative
    # This algorithm was originally written with samples as rows
    # so we need the transpose
    X = matrix.T
    #N = nsamples
    N = X.shape[0]
    #D = nsequences
    D = X.shape[1]
    #Column means without 0's included
    n = np.apply_along_axis(lambda x: x[np.nonzero(x)].sum(), 1, X)
    #Replacement matrix
    replace = delta*np.ones((D,N))*(threshold/n)
    replace = replace.T
    #Normalize by columns, using only nonzero values
    X2 = np.apply_along_axis(lambda x: x/(x[np.nonzero(x)].sum()), 1, X)
    colmins = np.apply_along_axis(lambda x: x[np.nonzero(x)].min(), 0, X2)
    corrected = 0
    for idx, row in enumerate(X2):
        zero_indices = np.where(row == 0)[1]
        nonzero_indices = np.where(row != 0)[1]
        X2[idx, zero_indices] = replace[idx, zero_indices]
        over_min = np.where(X2[idx, zero_indices] > colmins[zero_indices])
        if len(over_min[0]) > 0:
            corrected += len(over_min[0])
            X2[idx, over_min[1]] = delta*colmins[over_min[1]]
        X2[idx, nonzero_indices] = (1-X2[idx, zero_indices].sum()) * \
                                   X2[idx, nonzero_indices]
    normal_matrix = X2.T
    # Do the CLR transform
    normal_matrix = normal_matrix/gmean(normal_matrix)
    # Normalize within time-series to remove scaling factors
    normal_matrix = np.apply_along_axis(zscore, 1, normal_matrix)
    return normal_matrix

#  Main method
cluster_dbscan

# This not work with the old eps value
# visitor_matrix = StandardScaler().fit_transform(visitor_matrix)

# #############################################################################
# Compute DBSCAN
# db = DBSCAN(eps=3.7, min_samples=1).fit(visitor_matrix)
# I don't understand this eps value for normalized matrix
db = DBSCAN(eps=3.7, min_samples=1).fit(visitor_matrix)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
# Black removed and is used for noise instead.
unique_labels = set(labels)
print("labels:", labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = visitor_matrix[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=7)

    xy = visitor_matrix[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=4)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

# run_cluster(visitor_matrix, 'euclidean', param_min = 3.6, param_max = 4.2, param_step = 0.1, clr = False)
# run_cluster(visitor_matrix, 'euclidean', param_min = 3.6, param_max = 3.7, param_step = 0.1, clr = False)

# run_cluster(visitor_matrix, 'cityblock', param_min = 0.01, param_max = 1, param_step = 0.01, clr = False)
# run_cluster(visitor_matrix, 'cosine', param_min = 0.01, param_max = 1, param_step = 0.01, clr = False)
# run_cluster(visitor_matrix, 'manhattan', param_min = 0.01, param_max = 1, param_step = 0.01, clr = False)
# run_cluster(visitor_matrix, 'l1', param_min = 0.01, param_max = 1, param_step = 0.01, clr = False)
# run_cluster(visitor_matrix, 'l2', param_min = 0.01, param_max = 1, param_step = 0.01, clr = False)
# run_cluster(visitor_matrix, 'hamming', param_min = 0.01, param_max = 1, param_step = 0.01, clr = False)

# dbs = DBSCAN(eps=eps, metric=distance_measure, min_samples=2)
# cluster_labels = cluster_dbscan(visitor_matrix, distance_measure, eps)

# for i in list_timeseries:
#     ts = i
#     ts.plot()
#     plt.show()
#     # print("I:", i)
#
# ts1 = list_timeseries[0]
# ts2 = list_timeseries[1]

def euclid_dist(t1,t2):
    # TypeError: unsupported operand type(s) for +: 'int' and 'str'
    return math.sqrt(sum((t1-t2)**2))

def DTWDistance(s1, s2,w):
    DTW={}

    w = max(w, abs(len(s1)-len(s2)))

    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return math.sqrt(DTW[len(s1)-1, len(s2)-1])

def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):

        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2

    return math.sqrt(LB_sum)

