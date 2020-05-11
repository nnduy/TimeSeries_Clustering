from scipy.spatial.distance import pdist

import math
from math import floor
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

def percentage(part, whole, digits):
    val = float(part)/float(whole)
    val *= 10 ** (digits + 2)
    return (floor(val) / 10 ** digits)

# DTW Distance between 2 time series with fully window size complexity of O(nm)
def DTWDistance(s1, s2):
    DTW={}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return math.sqrt(DTW[len(s1)-1, len(s2)-1])

# DTW Distance between 2 time series with specific window size w to increase speed
def DTWDistance(s1, s2, w=5):
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

# ============ Step 08: Clustering using OPTICS algorithms ==============

# method: Plotting current cluster and compait it with dbscan
# input:
#       X: input matrix
#       clust: cluster after using OPTICs algorithm
def plot_current_cluster(X, clust):

    space = np.arange(len(X))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]
    print("labels-labels:", labels)

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 3)
    ax1 = plt.subplot(G[0, :])
    ax2 = plt.subplot(G[1, 0])
    ax3 = plt.subplot(G[1, 1])
    ax4 = plt.subplot(G[1, 2])

    # Reachability plot
    colors = ['b.', 'r.', 'g.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        ax1.plot(Xk, Rk, color, alpha=0.3, markersize=16)
    ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
    ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
    ax1.set_ylabel('Reachability (epsilon distance)')
    ax1.set_title('Reachability Plot')

    # OPTICS
    colors = ['b.', 'r.', 'g.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = X[clust.labels_ == klass]
        ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, markersize=16)
    ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
    ax2.set_title('Automatic Clustering\nOPTICS')


    plt.tight_layout()
    plt.show()

    return 1

# #############################################################################
# Perform OPTICS clustering from vector array or distance matrix.
# input:
#       eps: float - The maximum distance between two samples for one to be considered as in the neighborhood of the other.
#       distance_measure: string - The metric to use when calculating distance between instances in a feature array.
#       min_samples: int - The number of samples (or total weight) in a neighborhood for a point to be considered
#           as a core point. This includes the point itself.
# input:
#       distance_measure: string - The metric to use when calculating distance between instances in a feature array.
def clustering_by_optics(df_imputation_optics):
    # print("df_imputation_optics input clustering_by_optics:\n", df_imputation_optics)
    # Creating index for new dataframe
    imputation_optics_arg_index = 0
    # print('type of imputation_optics_arg_index: {}', type(imputation_optics_arg_index))
    list_cols = list(df_imputation_optics.columns.values)
    list_cols.extend(['labels', 'reachability', 'nclusters', 'n_noise_', 'percent_of_noise'])
    # list_cols.extend(['labels', 'reachability', 'labels_050', 'labels_200'])
    df_imputation_optics_arg = pd.DataFrame(columns=list_cols)

    for i, row in df_imputation_optics.iterrows():
        # print("X:\n", df_imputation_optics.iloc[i]['X'])
        X = df_imputation_optics.iloc[i]['X']

        # # Pairwise distances between observations in n-dimensional space.
        # X_distance = pdist(X, metric=DTWDistance)
        # print("pdist(X, metric=DTWDistance) function: X_distance :", X_distance)

        METRIC_ARG = df_imputation_optics.iloc[i]['METRIC_ARG']
        MIN_SAMPLES_ARG = df_imputation_optics.iloc[i]['MIN_SAMPLES_ARG']
        XI_ARG = df_imputation_optics.iloc[i]['XI_ARG']
        MIN_CLUSTER_SIZE_ARG = df_imputation_optics.iloc[i]['MIN_CLUSTER_SIZE_ARG']
        OPTICS_MIN_CLUSTERS_ARG = df_imputation_optics.iloc[i]['OPTICS_MIN_CLUSTERS_ARG']
        OPTICS_MAX_NOISE_PERCENT_ARG = df_imputation_optics.iloc[i]['OPTICS_MAX_NOISE_PERCENT_ARG']
        print("OPTICS_MIN_CLUSTERS_ARG:", OPTICS_MIN_CLUSTERS_ARG)
        print("OPTICS_MAX_NOISE_PERCENT_ARG:", OPTICS_MAX_NOISE_PERCENT_ARG)

        if METRIC_ARG == 'DTWDistance':
            # The code below is equivalent the above but with one more option to choose 3rd argument
            clust = OPTICS(metric=lambda X, Y: DTWDistance(X, Y, w=5), min_samples=4, xi=.01, min_cluster_size=.01)
            # clust = OPTICS(metric=DTWDistance, min_samples=7, xi=.01, min_cluster_size=.01)

        else:
            clust = OPTICS(metric=METRIC_ARG, min_samples=4, cluster_method='xi', xi=.01, min_cluster_size=.01)
        # Run the fit
        clust.fit(X)

        reachability = clust.reachability_[clust.ordering_]
        labels = clust.labels_[clust.ordering_]

        # Options to plot clusters or not
        # plot_current_cluster(X, clust)

        nclusters = len(list(np.unique(labels)))
        n_noise_ = list(labels).count(-1)
        total_number_of_store = len(labels)
        percent_of_noise = percentage(n_noise_, total_number_of_store, 2)
        # print('================= RESULTS ========================')
        # print('labels                 : \n {}'.format(labels))
        # print('Number of the clusters : {}'.format(nclusters))
        # print('Number of noise points : {}'.format(n_noise_))
        # print('Percent_of_noise       : {}'.format(percent_of_noise))

        if (nclusters > OPTICS_MIN_CLUSTERS_ARG) & (percent_of_noise < OPTICS_MAX_NOISE_PERCENT_ARG):
        # if True:

            df_imputation_optics_arg.loc[imputation_optics_arg_index] = [df_imputation_optics.iloc[i]['X_first_column']] + [df_imputation_optics.iloc[i]['X']]\
                        + [df_imputation_optics.iloc[i]['ALGORITHMS_ARG']] + [df_imputation_optics.iloc[i]['RES_DATASET_ARG']]\
                        + [df_imputation_optics.iloc[i]['SPLIT_FIRST_BY_ARG']] + [df_imputation_optics.iloc[i]['RESAMPLING_METHOD_ARG']]\
                        + [df_imputation_optics.iloc[i]['IMPUTATION_METHOD_ARG']] + [df_imputation_optics.iloc[i]['MAX_MISSING_PERCENTAGE_ARG']]\
                        + [METRIC_ARG] + [MIN_SAMPLES_ARG] + [XI_ARG] + [MIN_CLUSTER_SIZE_ARG]\
                        + [OPTICS_MIN_CLUSTERS_ARG] + [OPTICS_MAX_NOISE_PERCENT_ARG]\
                        + [labels] + [reachability] + [nclusters] + [n_noise_] + [percent_of_noise]


                        # + [labels] + [reachability] + [labels_050] + [labels_200]
        imputation_optics_arg_index = imputation_optics_arg_index + 1

    df_imputation_optics_arg = df_imputation_optics_arg.reset_index(drop=True)
    # print("Dataframe after imputation and optics clustering - df_imputation_optics_arg: \n", df_imputation_optics_arg)
    return df_imputation_optics_arg

