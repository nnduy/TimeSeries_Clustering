#! /usr/bin/env python
#coding=utf-8

import glob, re
import numpy as np
import pandas as pd
from datetime import date
import itertools
# from pandas import Timestamp
from sklearn import *
from datetime import datetime
# from xgboost import XGBRegressor
import math
import statistics
import kmedoids
import random
from math import floor
import matplotlib.pyplot as plt
# from .visualize_input import *
import matplotlib.patches as mpatches
from pylab import rcParams
from sklearn.preprocessing import Normalizer

# import sklearn
from sklearn.cluster import DBSCAN
# from sklearn.cluster import OPTICS, cluster_optics_dbscan
from OPTICS.optics import *

from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import gmean
from scipy.stats.mstats import gmean
# from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.metrics import pairwise_distances

def percentage(part, whole, digits):
    val = float(part)/float(whole)
    val *= 10 ** (digits + 2)
    return (floor(val) / 10 ** digits)


# # ============ Step 05: Clustering using DBSCAN and Pairwise Similarity Evaluation ==============
#
# visitor_matrix = visitor_df.values
# original_visitor_df = visitor_df
# print('visitor_matrix X :\n {} \n'.format(visitor_matrix))
#
# # Transpose the matrix for timeseries as row and timestamps as column
# visitor_matrix_transposed = visitor_matrix.transpose()
# print('visitor_matrix after transposed :\n {} \n'.format(visitor_matrix))
#
#
#
# Valid values for metric are:
# From scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]. These metrics support sparse matrix inputs.
# From scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’,
# ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]

# pairwise_manhattan = pairwise_distances(visitor_matrix_transposed, metric='manhattan')
# print('pairwise_distances manhattan:\n {} \n'.format(pairwise_manhattan))
#
# pairwise_euclidean = pairwise_distances(visitor_matrix_transposed, metric='euclidean')
# print('pairwise_distances euclidean:\n {} \n'.format(pairwise_euclidean))
#
# similarities = cosine_similarity(visitor_matrix_transposed)
# print('pairwise cosine_similarity:\n {}\n'.format(similarities))

# # pairwise_DTW = pairwise_distances(visitor_matrix_transposed, metric='euclidean')
# # print('pairwise_distances euclidean:\n {} \n'.format(pairwise_euclidean))

# # #############################################################################
# # Perform DBSCAN clustering from vector array or distance matrix.
# # input:
# #       eps: float - The maximum distance between two samples for one to be considered as in the neighborhood of the other.
# #       distance_measure: string - The metric to use when calculating distance between instances in a feature array.
# #       min_samples: int - The number of samples (or total weight) in a neighborhood for a point to be considered
# #           as a core point. This includes the point itself.
# def cluster_dbscan(matrix, distance_measure, eps, minS):
#     dbs = DBSCAN(eps=eps, metric=distance_measure, min_samples=minS).fit(matrix)
#     cluster_labels = dbs.labels_
#     core_samples_mask = np.zeros_like(dbs.labels_, dtype=bool)
#     core_samples_mask[dbs.core_sample_indices_] = True
#     return cluster_labels, core_samples_mask

# input:
#       distance_measure: string - The metric to use when calculating distance between instances in a feature array.
def run_cluster(X, distance_measure, param_min, param_max, param_step, minS):
    nrows = X.shape[0]
    if nrows <= 1:
        raise ValueError("Time-series matrix contains no information. " \
                         "Was all of your data filtered out?")
    prev_nclusters = 0
    break_out = False
    parameter_range = np.arange(param_min, param_max, param_step)
    actual_parameters = []
    cluster_label_matrix = np.empty(shape = (nrows, len(parameter_range)), dtype=int)
    for ind, eps in enumerate(parameter_range):
        actual_parameters.append(eps)

        dbs = DBSCAN(eps=eps, metric=distance_measure, min_samples=minS).fit(X)
        labels = dbs.labels_
        core_samples_mask = np.zeros_like(dbs.labels_, dtype=bool)
        core_samples_mask[dbs.core_sample_indices_] = True

        nclusters = len(list(np.unique(labels)))
        n_noise_ = list(labels).count(-1)
        total_number_of_store = len(labels)
        percent_of_noise = percentage(n_noise_, total_number_of_store, 2)
        cluster_label_matrix[:, ind] = labels
        if nclusters > 1:
            break_out = True
        # prev_nclusters != nclusters: Choose only one number of cluster - (prev_nclusters != nclusters) &
        # nclusters > 2: Number of clusters must be greater than 2
        # percent_of_noise<10: percent of noise must be less than 10 percent
        if (nclusters > 2) & (percent_of_noise<10):
            print('================= RESULTS ========================')
            print('Number of the clusters : {}'.format(nclusters))
            print('Number of noise points : {}'.format(n_noise_))
            print('Percent_of_noise       : {}'.format(percent_of_noise))
            print('cluster_labels index   : {}'.format(ind))
            print('cluster_labels eps     : {}'.format(eps))
            print('Noise points           : {}'.format(n_noise_))
            print('cluster_labels list : \n {}'.format(labels))
            print('================= RESULTS ========================')
        if (prev_nclusters == 1) & (nclusters == 1) & break_out:
          param_max = eps
          break
        else:
          prev_nclusters = nclusters
    # Print out the clusters with their sequence IDs
    # print('cluster_label_matrix:\n {} \n'.format(cluster_label_matrix))
    for i in range(0, cluster_label_matrix.shape[0]):
        encoded_labels = [ str(x).encode() for x \
                in cluster_label_matrix[i, 0:len(actual_parameters)] ]
    return labels, nclusters, core_samples_mask

def clustering_by_dbscan(X, METRIC_ARG, EPSILON_MIN_ARG, EPSILON_MAX_ARG, EPSILON_STEP_ARG, MINS_ARG):
    print("X:\n", X)

    pairwise_distance_matrix = pairwise_distances(X, metric=METRIC_ARG)
    print("pairwise_distance_matrix:\n", pairwise_distance_matrix)

    # This function need and input as a pairwise distance matrix. This is a must.
    labels, nclusters, core_samples_mask = run_cluster(pairwise_distance_matrix, METRIC_ARG,
                                                       param_min = EPSILON_MIN_ARG, param_max = EPSILON_MAX_ARG,
                                                       param_step = EPSILON_STEP_ARG, minS = MINS_ARG)
    # print("Labels:\n", labels)
    # print("nclusters:\n", nclusters)
    # print("core_samples_mask:\n", core_samples_mask)
    # check_central_majority(df_3genres_3locations, labels)

    # plot_all_ts(X, labels)
    # plot_each_group_ts(X, labels, core_samples_mask)

def check_central_majority(df_genre_location, labels):
    df = df_genre_location
    nclusters = len(list(np.unique(labels)))
    unique_labels = set(labels)
    print("Type of nclusters    :", type(nclusters))
    print("Type of unique_labels:", type(unique_labels))
    print("Type   of labels:", type(labels))
    print("Length of labels:", len(labels))
    df = df.copy()
    print("df dataframe:\n", df)
    df.loc[:,'labels'] = labels
    # print('df_genre_location        :\n {} \n'.format(df))
    # df.loc[df.air_area_name == '', 'make'] = df.air_area_name.str.split().str.get(0)
    df['first_word'] = df['air_area_name'].str.split().str[0]
    # print('df_genre_location        :\n {} \n'.format(df))
    # df.iloc[0:df[df.labels == '0'].index[0]]
    # For each k-th cluster in a set of clusters
    for k in unique_labels:
        df_label = df.loc[df['labels'] == k]
        print('df_genre_location new column labels :\n {} \n'.format(df_label))
        df_groups = df_label.groupby('first_word').count().sort_values('first_word', ascending=False)
        # print('df_groups first_word:\n {} \n'.format(df_groups))
        df_groups['perc']= df_groups['labels']/df_groups['labels'].sum()
        print('df_groups group_percent:\n {} \n'.format(df_groups))
    return True

def plot_each_group_ts(X, labels, core_samples_mask):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    # print('core_samples_mask : {}'.format(core_samples_mask))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        # print('k             : {}'.format(k))
        # print('unique_labels : {}'.format(unique_labels))
        # print('class_member_mask : {}'.format(class_member_mask))
        # print('length of class_member_mask : {}'.format(len(class_member_mask)))
        # print('length of core_samples_mask : {}'.format(len(core_samples_mask)))
        # print('length of X : {}'.format(len(X.columns)))
        # print("Type of X:", type(X))

        current_group_index = X.columns[class_member_mask]
        # print("Type of current_group_index:", type(current_group_index))
        # print("current_group_index:", current_group_index)

        current_group_columns = X[current_group_index]
        # print("Type of current_group_columns:", type(current_group_columns))
        # print('current_group_columns: {}'.format(current_group_columns.head()))
        current_group_columns.plot(legend=False, color=colors)

        color_list_patch = []
        for k, col in zip(unique_labels, colors):
            if k == -1:
                color_list_patch.append(mpatches.Patch(color=col, label='Group Noise'))
            else:
                color_list_patch.append(mpatches.Patch(color=col, label='Group ' + str(k)))
        plt.legend(handles=color_list_patch)
        plt.show()

def plot_all_ts(X, labels):
    print('labels        :\n {} \n'.format(labels))
    unique_labels = set(labels)
    print('unique_labels : {}'.format(unique_labels))
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    # print('colors        : {}'.format(colors))
    X.plot(legend=False, color=colors)
    color_list_patch = []
    for k, col in zip(unique_labels, colors):
        if k == -1:
            color_list_patch.append(mpatches.Patch(color=col, label='Group Noise'))
        else:
            color_list_patch.append(mpatches.Patch(color=col, label='Group ' + str(k)))

    plt.legend(handles=color_list_patch)
    plt.show()
