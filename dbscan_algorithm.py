#! /usr/bin/env python
#coding=utf-8

import math
import pandas as pd
from math import floor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import DBSCAN
from OPTICS.optics import *
from sklearn.metrics import pairwise_distances
import sys

def percentage(part, whole, digits):
    val = float(part)/float(whole)
    val *= 10 ** (digits + 2)
    return (floor(val) / 10 ** digits)

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


# #############################################################################
# Perform DBSCAN clustering from vector array or distance matrix.
# input:
#       eps: float - The maximum distance between two samples for one to be considered as in the neighborhood of the other.
#       distance_measure: string - The metric to use when calculating distance between instances in a feature array.
#       min_samples: int - The number of samples (or total weight) in a neighborhood for a point to be considered
#           as a core point. This includes the point itself.
# input:
#       distance_measure: string - The metric to use when calculating distance between instances in a feature array.
# def clustering_by_dbscan(X, METRIC_ARG, EPSILON_MIN_ARG, EPSILON_MAX_ARG, EPSILON_STEP_ARG, MINS_ARG):
def clustering_by_dbscan(df_imputation_dbscan):
    # result_list = clustering_by_dbscan(df_imputation_dbscan.iloc[imputation_dbscan_index]['X'], METRIC_ARG,
                #                                    EPSILON_MIN_ARG, EPSILON_MAX_ARG, EPSILON_STEP_ARG, MINS_ARG)
    # print("df_imputation_dbscan input clustering_by_dbscan:\n", df_imputation_dbscan)
    # Creating index for new dataframe
    imputation_dbscan_arg_index = 0
    # print('type of imputation_dbscan_arg_index: {}', type(imputation_dbscan_arg_index))
    list_cols = list(df_imputation_dbscan.columns.values)
    list_cols.extend(['eps', 'labels', 'nclusters', 'n_noise_', 'percent_of_noise'])
    df_imputation_dbscan_arg = pd.DataFrame(columns=list_cols)

    for i, row in df_imputation_dbscan.iterrows():
        # print("X:\n", df_imputation_dbscan.iloc[i]['X'])
        X = df_imputation_dbscan.iloc[i]['X']
        pairwise_distance_matrix = pairwise_distances(X, metric=df_imputation_dbscan.iloc[i]['METRIC_ARG'])
        # print("pairwise_distance_matrix:\n", pairwise_distance_matrix)

        nrows = pairwise_distance_matrix.shape[0]
        if nrows <= 1:
            raise ValueError("Time-series matrix contains no information. " \
                             "Was all of your data filtered out?")
        prev_nclusters = 0
        break_out = False
        EPSILON_MIN_ARG = df_imputation_dbscan.iloc[i]['EPSILON_MIN_ARG']
        EPSILON_MAX_ARG = df_imputation_dbscan.iloc[i]['EPSILON_MAX_ARG']
        EPSILON_STEP_ARG = df_imputation_dbscan.iloc[i]['EPSILON_STEP_ARG']
        METRIC_ARG = df_imputation_dbscan.iloc[i]['METRIC_ARG']
        MINS_ARG = df_imputation_dbscan.iloc[i]['MINS_ARG']
        MIN_CLUSTERS_ARG = df_imputation_dbscan.iloc[i]['MIN_CLUSTERS_ARG']
        MAX_NOISE_PERCENT_ARG = df_imputation_dbscan.iloc[i]['MAX_NOISE_PERCENT_ARG']

        parameter_range = np.arange(EPSILON_MIN_ARG, EPSILON_MAX_ARG, EPSILON_STEP_ARG)
        actual_parameters = []
        cluster_label_matrix = np.empty(shape = (nrows, len(parameter_range)), dtype=int)
        for ind, eps in enumerate(parameter_range):
            # Retest this epsilon value accumulation
            actual_parameters.append(eps)

            # if METRIC_ARG == 'DTWDistance':
            #     dbs = DBSCAN(eps=eps, metric=lambda X, Y: DTWDistance(X, Y, w=5), min_samples=MINS_ARG).fit(pairwise_distance_matrix)
            #     # clust = OPTICS(metric=lambda X, Y: DTWDistance(X, Y, w=5), min_samples=4, xi=.01, min_cluster_size=.01)
            #     # clust = OPTICS(metric=DTWDistance, min_samples=7, xi=.01, min_cluster_size=.01)
            # else:
            dbs = DBSCAN(eps=eps, metric=METRIC_ARG, min_samples=MINS_ARG).fit(pairwise_distance_matrix)
            labels = dbs.labels_
            nclusters = len(list(np.unique(labels)))
            n_noise_ = list(labels).count(-1)
            total_number_of_store = len(labels)
            percent_of_noise = percentage(n_noise_, total_number_of_store, 2)
            cluster_label_matrix[:, ind] = labels

            # core_samples_mask = np.zeros_like(dbs.labels_, dtype=bool)
            # core_samples_mask[dbs.core_sample_indices_] = True

            if nclusters > 1:
                break_out = True
            # prev_nclusters != nclusters: Choose only one number of cluster - (prev_nclusters != nclusters) &
            # nclusters > 2: Number of clusters must be greater than 2
            # percent_of_noise<10: percent of noise must be less than 10 percent
            test = MIN_CLUSTERS_ARG + 1
            print("test:", test)

            if (nclusters == test) & (percent_of_noise < MAX_NOISE_PERCENT_ARG):
            # if True:
            #     print('================= RESULTS ========================')
            #     print('cluster_labels index   : {}'.format(ind))
            #     print('eps                    : {}'.format(eps))
            #     print('labels                 : \n {}'.format(labels))
            #     print('Number of the clusters : {}'.format(nclusters))
            #     print('Number of noise points : {}'.format(n_noise_))
            #     print('Percent_of_noise       : {}'.format(percent_of_noise))

                df_imputation_dbscan_arg.loc[imputation_dbscan_arg_index] = [df_imputation_dbscan.iloc[i]['X_first_column']] + [df_imputation_dbscan.iloc[i]['X']]\
                    + [df_imputation_dbscan.iloc[i]['ALGORITHMS_ARG']] + [df_imputation_dbscan.iloc[i]['RES_DATASET_ARG']]\
                    + [df_imputation_dbscan.iloc[i]['SPLIT_FIRST_BY_ARG']] + [df_imputation_dbscan.iloc[i]['RESAMPLING_METHOD_ARG']]\
                    + [df_imputation_dbscan.iloc[i]['IMPUTATION_METHOD_ARG']] + [df_imputation_dbscan.iloc[i]['MAX_MISSING_PERCENTAGE_ARG']]\
                    + [METRIC_ARG] + [EPSILON_MIN_ARG] + [EPSILON_MAX_ARG] + [EPSILON_STEP_ARG] + [MINS_ARG]\
                    + [MIN_CLUSTERS_ARG] + [MAX_NOISE_PERCENT_ARG]\
                    + [eps] + [labels] + [nclusters] + [n_noise_] + [percent_of_noise]
                imputation_dbscan_arg_index = imputation_dbscan_arg_index + 1

                # print('================= RESULTS ========================')
            if (prev_nclusters == 1) & (nclusters == 1) & break_out:
              param_max = eps
              break
            else:
              prev_nclusters = nclusters

        # for i in range(0, cluster_label_matrix.shape[0]):
        #     encoded_labels = [ str(x).encode() for x \
        #             in cluster_label_matrix[i, 0:len(actual_parameters)] ]
    print("Dataframe after imputation and dbscan clustering - df_imputation_dbscan_arg: \n", df_imputation_dbscan_arg)
    # df_imputation_dbscan_arg.to_csv('df_imputation_dbscan_arg.csv')
    # sys.exit()
    return df_imputation_dbscan_arg

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
    df['first_word'] = df['air_area_name'].str.split().str[0]
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
