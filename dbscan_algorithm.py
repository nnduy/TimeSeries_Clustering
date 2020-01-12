#! /usr/bin/env python
#coding=utf-8

import pandas as pd
from math import floor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import DBSCAN
from OPTICS.optics import *
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

        parameter_range = np.arange(EPSILON_MIN_ARG, EPSILON_MAX_ARG, EPSILON_STEP_ARG)
        actual_parameters = []
        cluster_label_matrix = np.empty(shape = (nrows, len(parameter_range)), dtype=int)
        for ind, eps in enumerate(parameter_range):
            # Retest this epsilon value accumulation
            actual_parameters.append(eps)

            dbs = DBSCAN(eps=eps, metric=METRIC_ARG, min_samples=MINS_ARG).fit(pairwise_distance_matrix)
            labels = dbs.labels_
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
            if (nclusters > 2) & (percent_of_noise<60):
            # if True:
                print('================= RESULTS ========================')
                print('cluster_labels index   : {}'.format(ind))
                print('eps                    : {}'.format(eps))
                print('labels                 : \n {}'.format(labels))
                print('Number of the clusters : {}'.format(nclusters))
                print('Number of noise points : {}'.format(n_noise_))
                print('Percent_of_noise       : {}'.format(percent_of_noise))

                df_imputation_dbscan_arg.loc[imputation_dbscan_arg_index] = [df_imputation_dbscan.iloc[i]['X_first_column']] + [df_imputation_dbscan.iloc[i]['X']]\
                    + [df_imputation_dbscan.iloc[i]['ALGORITHMS_ARG']] + [df_imputation_dbscan.iloc[i]['RES_DATASET_ARG']]\
                    + [df_imputation_dbscan.iloc[i]['SPLIT_FIRST_BY_ARG']] + [df_imputation_dbscan.iloc[i]['RESAMPLING_METHOD_ARG']]\
                    + [df_imputation_dbscan.iloc[i]['IMPUTATION_METHOD_ARG']] + [df_imputation_dbscan.iloc[i]['MAX_MISSING_PERCENTAGE_ARG']]\
                    + [METRIC_ARG] + [EPSILON_MIN_ARG] + [EPSILON_MAX_ARG] + [EPSILON_STEP_ARG] + [MINS_ARG]\
                    + [eps] + [labels] + [nclusters] + [n_noise_] + [percent_of_noise]
                imputation_dbscan_arg_index = imputation_dbscan_arg_index + 1

                print('================= RESULTS ========================')
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
