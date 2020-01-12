#! /usr/bin/env python
#coding=utf-8

import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import sys

# read in all csv file to data variable
data = {
    'avd': pd.read_csv('input/air_visit_data.csv', parse_dates=['visit_date']).rename(columns={'air_store_id':'store_id'}),
    'asi' : pd.read_csv('input/air_store_info.csv').dropna()
        .rename(columns={'air_store_id':'store_id', 'air_genre_name':'genre_name', 'air_area_name':'area_name'}),
    'ar' : pd.read_csv('input/air_reserve.csv', parse_dates=['visit_datetime', 'reserve_datetime']),

    'hr' : pd.read_csv('input/hpg_reserve.csv', parse_dates=['visit_datetime', 'reserve_datetime']),
    'hsi' : pd.read_csv('input/hpg_store_info.csv')
        .rename(columns={'hpg_store_id':'store_id', 'hpg_genre_name':'genre_name', 'hpg_area_name':'area_name'}),

    'sidr' : pd.read_csv('input/store_id_relation.csv'),
    'tes': pd.read_csv('input/sample_submission.csv'),
    'hol': pd.read_csv('input/date_info.csv').rename(columns={'calendar_date':'visit_date'}),
    # 'test': pd.read_csv('df_imputation_dbscan_arg.csv')
    }

df_ar = data['ar']
df_sidr = data['sidr']
df_asi = data['asi']
df_avd = data['avd']
df_hsi = data['hsi']
df_hr = data['hr']
# df_test = data['test']

genre_name = 'genre_name'
area_name = 'area_name'
store_id = 'store_id'


# ============ Step 06: Kmeans clustering using Pairwise Similarity Evaluation DWT ==============
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
def DTWDistance(s1, s2, w):
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

#  LB Keogh lower bound of dynamic time warping to increase speed
def LB_Keogh(s1, s2, r):
    # print("s1 s1:", s1)
    # print("s2 s2:", s2)
    # print("r r:", r)
    LB_sum=0
    for ind,i in enumerate(s1):

        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2

    return math.sqrt(LB_sum)

# Method to convert dictionary object to labels list object
# input:
#   assignments: dictionary object
# output:
#   labels_list: list object item
def convert_assigments_to_labels(assignments):
    # count is the number of items in dictionary
    count = 0
    for key, value in assignments.items():
        if isinstance(value, list):
              count += len(value)

    # create a new array as lables
    labels_list = np.empty([count], dtype=int)
    for key in assignments:
        for k in assignments[key]:
            labels_list[k] = key
    labels_list.tolist()
    print(labels_list)
    return labels_list


#  k-means clustering
def k_means_clust(data, num_clust, num_iter, window_size):
    print("data:\n", data)
    # print("type of data:", type(data))
    # centroids is the random members of the data
    centroids = random.sample(list(data), num_clust)
    # print("centroids before loop:", centroids)
    centroids_backup = centroids
    # counter is number of iteration, which we use to recalculate the centroids
    counter = 0
    for n in range(num_iter):
        counter += 1
        # print("counter:", counter)
        # assignments is the list of clusters and it's members
        assignments = {}
        #assign data points to clusters
        for ind, i in enumerate(data):
            # print("data i before calculate distance:", i)
            min_dist = float('inf')
            closest_clust = None
            for c_ind, j in enumerate(centroids):
                # print("i values:", i)
                # print("centroids inside loop:", centroids)
                # print("data j before calculate distance:", j)
                # print("LB_Keogh(i, j, 5):", LB_Keogh(i, j, 5))
                # print("closest_clust    :", closest_clust)
                if LB_Keogh(i, j, 5) < min_dist:
                    cur_dist = DTWDistance(i, j, window_size)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_ind
            #     print("c_ind         :", c_ind)
            # print("closest_clust         :", closest_clust)
            # print("assignments:", assignments)
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust] = []
                assignments[closest_clust].append(ind)
        # assignments is a set of labels and their member timeseries
        # print("assignments:", assignments)
        # print("==============================")

        #recalculate centroids of clusters
        for key in assignments:
            # print("assignments:", assignments)
            # print("key:", key)
            clust_sum = 0
            for k in assignments[key]:
                # clust_sum = clust_sum + data[k]
                clust_sum = np.sum([clust_sum, data[k]], axis=0)
                # clust_sum = np.vstack((clust_sum, data[k]))
                # print("assignments[", key, "]:", assignments[key])
                # print("data[", k, "]         :", data[k])
                # print("clust_sum[", k, "]    :", clust_sum)
                # for m in list(clust_sum):
                #     print("m value:", m)
            # centroids[key]=[m/len(assignments[key]) for m in list(clust_sum)]
            divisor = len(assignments[key])
            centroids[key] = np.divide(clust_sum, divisor)
            np.rint(centroids[key])
            # print("centroids[", key, "]:", centroids[key])

    lables = convert_assigments_to_labels(assignments)
    return centroids, lables

# Method: clustering by kmeans and DTW distances
# input:
#   df_imputation_kmeans: input dataframe of kmeans DTW arguments
# output:
#   df_imputation_kmeans_arg
def clustering_by_kmeans(df_imputation_kmeans):
    imputation_kmeans_arg_index = 0
    list_cols = list(df_imputation_kmeans.columns.values)
    list_cols.extend(['labels', 'centroids'])
    df_imputation_kmeans_arg = pd.DataFrame(columns=list_cols)
    # print("df_imputation_kmeans df_imputation_kmeans:\n", df_imputation_kmeans)

    for i, row in df_imputation_kmeans.iterrows():
        X = df_imputation_kmeans.iloc[i]['X']
        NUM_CLUSTERS_ARG = df_imputation_kmeans.iloc[i]['NUM_CLUSTERS_ARG']
        ITERATIONS_ARG = df_imputation_kmeans.iloc[i]['ITERATIONS_ARG']
        WINDOW_SIZE_ARG = df_imputation_kmeans.iloc[i]['WINDOW_SIZE_ARG']

        # Running clustering and get labels list
        # labels = labeling_hierachy_cluster(X, AFFINITY_ARG, LINKAGE_ARG, NUM_OF_HC_CLUSTER_ARG)
        centroids, labels = k_means_clust(X, NUM_CLUSTERS_ARG, ITERATIONS_ARG, WINDOW_SIZE_ARG)
        # print("labels        :", labels)
        # print("centroids list:", centroids)
        # print("X shape:", X.shape)

        print('================= RESULTS ========================')
        print('NUM_CLUSTERS_ARG   : {}'.format(NUM_CLUSTERS_ARG))
        print('ITERATIONS_ARG     : {}'.format(ITERATIONS_ARG))
        print('WINDOW_SIZE_ARG    : {}'.format(WINDOW_SIZE_ARG))
        # print('labels           : \n {}'.format(labels))

        df_imputation_kmeans_arg.loc[imputation_kmeans_arg_index] = [df_imputation_kmeans.iloc[i]['X_first_column']] + [df_imputation_kmeans.iloc[i]['X']]\
            + [df_imputation_kmeans.iloc[i]['ALGORITHMS_ARG']] + [df_imputation_kmeans.iloc[i]['RES_DATASET_ARG']]\
            + [df_imputation_kmeans.iloc[i]['SPLIT_FIRST_BY_ARG']] + [df_imputation_kmeans.iloc[i]['RESAMPLING_METHOD_ARG']]\
            + [df_imputation_kmeans.iloc[i]['IMPUTATION_METHOD_ARG']] + [df_imputation_kmeans.iloc[i]['MAX_MISSING_PERCENTAGE_ARG']]\
            + [NUM_CLUSTERS_ARG] + [ITERATIONS_ARG] + [WINDOW_SIZE_ARG]\
            + [labels] + [centroids]
        imputation_kmeans_arg_index = imputation_kmeans_arg_index + 1

    # print("Dataframe after imputation and kmeans clustering - df_imputation_kmeans_arg: \n", df_imputation_kmeans_arg)
    # df_imputation_kmeans_arg.to_csv('df_imputation_kmeans_arg.csv')

    return df_imputation_kmeans_arg


# Method: clustering by kmeans and auto distances by calling KMeans built-in algorithm
# input:
#   data:
#   num_clust:
#   num_iter:
# output:
#   df_imputation_kmeans_arg
def k_means_clust_auto(data, num_clust, num_iter):
    print("data:\n", data)
    Kmean = KMeans(n_clusters=num_clust, max_iter=num_iter)
    Kmean.fit(data)
    return Kmean.cluster_centers_, Kmean.labels_

# Method: clustering by kmeans and DTW distances
# input:
#   df_imputation_kmeans: input dataframe of kmeans auto arguments
# output:
#   df_imputation_kmeans_arg
def clustering_by_kmeans_auto(df_imputation_kmeans_auto):
    imputation_kmeans_arg_index = 0
    list_cols = list(df_imputation_kmeans_auto.columns.values)
    list_cols.extend(['labels', 'centroids'])
    df_imputation_kmeans_arg = pd.DataFrame(columns=list_cols)

    for i, row in df_imputation_kmeans_auto.iterrows():
        X = df_imputation_kmeans_auto.iloc[i]['X']
        NUM_CLUSTERS_ARG = df_imputation_kmeans_auto.iloc[i]['NUM_CLUSTERS_ARG']
        ITERATIONS_ARG = df_imputation_kmeans_auto.iloc[i]['ITERATIONS_ARG']

        # Running clustering and get labels list
        centroids, labels = k_means_clust_auto(X, NUM_CLUSTERS_ARG, ITERATIONS_ARG)
        print("labels        :", labels)
        print("centroids list:", centroids)
        print("X shape:", X.shape)

        print('================= RESULTS ========================')
        print('NUM_CLUSTERS_ARG   : {}'.format(NUM_CLUSTERS_ARG))
        print('ITERATIONS_ARG     : {}'.format(ITERATIONS_ARG))
        # print('labels           : \n {}'.format(labels))

        df_imputation_kmeans_arg.loc[imputation_kmeans_arg_index] = [df_imputation_kmeans_auto.iloc[i]['X_first_column']]\
            + [df_imputation_kmeans_auto.iloc[i]['X']]\
            + [df_imputation_kmeans_auto.iloc[i]['ALGORITHMS_ARG']] + [df_imputation_kmeans_auto.iloc[i]['RES_DATASET_ARG']]\
            + [df_imputation_kmeans_auto.iloc[i]['SPLIT_FIRST_BY_ARG']] + [df_imputation_kmeans_auto.iloc[i]['RESAMPLING_METHOD_ARG']]\
            + [df_imputation_kmeans_auto.iloc[i]['IMPUTATION_METHOD_ARG']] + [df_imputation_kmeans_auto.iloc[i]['MAX_MISSING_PERCENTAGE_ARG']]\
            + [NUM_CLUSTERS_ARG] + [ITERATIONS_ARG]\
            + [labels] + [centroids]
        imputation_kmeans_arg_index = imputation_kmeans_arg_index + 1

    print("Dataframe after imputation and kmeans clustering - df_imputation_kmeans_arg: \n", df_imputation_kmeans_arg)
    # df_imputation_kmeans_arg.to_csv('df_imputation_kmeans_arg.csv')
    # sys.exit()
    return df_imputation_kmeans_arg
