#! /usr/bin/env python
#coding=utf-8

from dbscan_algorithm import *
from OPTICS.optics import *
from sklearn.cluster import AgglomerativeClustering
import pandas as pd

# This option help to print out all data columns of a dataframe
pd.set_option('display.expand_frame_repr', False)

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
    'hol': pd.read_csv('input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }


df_ar = data['ar']
df_sidr = data['sidr']
df_asi = data['asi']
df_avd = data['avd']
df_hsi = data['hsi']
df_hr = data['hr']

genre_name = 'genre_name'
area_name = 'area_name'
store_id = 'store_id'

def format_arename_col_first_word(df):
    # Get first letter from area name column
    df['new_col'] = df[area_name].astype(str).str.split().str.get(0)
    df[area_name] = df['new_col']
    df.drop('new_col', axis=1, inplace=True)
    return df

# Method used: get labels from a dataset according to affinity and linkage
# input:
#   dataset_ts_arg  : dataset of time series as argument
#   affinity_arg    :
#   linkage_arg     :
# output:
#   labels_hc       : labels of hierachy cluster
def labeling_hierachy_cluster(dataset_ts_arg, affinity_arg, linkage_arg, NUM_OF_HC_CLUSTER_ARG):
    #3 Using the dendrogram to find the optimal numbers of clusters.
    # First thing we're going to do is to import scipy library. scipy is an open source
    # Python library that contains tools to do hierarchical clustering and building dendrograms.

    #Lets create a dendrogram variable
    # linkage is actually the algorithm itself of hierarchical clustering and then in
    #linkage we have to specify on which data we apply and engage. This is X dataset

    # dendrogram = sch.dendrogram(sch.linkage(dataset_ts, method = "ward"))
    # plt.title('Dendrogram')
    # plt.xlabel('Series')
    # plt.ylabel('Euclidean distances')
    # plt.show()

    #4 Fitting hierarchical clustering to the Mall_Customes dataset
    # There are two algorithms for hierarchical clustering: Agglomerative Hierarchical Clustering and
    # Divisive Hierarchical Clustering. We choose Euclidean distance and ward method for our
    # algorithm class
    hc = AgglomerativeClustering(n_clusters=NUM_OF_HC_CLUSTER_ARG, affinity=affinity_arg, linkage=linkage_arg)

    # Lets try to fit the hierarchical clustering algorithm  to dataset X while creating the
    # clusters vector that tells for each customer which cluster the customer belongs to.
    # print("type of one element in dataset_ts_arg:", dataset_ts_arg[0][0])
    # print("type of one element in dataset_ts_arg:", type(dataset_ts_arg[0][0]))
    labels_hc = hc.fit_predict(dataset_ts_arg)
    # print("labels_hc:\n", labels_hc)
    # print("labels_hc type:", type(labels_hc))
    return labels_hc

# Method used: Clustering by hierachy method
# input:
#   vm_values : visitor matrix after formatted and get values
#   df_merge  :
# output:
#
def clustering_by_hierachy(df_imputation_hierachy):
    imputation_hierachy_arg_index = 0
    list_cols = list(df_imputation_hierachy.columns.values)
    list_cols.extend(['labels', 'nclusters', 'n_noise_', 'percent_of_noise'])
    df_imputation_hierachy_arg = pd.DataFrame(columns=list_cols)

    for i, row in df_imputation_hierachy.iterrows():
        prev_nclusters = 0
        break_out = False
        X = df_imputation_hierachy.iloc[i]['X']
        NUM_OF_HC_CLUSTER_ARG = df_imputation_hierachy.iloc[i]['NUM_OF_HC_CLUSTER']
        AFFINITY_ARG = df_imputation_hierachy.iloc[i]['AFFINITY']
        LINKAGE_ARG = df_imputation_hierachy.iloc[i]['LINKAGE']

        # Running clustering and get labels list
        labels = labeling_hierachy_cluster(X, AFFINITY_ARG, LINKAGE_ARG, NUM_OF_HC_CLUSTER_ARG)

        # Calculating some metrics for further visualization
        nclusters = len(list(np.unique(labels)))
        n_noise_ = list(labels).count(-1)
        total_number_of_store = len(labels)
        percent_of_noise = percentage(n_noise_, total_number_of_store, 2)
        if nclusters > 1:
            break_out = True
        # prev_nclusters != nclusters: Choose only one number of cluster - (prev_nclusters != nclusters) &
        # nclusters > 2: Number of clusters must be greater than 2
        # percent_of_noise<10: percent of noise must be less than 10 percent
        if (nclusters > 2) & (percent_of_noise<60):
        # if True:
            print('================= RESULTS ========================')
            print('NUM_OF_HC_CLUSTER_ARG  : {}'.format(NUM_OF_HC_CLUSTER_ARG))
            print('AFFINITY_ARG           : {}'.format(AFFINITY_ARG))
            print('LINKAGE_ARG            : {}'.format(LINKAGE_ARG))
            print('labels                 : \n {}'.format(labels))
            print('Number of the clusters : {}'.format(nclusters))
            print('Number of noise points : {}'.format(n_noise_))
            print('Percent_of_noise       : {}'.format(percent_of_noise))

            df_imputation_hierachy_arg.loc[imputation_hierachy_arg_index] = [df_imputation_hierachy.iloc[i]['X_first_column']] + [df_imputation_hierachy.iloc[i]['X']]\
                + [df_imputation_hierachy.iloc[i]['ALGORITHMS_ARG']] + [df_imputation_hierachy.iloc[i]['RES_DATASET_ARG']]\
                + [df_imputation_hierachy.iloc[i]['SPLIT_FIRST_BY_ARG']] + [df_imputation_hierachy.iloc[i]['RESAMPLING_METHOD_ARG']]\
                + [df_imputation_hierachy.iloc[i]['IMPUTATION_METHOD_ARG']] + [df_imputation_hierachy.iloc[i]['MAX_MISSING_PERCENTAGE_ARG']]\
                + [NUM_OF_HC_CLUSTER_ARG] + [AFFINITY_ARG] + [LINKAGE_ARG]\
                + [labels] + [nclusters] + [n_noise_] + [percent_of_noise]
            imputation_hierachy_arg_index = imputation_hierachy_arg_index + 1

        if (prev_nclusters == 1) & (nclusters == 1) & break_out:
          break
        else:
          prev_nclusters = nclusters
    print("Dataframe after imputation and hierachy clustering - df_imputation_hierachy_arg: \n", df_imputation_hierachy_arg)
    # df_imputation_hierachy_arg.to_csv('df_imputation_hierachy_arg.csv')
    return df_imputation_hierachy_arg
