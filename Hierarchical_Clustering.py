# Hierarchical Clustering Model: 

#1 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# read in all csv file to data variable
data = {
    'avd': pd.read_csv('input/air_visit_data.csv', parse_dates=['visit_date']),
    'asi' : pd.read_csv('input/air_store_info.csv').dropna(),
    'hsi' : pd.read_csv('input/hpg_store_info.csv'),
    'ar' : pd.read_csv('input/air_reserve.csv', parse_dates=['visit_datetime', 'reserve_datetime']),
    'hr' : pd.read_csv('input/hpg_reserve.csv', parse_dates=['visit_datetime', 'reserve_datetime']),
    'sidr' : pd.read_csv('input/store_id_relation.csv'),
    'tes': pd.read_csv('input/sample_submission.csv'),
    'hol': pd.read_csv('input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }

visitor_matrix = pd.read_csv('output_visitor.csv')

RES_DATASET = ['air', 'hpg', 'air_hpg']
print(RES_DATASET[0])

#2 Importing the Restaurant store id dataset by pandas
def df_dendogram(res, data, visitor_matrix):

    print("visitor_matrix 1111:\n", visitor_matrix)
    # visitor_matrix = visitor_matrix.reset_index(inplace = True)


    df_id = visitor_matrix['Unnamed: 0']
    print("visitor_matrix 2222:\n", df_id)

    df_id = df_id.to_frame()
    df_id.rename(columns={'Unnamed: 0':'store_id'}, inplace=True)
    print("df_id:\n", df_id)
    del visitor_matrix['Unnamed: 0']


    X = visitor_matrix.iloc[:, :].values
    print("X ------------:\n", X)

    if res == "air":
        df_store_id = data['asi']
        df_store_id = df_store_id.drop(['latitude', 'longitude'], axis = 1)
        df_store_id.rename(columns={'air_store_id':'store_id', 'air_genre_name':'genre_name', 'air_area_name':'area_name'}, inplace = True)
        print("df_store_id:\n", df_store_id)
        df_merge_id = pd.merge(df_store_id, df_id, how='inner', on=['store_id'])
    elif res =="hpg":
        df_store_id = data['hsi']
        df_store_id = df_store_id.drop(['latitude', 'longitude'], axis = 1)
        df_store_id.rename(columns={'hpg_store_id':'store_id', 'hpg_genre_name':'genre_name', 'hpg_area_name':'area_name'}, inplace = True)
        print("df_store_id:\n", df_store_id)
        df_merge_id = pd.merge(df_store_id, df_id, how='inner', on=['store_id'])
    else:
        print("Not sufficient restaurant store chain")

    print("df_merge_id head:\n", df_merge_id.head())
    return X, df_merge_id

dataset_ts, df_merge_id = df_dendogram("hpg", data, visitor_matrix)


def hierachy_clustering(dataset_ts_arg, affinity_arg, linkage_arg):

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

    hc = AgglomerativeClustering(n_clusters = 3, affinity = affinity_arg, linkage =linkage_arg)

    # Lets try to fit the hierarchical clustering algorithm  to dataset X while creating the
    # clusters vector that tells for each customer which cluster the customer belongs to.
    labels_hc=hc.fit_predict(dataset_ts_arg)
    print("labels_hc:", labels_hc)
    print("dataset_ts_arg:\n", dataset_ts_arg)

    # print("labels_hc type:", type(labels_hc))
    return labels_hc

def get_main_label(df, lables):
    df['hc'] = lables.tolist()
    print("df_merge_id:111\n", df.head())
    # df_merge_id = df_merge_id.groupby(["air_genre_name"]).count()
    # df_merge_id= df_merge_id.groupby('air_genre_name').air_genre_name.count()

    df = df.groupby(['genre_name', 'hc']).size()
    df = df.to_frame(name = 'size').reset_index()
    print("df_merge_id:222\n", df)

    idx = df.groupby(['genre_name'])['size'].transform(max) == df['size']
    df = df[idx]
    print("df_merge_id:333\n", df)
    df = df.groupby('genre_name')['hc'].apply(lambda x: ','.join(map(str, x))).reset_index()
    # print("df_merge_id:", df)
    df = df[['hc']]
    # print("df_merge_id:", df)
    return df

# If linkage is “ward”, only “euclidean” is accepted. If “precomputed”,
# a distance matrix (instead of a similarity matrix) is needed as input for the fit method.
# linkage_list = ['ward']
linkage_list = ['ward', 'complete', 'average', 'single']

temp_df = df_merge_id
print("temp_df:=====\n", temp_df)

# df_genre_clusters = temp_df.groupby(['air_genre_name']).size().reset_index()
# # print("list(my_dataframe.columns.values):", list(df_genre_clusters.columns.values))
# print("temp_df_first_col:==============\n", df_genre_clusters.head())

for lnk in linkage_list:
    if lnk=='ward':
        affinity_list = ['euclidean']
    else:
        affinity_list = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']

    for aff in affinity_list:
        lables = hierachy_clustering(dataset_ts, aff, lnk)
        main_labels_df = get_main_label(temp_df, lables)
        # df_genre_clusters['hc_' + lnk + '_' + aff] = main_labels_df



# print("df_genre_clusters:", df_genre_clusters.head())
# df_genre_clusters.to_csv('df_genre_clusters.csv')

# 5 Visualizing the clusters. This code is similar to k-means visualization code.
# We only replace the y_kmeans vector name to y_hc for the hierarchical clustering
#
# plt.scatter(dataset_ts[y_hc==0, 0], dataset_ts[y_hc==0, 1], s=100, c='red', label ='Cluster 1')
# plt.scatter(dataset_ts[y_hc==1, 0], dataset_ts[y_hc==1, 1], s=100, c='blue', label ='Cluster 2')
# plt.scatter(dataset_ts[y_hc==2, 0], dataset_ts[y_hc==2, 1], s=100, c='green', label ='Cluster 3')
# plt.title('Clusters of Customers (Hierarchical Clustering Model)')
# plt.xlabel('Annual Income(k$)')
# plt.ylabel('Spending Score(1-100')
# plt.show()
