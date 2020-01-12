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
from dbscan_algorithm import *
from optics_algorithm import *
from hierachy_algorithm import *
from kmeans_algorithm import *
import copy
# import sklearn
from sklearn.cluster import DBSCAN
# from sklearn.cluster import OPTICS, cluster_optics_dbscan
from OPTICS.optics import *
import sys
import uuid
pd.set_option('precision', 0)

from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import gmean
from scipy.stats.mstats import gmean
# from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.metrics import pairwise_distances

ALGORITHMS = ['KMEANS_AUTO']
ALGORITHMS_ARG = ['HIERACHY', 'DBSCAN', 'KMEANS_AUTO', 'KMEANS_DTW', 'OPTICS']
RES_DATASET = ['air']
# RES_DATASET = ['air']
RES_DATASET_ARG = 'air'
RESAMPLING_METHOD = ['over']
RESAMPLING_METHOD_ARG = 'under', 'over'
# SPLIT_GROUPS = [3, 9]
SPLIT_FIRST_BY = ['area']
SPLIT_FIRST_BY_ARG = 'area'
# split groups arguments always is 9
# SPLIT_GROUPS_ARG = 9
# IMPUTATION_METHOD = ['median', 'mean', 'linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'barycentric',
#           'krogh', 'polynomial', 'spline', 'piecewise_polynomial', 'from_derivatives', 'pchip', 'akima']
# IMPUTATION_METHOD = ['barycentric', 'krogh', 'polynomial', 'spline',] --> fail
IMPUTATION_METHOD = ['median']
# IMPUTATION_METHOD_ARG = 'median'
# MAX_MISSING_PERCENTAGE = 100 means: we take all of timeseries which have missing point
# MAX_MISSING_PERCENTAGE = 20  means: we take only first 20 percent of timeseries which have missing point
MAX_MISSING_PERCENTAGE = [75]
MAX_MISSING_PERCENTAGE_ARG = 90

# NUM_OF_HC_CLUSTER = [3, 9]
NUM_OF_HC_CLUSTER_ARG = 5

import pandas as pd

# This option help to print out all data columns of a dataframe
pd.set_option('display.expand_frame_repr', False)

nl = '\n'

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
visit_date = 'visit_date'
visitors = 'visitors'

# This function will restructure or reformat and group hpg store info. Remove unnecessary columns ['latitude', 'longitude']
def store_info_format(df):
    if set(['latitude', 'longitude']).issubset(df.columns):
        df = df.drop(['latitude', 'longitude'], axis = 1)
    # convert non-ascii characters by ignore them
    df[area_name] = df[area_name].apply(lambda x: x.\
                                              encode('ascii', 'ignore').\
                                              decode('ascii').\
                                              strip())
    return df





# ============ Step 00: Preprocessing dataset ==============
# This method is used to work with air reserve visistor file, which is not important
# Because this file did not contain enough visit date like air visit data file
def air_reserve_visitor_format(df):
    df['visit_datetime'] = pd.to_datetime(df['visit_datetime'], format="%Y/%m/%d").dt.date
    df = (df.groupby(['air_store_id','visit_datetime'])
       .agg({'reserve_visitors': 'sum'})
       .reset_index()
       .rename(columns={'air_store_id':'store_id', 'visit_datetime':'visit_date', 'reserve_visitors':'visitors'})
    )
    return df

# This function will restructure or reformat and group hpg reserve file
# output:
#   df: dataframe for hpg visit dataset.
def hpg_reserve_visitor_format(df, df_hsi):
    df['visit_datetime'] = pd.to_datetime(df['visit_datetime'], format="%Y/%m/%d").dt.date
    df = (df.groupby(['hpg_store_id','visit_datetime'])
        .agg({'reserve_visitors': 'sum'})
        .reset_index()
        .rename(columns={'hpg_store_id':'store_id', 'visit_datetime':'visit_date', 'reserve_visitors':'visitors'})
    )
    # Total = df['visitors'].sum()
    # print("Sum of df hpg reserve visitors:\n", Total)
    # print("hpg_reserve_visitor_format:\n", df)

    print("df_hsi-----------:\n", df_hsi)
    mask_store_id = df_hsi[store_id].unique()
    print("mask_store_id-----------:\n", mask_store_id)
    print("mask_store_id type      :\n", type(mask_store_id))
    # Convert array to list
    mask_store_id = mask_store_id.tolist()
    # df = df[mask_store_id].reset_index(drop=True)
    print("df_hr of all stores from hpg reserve:\n", df)
    # Get only store with store id in the mask. Because store id dataset has 4689 store, but df_hr has many more store.
    # We need to trim all those stores without genres and areas. Because we need those properties for splitting later.
    df = df[df[store_id].isin(mask_store_id)].reset_index(drop=True)
    print("df_hr after trimming store which are not exist in store id dataset:\n", df)
    return df

# Get 3 first values of the first column in a dataframe
def get_first_3values_from_df(df):
    # print("type of df:", type(df))
    location_1st = df.iloc[0][0]
    location_2nd = df.iloc[1][0]
    location_3rd = df.iloc[2][0]
    return location_1st, location_2nd, location_3rd

def format_arename_col_first_word(df):
    # Get first letter from area name column
    df['new_col'] = df[area_name].astype(str).str.split().str.get(0)
    df[area_name] = df['new_col']
    df.drop('new_col', axis=1, inplace=True)
    return df

# method: under sampling by shuffling and getting only first size
#   input:
#       df: dataframe for under sampling
#       size: maximum remain size of dataframe
#   output:
#       df: contains only rows with size
def under_sampling(df, size):
    # Shuffle dataset in row
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.iloc[0:size] # first size rows of dataframe
    return df

# method: over sampling by shuffling and generate new timeseries base on the existing timeseries
#   input:
#       df: input dataframe for over sampling
#       size: maximum size of new dataframe
#       attrib_1_type: is the type of the gerne_name or area_name
#       attrib_1_val : is the val  of the gerne or area. Ex: Izakaya, Cafe/Sweets or Dining bar
#   output:
#       df: contain exisiting dataframe and a new generated one
#       df_vd_added: is the new generated dataframe
def over_sampling(df, df_vd, size_max, attrib_1, attrib_2):
    size_current_df = df.shape[0]
    # Get 2 values of 2 attributes, which are be used for creating columns for df_store_info_added
    attrib_1_val = df.iloc[0][attrib_1]
    attrib_2_val = df.iloc[0][attrib_2]

    # Create a new dataframe store information, which contains genre and area for further processing in upcoming steps
    n = size_max-size_current_df
    df_si_added = pd.DataFrame()
    df_original = df.copy()


    # Create a mask, then apply it for greater group of stores is df_vd
    mask_store_id = df[store_id].unique()
    # Convert array to list
    mask_store_id = mask_store_id.tolist()
    # Get only store with store id in the mask. Because store id dataset has 4689 store, but df_hr has many more store.
    # We need to trim all those stores without genres and areas. Because we need those properties for splitting later.
    df = df_vd[df_vd[store_id].isin(mask_store_id)].reset_index(drop=True)
    df = df.pivot(index=store_id, columns='visit_date', values='visitors')

    # Creating empty dataframe as visit date
    df_vd_added = pd.DataFrame()
    df_si_added = pd.DataFrame()

    # In case df has size is the same with maximum of 9 groups: return the old df.
    if size_max == size_current_df:
        print("Current size of dataframe equals size to upsampling: return old df")
        # print("df_si_added:\n", df_si_added)
        # print("df_vd_added after :\n", df_vd_added)
        # print("input dataframe for over sampling df:\n", df)
        return df_original, df_vd_added, df_si_added

    else: # Current size of dataframe is not equals with size to upsampling --> over-sampling
        columns = list(df)

        for j in range(n):
            my_list = []
            # Iterating through columns of dataframe then get one random value from that column.
            for i in columns:
                my_list.append(random.choice(df[i].values))
            # id = uuid.uuid1()
            id = 'over_' + uuid.uuid4().hex[:16]
            # print("type of id:", type(id))
            # Create a new dataframe with column name as a id, which is used as store id for new upsamples
            df1 = pd.DataFrame(my_list, columns=[id])
            # Concatenate multiple dataframes
            df_vd_added = pd.concat([df_vd_added, df1], axis=1)

        # Transposing the concatenated dataframe and assign new columns by the above dataframe df
        df_vd_added = df_vd_added.transpose()
        df_vd_added.columns = df.columns

        # Concatening new dataframe with the existed one to form up a bigger dataframe

        df_vd_added = df_vd_added.reset_index()
        df_vd_added.rename(columns={'index': 'store_id'}, inplace=True)
        df_vd_added = df_vd_added.set_index('store_id')
        df_vd_added = df_vd_added.stack().reset_index()
        df_vd_added.columns=['store_id', 'visit_date', 'visitors']

        df_si_added = df_vd_added[['store_id']].copy()
        df_si_added = df_si_added.drop_duplicates('store_id').reset_index()
        df_si_added = df_si_added.assign(attrib_1=attrib_1_val)
        df_si_added = df_si_added.assign(attrib_2=attrib_2_val)
        df_si_added.rename(columns={'attrib_1':attrib_1, 'attrib_2':attrib_2}, inplace=True)
        df_si_added = df_si_added.drop(columns = ['index'])

        # print("df_original:\n", df_original)
        # print("df_si_added:\n", df_si_added)

        df = pd.merge(df_original, df_si_added, on=[store_id, attrib_1, attrib_2], how='outer')
        # print("df ==========:\n", df)
    return df, df_vd_added, df_si_added

# Method: get 3 equal size of sampling
#   input:
#       df
#       attrib_1_1st, attrib_1_2nd, attrib_1_3rd: 3 first values of the first feature or attribute
#       attrib_1, attrib_2  : 2 features
#       RESAMPLING_METHOD_ARG : Method for sampling
#   output:
#       df_attrib_1: All stores after under-sampling or over-sampling
#       df_attrib_1_1st, df_attrib_1_2nd, df_attrib_1_3rd
def get_3_equal_attrib_1_size_sampling(df, attrib_1_1st, attrib_1_2nd, attrib_1_3rd, attrib_1, attrib_2, df_vd):
    df_attrib_1_1st = df.loc[df[attrib_1].isin([attrib_1_1st])].reset_index(drop=True)
    df_attrib_1_2nd = df.loc[df[attrib_1].isin([attrib_1_2nd])].reset_index(drop=True)
    df_attrib_1_3rd = df.loc[df[attrib_1].isin([attrib_1_3rd])].reset_index(drop=True)

    # if RESAMPLING_METHOD_ARG == 'under':
    size = df_attrib_1_3rd.shape[0]
    df_attrib_1_1st = under_sampling(df_attrib_1_1st, size)
    # Due to randomly choosing in undersampling method, location or area of those series will be shuffle as well.
    # This action will affect in choosing 3 largest areas in the next step
    print("df_attrib_1_1st:\n", df_attrib_1_1st)
    print("df_attrib_1_1st shape 0 :", df_attrib_1_1st.shape[0])

    df_attrib_1_2nd = under_sampling(df_attrib_1_2nd, size)
    print("df_attrib_1_2nd:\n", df_attrib_1_2nd)
    print("df_attrib_1_2nd shape:", df_attrib_1_2nd.shape[0])

    frames_after_sampling = [df_attrib_1_1st, df_attrib_1_2nd, df_attrib_1_3rd]
    df_attrib_1 = pd.concat(frames_after_sampling).reset_index(drop=True)

    return df_attrib_1, df_attrib_1_1st, df_attrib_1_2nd, df_attrib_1_3rd

# Method: get 3 upsampling_df
#   input:
#       df
#       attrib_1_1st, attrib_1_2nd, attrib_1_3rd: 3 first values of the first feature or attribute
#       attrib_1, attrib_2  : 2 features
#       RESAMPLING_METHOD_ARG : Method for sampling
#   output:
#       df_si_sampled: Stores after over-sampling
#       df_vd_added: Stores visit date added after over-sampling
#       df_si_added: Stores information added after over-sampling
def get_upsampling_df(df, attrib_1_1st, attrib_1_2nd, attrib_1_3rd, attrib_1, attrib_2, df_vd):
    df_attrib_1_1st = df.loc[df[attrib_1].isin([attrib_1_1st])].reset_index(drop=True)
    df_attrib_1_2nd = df.loc[df[attrib_1].isin([attrib_1_2nd])].reset_index(drop=True)
    df_attrib_1_3rd = df.loc[df[attrib_1].isin([attrib_1_3rd])].reset_index(drop=True)

    # print("input df:\n", df)
    # print("df_attrib_1_1st:\n", df_attrib_1_1st)
    # print("df_attrib_1_2nd:\n", df_attrib_1_2nd)
    # print("df_attrib_1_3rd:\n", df_attrib_1_3rd)
    #
    # print("attrib_1:", attrib_1)
    # print("attrib_2:", attrib_2)

    df_attrib_1_1st_groupby_attrib_2 = df_attrib_1_1st.groupby(attrib_2).count().sort_values(store_id, ascending=False).reset_index()
    # print("df_attrib_1_1st_groupby_attrib_2:\n", df_attrib_1_1st_groupby_attrib_2)

    attrib_1_1st_2_1st, attrib_1_1st_2_2nd, attrib_1_1st_2_3rd = get_first_3values_from_df(df_attrib_1_1st_groupby_attrib_2)
    # print("attrib_1_1st_2_1st:", attrib_1_1st_2_1st)
    # print("attrib_1_1st_2_2nd:", attrib_1_1st_2_2nd)
    # print("attrib_1_1st_2_3rd:", attrib_1_1st_2_3rd)

    # These 9 dataframes below cover all rows of 9 groups - 3 for first attrib product with 3 for second attrib
    # Filtering from 3 groups of data for first feature, which have second feature
    df_attrib_1_1st_2_1st = df_attrib_1_1st.loc[df_attrib_1_1st[attrib_2] == attrib_1_1st_2_1st].reset_index(drop=True)
    df_attrib_1_1st_2_2nd = df_attrib_1_1st.loc[df_attrib_1_1st[attrib_2] == attrib_1_1st_2_2nd].reset_index(drop=True)
    df_attrib_1_1st_2_3rd = df_attrib_1_1st.loc[df_attrib_1_1st[attrib_2] == attrib_1_1st_2_3rd].reset_index(drop=True)

    # print("df_attrib_1_1st_2_1st:\n", df_attrib_1_1st_2_1st)
    # print("df_attrib_1_1st_2_2nd:\n", df_attrib_1_1st_2_2nd)
    # print("df_attrib_1_1st_2_3rd:\n", df_attrib_1_1st_2_3rd)

    df_attrib_1_2nd_2_1st = df_attrib_1_2nd.loc[df_attrib_1_2nd[attrib_2] == attrib_1_1st_2_1st].reset_index(drop=True)
    df_attrib_1_2nd_2_2nd = df_attrib_1_2nd.loc[df_attrib_1_2nd[attrib_2] == attrib_1_1st_2_2nd].reset_index(drop=True)
    df_attrib_1_2nd_2_3rd = df_attrib_1_2nd.loc[df_attrib_1_2nd[attrib_2] == attrib_1_1st_2_3rd].reset_index(drop=True)

    df_attrib_1_3rd_2_1st = df_attrib_1_3rd.loc[df_attrib_1_3rd[attrib_2] == attrib_1_1st_2_1st].reset_index(drop=True)
    df_attrib_1_3rd_2_2nd = df_attrib_1_3rd.loc[df_attrib_1_3rd[attrib_2] == attrib_1_1st_2_2nd].reset_index(drop=True)
    df_attrib_1_3rd_2_3rd = df_attrib_1_3rd.loc[df_attrib_1_3rd[attrib_2] == attrib_1_1st_2_3rd].reset_index(drop=True)

    # Get standard size of the maximum of both 1st attrib and 2nd attrib
    size_1_1st_max = max(df_attrib_1_1st_2_1st.shape[0], df_attrib_1_1st_2_2nd.shape[0], df_attrib_1_1st_2_3rd.shape[0])
    size_1_2nd_max = max(df_attrib_1_2nd_2_1st.shape[0], df_attrib_1_2nd_2_2nd.shape[0], df_attrib_1_2nd_2_3rd.shape[0])
    size_1_3rd_max = max(df_attrib_1_3rd_2_1st.shape[0], df_attrib_1_3rd_2_2nd.shape[0], df_attrib_1_3rd_2_3rd.shape[0])
    max_size_9_groups = max(size_1_1st_max, size_1_2nd_max, size_1_3rd_max)

    print("max_size_9_groups:", max_size_9_groups)

    # Oversampling 8 dataframes base on the first maximum dataframe
    df_attrib_1_1st_2_1st, df_vd_added_01, df_hsi_added_01 = over_sampling(df_attrib_1_1st_2_1st, df_vd, max_size_9_groups, attrib_1, attrib_2)
    df_attrib_1_1st_2_2nd, df_vd_added_02, df_hsi_added_02 = over_sampling(df_attrib_1_1st_2_2nd, df_vd, max_size_9_groups, attrib_1, attrib_2)
    df_attrib_1_1st_2_3rd, df_vd_added_03, df_hsi_added_03 = over_sampling(df_attrib_1_1st_2_3rd, df_vd, max_size_9_groups, attrib_1, attrib_2)

    df_attrib_1_2nd_2_1st, df_vd_added_04, df_hsi_added_04 = over_sampling(df_attrib_1_2nd_2_1st, df_vd, max_size_9_groups, attrib_1, attrib_2)
    df_attrib_1_2nd_2_2nd, df_vd_added_05, df_hsi_added_05 = over_sampling(df_attrib_1_2nd_2_2nd, df_vd, max_size_9_groups, attrib_1, attrib_2)
    df_attrib_1_2nd_2_3rd, df_vd_added_06, df_hsi_added_06 = over_sampling(df_attrib_1_2nd_2_3rd, df_vd, max_size_9_groups, attrib_1, attrib_2)


    df_attrib_1_3rd_2_1st, df_vd_added_07, df_hsi_added_07 = over_sampling(df_attrib_1_3rd_2_1st, df_vd, max_size_9_groups, attrib_1, attrib_2)
    df_attrib_1_3rd_2_2nd, df_vd_added_08, df_hsi_added_08 = over_sampling(df_attrib_1_3rd_2_2nd, df_vd, max_size_9_groups, attrib_1, attrib_2)
    df_attrib_1_3rd_2_3rd, df_vd_added_09, df_hsi_added_09 = over_sampling(df_attrib_1_3rd_2_3rd, df_vd, max_size_9_groups, attrib_1, attrib_2)

    # Concatenate list of frames for 9 groups
    frames_9_groups = [df_attrib_1_1st_2_1st, df_attrib_1_1st_2_2nd, df_attrib_1_1st_2_3rd,
                      df_attrib_1_2nd_2_1st, df_attrib_1_2nd_2_2nd, df_attrib_1_2nd_2_3rd,
                      df_attrib_1_3rd_2_1st, df_attrib_1_3rd_2_2nd, df_attrib_1_3rd_2_3rd]
    df_si_sampled = pd.concat(frames_9_groups).reset_index(drop=True)


    frames_vd_added = [df_vd_added_01, df_vd_added_02, df_vd_added_03,
                      df_vd_added_04, df_vd_added_05, df_vd_added_06,
                      df_vd_added_07, df_vd_added_08, df_vd_added_09]
    df_vd_added = pd.concat(frames_vd_added).reset_index(drop=True)

    frames_si_added = [df_hsi_added_01, df_hsi_added_02, df_hsi_added_03,
                      df_hsi_added_04, df_hsi_added_05, df_hsi_added_06,
                      df_hsi_added_07, df_hsi_added_08, df_hsi_added_09]
    df_si_added = pd.concat(frames_si_added).reset_index(drop=True)

    # print("df_si_sampled:", df_si_sampled)
    # print("df_vd_added:", df_vd_added)
    # print("df_si_added:", df_si_added)
    # sys.exit()

    return df_si_sampled, df_vd_added, df_si_added

# Method use: Get store info dataset and visit dataset of 2 datasouces
# input:
#   RES_DATASET_ARG: resource dataset argument
# output:
def get_storeinfo_visitdata(df_asi, df_avd, df_hsi, df_hr, RES_DATASET_ARG):
    print("RES_DATASET_ARG:", RES_DATASET_ARG)

    if RES_DATASET_ARG == 'air':
        df_asi = store_info_format(df_asi)
        return df_asi, df_avd

    else: # RES_DATASET_ARG == 'hpg':
        df_hsi = store_info_format(df_hsi)
        df_hr = hpg_reserve_visitor_format(df_hr, df_hsi)
        return df_hsi, df_hr


# Method use: Merging store_info_dataset with visit data from one of the resource
# input:
#   store_info_dataset, df_vd
# output:
#   df_merged_storeid_visits: dataset after merging store_info_dataset, df_vd
def merge_store_visit_2nd(store_info_dataset, df_vd):
    print("Store id dataset before merging - store_info_dataset - store_info_dataset:\n", store_info_dataset)
    print("Store id dataset and visitor data before merging - df_vd:\n", df_vd)

    df_merged_storeid_visits = pd.merge(store_info_dataset, df_vd, how='inner', on=['store_id'])
    print("Store id dataset and visitor data after  merging:\n", df_merged_storeid_visits)
    df_merged_storeid_visits = format_arename_col_first_word(df_merged_storeid_visits)
    print("df_merged_storeid_visits after format first area_name column :\n", df_merged_storeid_visits)
    return df_merged_storeid_visits

# Method use: Merging store_info_dataset with visit data from one of the resource
# input:
#   RES_DATASET_ARG: resource dataset argument
# output:
#
def split_store_visit_2nd(df_merged, RES_DATASET_ARG, RESAMPLING_METHOD_ARG, df_asi, df_avd, df_hsi, df_hr):
    print("df_merged:\n", df_merged)
    df_groups_store_id = df_merged
    df_groups_store_id = df_groups_store_id.drop(['visit_date', 'visitors'], axis=1)
    df_groups_store_id = df_groups_store_id.drop_duplicates().reset_index()
    print("df_groups_store_id:\n", df_groups_store_id)

    df_groups = df_groups_store_id.groupby(genre_name).count().sort_values(store_id, ascending=False).reset_index()
    print('=== LIST OUT ALL GENRE GROUPS ASCENDING df_groups: === \n', df_groups)

    store_info_dataset, df_vd = get_storeinfo_visitdata(df_asi, df_avd, df_hsi, df_hr, RES_DATASET_ARG)

    # ========= Overall of 3 genres and 3 locations =========
    # Filtering air_store_info df to 3 main genres


    gerne_1st, gerne_2nd, gerne_3rd = get_first_3values_from_df(df_groups)
    print("gerne_1st:", gerne_1st)
    print("gerne_2nd:", gerne_2nd)
    print("gerne_3rd:", gerne_3rd)

    print("store_info_dataset:\n", store_info_dataset)
    df_genre, df_genre_1st, df_genre_2nd, df_genre_3rd = get_3_equal_attrib_2_size_sampling(df_groups_store_id,
                                                                gerne_1st, gerne_2nd, gerne_3rd, RESAMPLING_METHOD_ARG)

    print("=== ALL RESTAURANTS OF 3 MAIN GENRES AFTER SAMPLING - BEGIN: ===\n")
    print("df_genre:\n",df_genre)
    print("df_genre_1st:\n",df_genre_1st)
    print("df_genre_2nd:\n",df_genre_2nd)
    print("df_genre_3rd:\n",df_genre_3rd)
    print('Total number of restaurants in three equalled-size main genres:', df_genre.shape[0])
    print("=== ALL RESTAURANTS OF 3 MAIN GENRES AFTER SAMPLING - END:   ===\n")


# === Find the first and the last moments of a time serie
# input:
# df_sav - dataframe is the merged between store list and visit date
# output:
#     first_moment: dataframe contains all first moment of all time series
#     last_moment : dataframe contains all last  moment of all time series
def first_last_moments(df_sav):
    # print("=== df_sav ===\n", df_sav)
    store_id_list = df_sav[store_id].tolist()
    store_id_list = list(set(store_id_list))
    # print("=== store_id_list ===\n", store_id_list)
    print('Length of Store id list:', len(store_id_list))
    store_id_array = np.asarray(store_id_list)

    full_series = df_sav.loc[df_sav[store_id].isin(store_id_array)]
    # print("All series has in 3 genres and 3 locations with visit date:\n", full_series)
    first_moment = full_series.groupby(store_id).head(1).reset_index(drop=True)
    last_moment  = full_series.groupby(store_id).tail(1).reset_index(drop=True)
    return first_moment, last_moment

# This function get input of a df of visit date and get the average from visit date column
# input:
#   fm  : first moment dataframe
#   lm  : last moment dataframe
# output:
#   average_fm: average time point first moment
#   average_lm: average time point last  moment
def average_first_last_moments(fm, lm):
    average_fm = fm["visit_date"].pipe(lambda d: (lambda m: m + (d - m).mean())(d.min()))
    average_lm = lm["visit_date"].pipe(lambda d: (lambda m: m + (d - m).mean())(d.min()))
    average_fm = average_fm.strftime('%m-%d-%Y')
    average_lm = average_lm.strftime('%m-%d-%Y')

    print('Median value of first moment dataframe:', average_fm)
    print('Median value of last  moment dataframe:', average_lm)
    return average_fm, average_lm

# Method: This function get input of a df of visit date and get the first and last from visit date column
# input:
#   fm  : first moment dataframe
#   lm  : last moment dataframe
# output:
#   min_fm: min time point of first moment
#   max_lm: max time point of last  moment
def min_max_first_last_moments(fm, lm):
    min_fm = fm["visit_date"].min()
    max_lm = lm["visit_date"].max()
    min_fm = min_fm.strftime('%m-%d-%Y')
    max_lm = max_lm.strftime('%m-%d-%Y')

    print('Min value of first moment dataframe:', min_fm)
    print('Max value of last  moment dataframe:', max_lm)
    return min_fm, max_lm

def percentage(part, whole, digits):
    val = float(part)/float(whole)
    val *= 10 ** (digits + 2)
    return (floor(val) / 10 ** digits)

# Method used: Get the differences betweend 2 date
# input:
#   last_timepoint : last  time point
#   first_timepoint: first time point
# output:
#   days_diff: differences in days
def get_days_different(last_timepoint, first_timepoint):
    last_timepoint  = datetime.strptime(last_timepoint, "%m-%d-%Y")
    first_timepoint = datetime.strptime(first_timepoint, "%m-%d-%Y")
    days_diff = abs((last_timepoint - first_timepoint).days)
    print("Days different between first day and last day:", days_diff)
    return days_diff

# Method used: Find the missing percentage of df_store_and_visit
# input:
#   df: contains all stores and all visits
#   fm: first moment dataframe of all stores
#   lm: last  moment dataframe of all stores
# output:
#   Total missing percentage of all stores base on first and last moment date range
def missing_percentage(df, first_timepoint, last_timepoint):
    # date_range_idx = pd.date_range(min_date, max_date)

    # print("---------------df\n", df)
    current_num_store_and_visit = df.shape[0]
    print("---------------total_num_rows including stores and visitors:", current_num_store_and_visit)

    total_num_store = df[store_id].nunique()
    print("---------------total_num_store:", total_num_store)
    # Finding the min and max day of a column
    # d1 = df.visit_date.min()
    # d2 = df.visit_date.max()

    last_timepoint  = datetime.strptime(last_timepoint, "%m-%d-%Y")
    first_timepoint = datetime.strptime(first_timepoint, "%m-%d-%Y")
    days_diff = abs((last_timepoint - first_timepoint).days)
    # print("Days different between first day and last day:", days_diff)

    # Total number of possible day for all stores
    total_num_possible_day = total_num_store*days_diff
    return percentage(current_num_store_and_visit, total_num_possible_day, 2)



# method: Filtering visitor data of high missing percentage of stores
# input:
#   df_vd
#   MAX_MISSING_PERCENTAGE_ARG: Percent of top stores values
# output:
#   df_vd after filtering high missing percentage
def filtering_high_missing_rate_store(df_vd, MAX_MISSING_PERCENTAGE_ARG):
    print("+++Filtering_high_missing_rate_store function+++")
    print("input visitor data from air or hpg - df_vd:\n", df_vd)
    print("input MAX_MISSING_PERCENTAGE_ARG:", MAX_MISSING_PERCENTAGE_ARG)

    # Get the first_moment and last_moment dataframe of all stores
    first_moment, last_moment = first_last_moments(df_vd)
    # print("first_moment dataframe:\n", first_moment)
    # print("last_moment  dataframe:\n", last_moment)

    # Get the min date and max date from first moment and last moment
    min_date, max_date = min_max_first_last_moments(first_moment, last_moment)

    # Total missing rate percent
    miss_percent = missing_percentage(df_vd, min_date, max_date)
    print("Overall missing percentage of all stores:", miss_percent)

    # Create new temporary dataframe to contain percentage for each store
    df_store_percentage = df_vd
    # Adding one more column, which counts number of appearance of in store id for each store. And call it as 'store_count'
    df_store_percentage = df_store_percentage[['store_id']].groupby(['store_id']).store_id.agg('count').to_frame('store_day_count').reset_index()
    days_different = get_days_different(max_date, min_date)
    # Create a new column as store percentage by calculate number of store id over
    df_store_percentage['store_percentage'] = np.vectorize(percentage)(df_store_percentage['store_day_count'],days_different, 2)

    # Sort values of percentage from max to min and reset index for this dataframe
    df_store_percentage = df_store_percentage.sort_values(by=['store_percentage'], ascending=False).reset_index(drop=True)
    # Get only MAX_MISSING_PERCENTAGE_ARG percent of top values
    df_store_percentage = df_store_percentage.head(int(len(df_store_percentage)*(MAX_MISSING_PERCENTAGE_ARG/100)))
    print("Stores after getting only MAX_MISSING_PERCENTAGE_ARG percent from df_store_percentage:\n", df_store_percentage)
    print("Describing current dataframe as data analysis:\n", df_store_percentage.describe())
    return df_store_percentage

# Left join 3 dataframes to get a new dataframe with maximium rows.
def left_join_df(df_left, df_middle, df_right, attrib_2):
    print("==== Left join function ===")
    # Fitering 2 first columns of dataframe for easier to work with area name and sum
    df_left   = df_left.iloc[:,0:2]
    df_middle = df_middle.iloc[:,0:2]
    df_right  = df_right.iloc[:,0:2]


    # Get the shape of 3 dataframes for left join
    left   = df_left.shape[0]
    middle = df_middle.shape[0]
    right  = df_right.shape[0]

    # Finding maximum number of the
    num = [left, middle, right]
    max1 = max(num)
    if max1 == middle:
        df_left, df_middle, = df_middle, df_left
    elif max1 == right:
        df_left, df_right, = df_right, df_left

    # print("df_left   df:\n", df_left)
    # print("df_middle df:\n", df_middle)
    # print("df_right  df:\n", df_right)

    df_left_join = df_left.merge(df_middle,how='left', left_on=attrib_2, right_on=attrib_2)
    df_left_join = df_left_join.merge(df_right,how='left', left_on=attrib_2, right_on=attrib_2)
    df_left_join = df_left_join.fillna(0)
    print("df_left_join df_left_join:\n", df_left_join)
    df = df_left_join[attrib_2]
    # print("df df:\n", df)
    df_left_join = df_left_join.min(axis = 1, skipna = True)
    df_left_join = pd.concat([df, df_left_join], axis=1)
    # list all column names of the dataframe
    print("list(df_left_join.columns.values):", list(df_left_join.columns.values))

    # Rename column by position
    df_left_join.rename(columns={ df_left_join.columns[1]: "restaurants_count" }, inplace = True)
    df_left_join = df_left_join.sort_values('restaurants_count', ascending=False).reset_index(drop=True)
    df_left_join["restaurants_count"] = df_left_join["restaurants_count"].astype(int)
    print("df_left_join df_left_join -----:\n", df_left_join)

    return df_left_join

# Get a new dataframe which contain equal size of restaurants for 3 areas
# by resampling dataset by undersampling or oversampling
# Input:
#   df: full dataset of 1 genre or area
#   attrib_2_1st, attrib_2_2nd, attrib_2_3rd: 3 max gernes or are of the full input dataframe
#   size: size of the least number of restaurant of 3 areas. By this we have 3 equal-sized areas
# Output:
def get_3_equal_attrib_2_size_sampling(df, attrib_2_1st, attrib_2_2nd, attrib_2_3rd, size, RESAMPLING_METHOD_ARG, attrib_2):
    df_attrib_2_1st = df.loc[df[attrib_2].isin([attrib_2_1st])].reset_index(drop=True)
    df_attrib_2_2nd = df.loc[df[attrib_2].isin([attrib_2_2nd])].reset_index(drop=True)
    df_attrib_2_3rd = df.loc[df[attrib_2].isin([attrib_2_3rd])].reset_index(drop=True)

    print("Test df_attrib_2_1st:\n", df_attrib_2_1st)
    print("Default size need to be cut of from the main part:", size)

    if RESAMPLING_METHOD_ARG == 'under':
        df_attrib_2_1st = under_sampling(df_attrib_2_1st, size)
        print("df_attrib_2_1st:\n", df_attrib_2_1st)
        print("df_attrib_2_1st shape 0 :", df_attrib_2_1st.shape[0])

        df_attrib_2_2nd = under_sampling(df_attrib_2_2nd, size)
        print("df_attrib_2_2nd:\n", df_attrib_2_2nd)
        print("df_attrib_2_2nd shape:", df_attrib_2_2nd.shape[0])

        df_attrib_2_3rd = under_sampling(df_attrib_2_3rd, size)
        print("df_attrib_2_3rd:", df_attrib_2_3rd)
        print("df_attrib_2_3rd shape:", df_attrib_2_3rd.shape[0])
    # else:
    # #     sampling method == 'over'
    #     size = df_location_1st.shape[0]
    #     df_location_2nd = over_sampling(df_location_2nd, size)
    #     df_location_3rd = over_sampling(df_location_3rd, size)

    frames_after_sampling = [df_attrib_2_1st, df_attrib_2_2nd, df_attrib_2_3rd]
    df_genre_area = pd.concat(frames_after_sampling).reset_index(drop=True)
    print("Full df_genre_area for 1 genre:\n", df_genre_area)

    return df_genre_area

# Method to split dataset to 3 genres/area groups and 9 genre-area/area-genre groups.
#   input: df - input store info file with full genre, area. This file did not include time series
#   output:  df_attrib_1, df_attrib_1_attrib_2
def split_to_9_groups(df, df_vd, df_si, SPLIT_FIRST_BY_ARG, RESAMPLING_METHOD_ARG):
    if SPLIT_FIRST_BY_ARG == 'genre':
        attrib_1 = 'genre_name'
        attrib_2 = 'area_name'
    else: # SPLIT_FIRST_BY_ARG == 'are'
        attrib_1 = 'area_name'
        attrib_2 = 'genre_name'

    print("input dataframe of split_to_9_groups function:\n", df)
    # Group by attrib_1 and count unique values with pandas per groups for air_store_info.csv,
    df_groups = df.groupby(attrib_1).count().sort_values(store_id, ascending=False).reset_index()
    print('=== LIST OUT ALL GROUPS ASCENDING BY: ', attrib_1, '===\n', df_groups)

    # Get names of 3 first columns (col1). These three variables are type of string
    attrib_1_1st, attrib_1_2nd, attrib_1_3rd = get_first_3values_from_df(df_groups)
    print("attrib_1_1st:", attrib_1_1st)
    print("attrib_1_2nd:", attrib_1_2nd)
    print("attrib_1_3rd:", attrib_1_3rd)


    # df_genre: full of 3 equal attribute 1 dataframe
    if RESAMPLING_METHOD_ARG == 'under':
        df_attrib_1, df_attrib_1st, df_attrib_2nd, df_attrib_3rd = get_3_equal_attrib_1_size_sampling(df,
                            attrib_1_1st, attrib_1_2nd, attrib_1_3rd, attrib_1, attrib_2, df_vd)

        print("=== ALL RESTAURANTS OF 3 MAIN", attrib_1, "AFTER SAMPLING ===\n")
        print(df_attrib_1)
        print('Total number of restaurants in three equalled-size main by ', attrib_1, ':', df_attrib_1.shape[0])

        # Group by area for 3 genres df_attrib_1st, df_attrib_2nd, df_attrib_3rd then pipe it to input for
        # equallizing size of area or genre
        df_genre_area_1st = df_attrib_1st.groupby(attrib_2).count().sort_values(store_id, ascending=False).reset_index()
        print("df_genre_area_1st:\n", df_genre_area_1st)
        df_genre_area_2nd = df_attrib_2nd.groupby(attrib_2).count().sort_values(store_id, ascending=False).reset_index()
        print("df_genre_area_2nd:\n", df_genre_area_2nd)
        df_genre_area_3rd = df_attrib_3rd.groupby(attrib_2).count().sort_values(store_id, ascending=False).reset_index()
        print("df_genre_area_3rd:\n", df_genre_area_3rd)

        # Create left join dataframe to get merge of 3 genres and 3 areas.
        # With this dataframe, we can identify 3 top areas with the LEAST number of restaurants of the third place
        df_left_join = left_join_df(df_genre_area_1st, df_genre_area_2nd, df_genre_area_3rd, attrib_2)
        print("df_left_join:\n", df_left_join)

        # Get first 3 attributes from the df_left_join above
        attrib_2_1st, attrib_2_2nd, attrib_2_3rd = get_first_3values_from_df(df_left_join)
        print("attrib_2_1st:", attrib_2_1st)
        print("attrib_2_2nd:", attrib_2_2nd)
        print("attrib_2_3rd:", attrib_2_3rd)

        # Get the size of the least areas in top 3 most areas
        genre_area_size = df_left_join.iloc[2][1]
        print("genre_area_size:", genre_area_size)

        # Method 02: search in each of df_genre_area_3rd, equalize 3 proportions
        df_attrib_1_attrib_2_1st = get_3_equal_attrib_2_size_sampling(df_attrib_1st, attrib_2_1st, attrib_2_2nd, attrib_2_3rd, genre_area_size, RESAMPLING_METHOD_ARG, attrib_2)
        df_attrib_1_attrib_2_2nd = get_3_equal_attrib_2_size_sampling(df_attrib_2nd, attrib_2_1st, attrib_2_2nd, attrib_2_3rd, genre_area_size, RESAMPLING_METHOD_ARG, attrib_2)
        df_attrib_1_attrib_2_3rd = get_3_equal_attrib_2_size_sampling(df_attrib_3rd, attrib_2_1st, attrib_2_2nd, attrib_2_3rd, genre_area_size, RESAMPLING_METHOD_ARG, attrib_2)

        # Full store id list
        frames = [df_attrib_1_attrib_2_1st, df_attrib_1_attrib_2_2nd, df_attrib_1_attrib_2_3rd]
        df_si_sampled = pd.concat(frames).reset_index(drop=True)
        print("df_si_sampled:===\n", df_si_sampled)
        print('Total number of restaurants in 3 main ', attrib_1,' with 3 main', attrib_2,':', df_si_sampled.shape[0], '\n')

        # df_si_sampled.to_csv("df_si_sampled.csv")

        # sys.exit()
        return df_si_sampled, df_vd, df_si
    else: # sampling method == 'over'
        df_si_sampled, df_vd_added, df_si_added = get_upsampling_df(df,
                            attrib_1_1st, attrib_1_2nd, attrib_1_3rd, attrib_1, attrib_2, df_vd)
        # df_attrib_1_attrib_2, df_vd_added, df_si_added



        print("df_si original from air or hpg:\n", df_si)
        print("df_si_added:\n", df_si_added)
        # sys.exit()
        df_vd = pd.merge(df_vd, df_vd_added, on=[store_id, visit_date, visitors], how='outer')
        # print("df_si_added before merging:\n", df_si_added)
        # print("df_si before:\n", df_si)
        df_si = pd.merge(df_si, df_si_added, on=[store_id, attrib_1, attrib_2], how='outer')
        # print("df_si after:\n", df_si)
        # sys.exit()

        # print("df_si_sampled:\n", df_si_sampled)
        # print("df_vd:\n", df_vd)
        print("df_si:\n", df_si)
        # sys.exit()
        return df_si_sampled, df_vd, df_si

# method: Filtering, merging and splitting dataset to get visitor matrix
# input:
#   df_asi, df_avd, df_hsi, df_hr: input dataset
#   RES_DATASET_ARG
#   MAX_MISSING_PERCENTAGE_ARG
#   SPLIT_BY_ARG
#   RESAMPLING_METHOD_ARG
# output:
#   visitor matrix for clustering methods
def filtering_splitting_merging(df_asi, df_avd, df_hsi, df_hr,
                                    RES_DATASET_ARG, MAX_MISSING_PERCENTAGE_ARG,
                                    SPLIT_FIRST_BY_ARG, RESAMPLING_METHOD_ARG):
    # Getting appropriate dataset resources
    df_si, df_vd = get_storeinfo_visitdata(df_asi, df_avd, df_hsi, df_hr, RES_DATASET_ARG)

    # Filtering visistor data from dataframe base on MAX_MISSING_PERCENTAGE_ARG
    df_vd_high_missing_filtered = filtering_high_missing_rate_store(df_vd, MAX_MISSING_PERCENTAGE_ARG)

    # Merging store info dataset and visit data to one dataframe
    df_merged = merge_store_visit_2nd(df_si, df_vd_high_missing_filtered)

    # Splitting store infodataset by genres or by area
    df_si_sampled, df_vd, df_si = split_to_9_groups(df_merged, df_vd, df_si, SPLIT_FIRST_BY_ARG, RESAMPLING_METHOD_ARG)


    # Merging df_attrib_1, df_attrib_1_attrib_2 with df_vd which preparing for imputation step
    print("df_si_sampled before merging:\n", df_si_sampled)
    print("df_vd dataframe contains all visit date added if oversampling:\n", df_vd)
    print("df_si dataframe contains all store information including added if oversampling:\n", df_si)
    # sys.exit()

    # df_attrib_1 = pd.merge(df_attrib_1, df_vd, on=[store_id], how='inner')
    df_si_sampled = pd.merge(df_si_sampled, df_vd, on=[store_id], how='inner')
    print("df_si_sampled after  merging:\n", df_si_sampled)

    # Delete multiple columns from the dataframe
    df_si_sampled = df_si_sampled.drop(["genre_name", "area_name", "store_day_count", "store_percentage"], axis=1)
    # sys.exit()
    return df_si_sampled, df_vd, df_si

# Method uses: Imputation for only one time series.
# input:
#   df - df_store_and_visit : dataframe contains the merge of all store and visits,
#       which is used as input to create distance matrix of all time series
#   sid: store id
#   dr_idx: date range index
#   method: method for data imputation
#   column: column for imputing
#   j:
def imputing_one_timeseries(df, sid, dr_idx, method, column, j):
    # print("Input of imputing_one_timeserie function:111\n", df)
    # print("Imputing one timeseries:", sid)
    series = df.loc[df[store_id] == sid]

    series = series.set_index('visit_date')
    series.index = pd.DatetimeIndex(series.index)
    # print("imputing_one_timeserie - series 111:\n", series)
    series = series.reindex(dr_idx)
    # print("imputing_one_timeserie - series 222:\n", series)

    if method == 'mean':
        # print("series['visitors'] before:\n",series['visitors'])
        series[column] = series[column].fillna(series[column].mean())
        # roundup and convert to int
        series['visitors'] = series['visitors'].apply(lambda x: round(x, 0)).astype(int)
        # print("series['visitors'] after:\n",series['visitors'])
    elif method == 'median':
        # Median values imputation method
        series['visitors'] = series['visitors'].fillna(series['visitors'].median())
    elif method == 'linear':
        # Please note that only method='linear' is supported for DataFrame
        # print("series['visitors'] before:\n", series['visitors'])
        upsampled = series['visitors']
        interpolated = upsampled.interpolate(method='linear', limit=None, limit_direction='both')
        series['visitors'] = interpolated
        series['visitors'] = series['visitors'].astype(np.int64)
        # print("series['visitors'] after:\n", series['visitors'])

    else:
        # By choosing method, we can conclude that, dataset can be filled in order:
        # 1. The method
        # 2. forward fill
        # 3. backward fill
        # Method list: time, index, values, nearest, zero, slinear, cubic, barycentric, krogh, polynomial, spline
        # piecewise_polynomial, from_derivatives, pchip, akima
        # ‘ffill’ stands for ‘forward fill’ and will propagate last valid observation forward.
        # Pandas dataframe.bfill() is used to backward fill the missing values in the dataset.
        # print("series['visitors'] before:\n", series['visitors'])
        series['visitors'] = series['visitors'].interpolate(method=method, order=1, limit=None, limit_direction='both').ffill().bfill()
        series['visitors'] = series['visitors'].astype(np.int64)
        # print("series['visitors'] after:\n", series['visitors'])

    # series['visitors'] = series['visitors'].astype(np.int64)
    # series.drop([store_id], axis=1)
    # drop store_id column
    series = series.drop(store_id, 1)
    # rename the column with the id of the store id("core_samples_mask=== :", core_samples_mask)
    str1 = str(sid)
    # str1 = str1 + ' ' + str(j) +' visitors'
    # print("j value:", j)
    series.rename(columns={'visitors':str1}, inplace=True)
    # print("imputing_one_timeserie - series 333:\n", series)
    return series

# # Method uses: Imputation for only one time series.
# # input:
# #   df - df_store_and_visit : dataframe contains the merge of all store and visits,
# #       which is used as input to create distance matrix of all time series
# #   sid: store id
# #   dr_idx: date range index
# #   method: method for data imputation
# #   column: column for imputing
# #   j:
# def tranposing_one_timeseries(df, sid, dr_idx, j):
#     # print("Input of imputing_one_timeserie function:111\n", df)
#     # print("Imputing one timeseries:", sid)
#     series = df.loc[df[store_id] == sid]
#
#     series = series.set_index('visit_date')
#     series.index = pd.DatetimeIndex(series.index)
#     print("imputing_one_timeserie - series 111:\n", series)
#     series = series.reindex(dr_idx)
#     print("imputing_one_timeserie - series 222:\n", series)
#
#     sys.exit()
#     # # series['visitors'] = series['visitors'].astype(np.int64)
#     # # series.drop([store_id], axis=1)
#     # # drop store_id column
#     # series = series.drop(store_id, 1)
#     # # rename the column with the id of the store id("core_samples_mask=== :", core_samples_mask)
#     # str1 = str(sid)
#     # # str1 = str1 + ' ' + str(j) +' visitors'
#     # # print("j value:", j)
#     # series.rename(columns={'visitors':str1}, inplace=True)
#     # # print("imputing_one_timeserie - series 333:\n", series)
#     return series

# Method used: Get values of all time series as matrix values
# input:
#   visitor_matrix_transposed - Matrix of store id and their visitors after transposed with columns and rows name
# output:
#   matrix_values: Contain values of the input matrix
def split_matrix_values(vmf):
    print("vmf:\n", vmf)
    # first_column_values = vmf.iloc[:, 0:1].values
    # matrix_values = vmf.iloc[:, 1:].values

    first_column_values = vmf.index
    matrix_values = vmf.iloc[:, 0:].values

    # print("First column values:\n", first_column_values)
    print("Stacking all timeseries and use it as distance matrix values:\n", matrix_values)
    return first_column_values, matrix_values

# Create new visitor dataframe
# input:
# df - df_store_and_visit : dataframe contains the merge of all store and visits,
#   which is used as input to create distance matrix of all time series
#   dr_idx            : date range of for the index
#   method            : method for imputation -- we can use a lot of other types of method -- mean, ...
# output:
#   list_removed_timeseries_index: list of timeseries which are removed for high missing percentage rate
def imputing_all_timeseries(df_store_and_visit, method):
    # Find date rang index from average first moment and average last moment
    # Get the first_moment and last_moment dataframe of all stores
    first_moment, last_moment = first_last_moments(df_store_and_visit)
    # print("first_moment dataframe:\n", first_moment)
    # print("last_moment  dataframe:\n", last_moment)

    # Get the min date, max date from first moment and last moment then forming date range index
    min_date, max_date = min_max_first_last_moments(first_moment, last_moment)
    date_range_idx = pd.date_range(min_date, max_date)

    # print("Dataframe before imputed - df_store_and_visit:\n", df_store_and_visit)

    store_id_list = df_store_and_visit[store_id].tolist()
    # Removing duplicate values
    store_id_list = list(set(store_id_list))
    print('Length of Store id list of input dataframe:', len(store_id_list))
    store_id_list = np.asarray(store_id_list)

    # Create an empty dataframe in creating new dataframe by adding one by one of imputed time series
    all_imputed_timeseries = pd.DataFrame()
    df_standard_range = pd.DataFrame(index=date_range_idx)
    # Create a list of one-timeseries dataframe. Then I concatenate all of them.
    list_timeseries = []
    for i, j in zip(store_id_list, range(len(store_id_list))):
        # print("List item:", i)
        # Proceeding data imputation for each of timeseries by the chosen imputation method
        one_imputed_timeseries = imputing_one_timeseries(df_store_and_visit, i, date_range_idx, method, 'visitors', j)
        # Concatenate a list of of series to form up an dataframe
        all_imputed_timeseries = pd.concat([all_imputed_timeseries, one_imputed_timeseries], axis=1)
        list_timeseries.append(one_imputed_timeseries)

    # Transpose the result
    all_imputed_timeseries_transposed = all_imputed_timeseries.transpose()
    print("Dataframe after imputed and transposed - all_imputed_timeseries_transposed:\n", all_imputed_timeseries_transposed)
    # sys.exit()
    return all_imputed_timeseries_transposed






# # ============ Step 10: Find the corelation between genres and clusters ==============
# Method use: Find the corelation between genres and clusters
# input:
#   timeseries_hierachy_clustered      : dataframe contains 3 columns of store id, genres, locations and clusters label
# output:
#   df      : dataframe contains corelation between genre groups and clusters
# def corelation_genre_clusters(df):
#     # print("Input dataframe with clustered time series:\n", df)
#
#     df_temp = df
#
#     # Get first column to form up corelation dataframe
#     df_first_col = df_temp.groupby([genre_name]).size()
#     df_first_col = df_first_col.to_frame(name = 'size').reset_index()
#     # print("df_first_col: \n", df_first_col)
#
#     # Create a new empty corelation dataframe
#     df_corelation_genre_clusters = pd.DataFrame()
#     # Concatenate the new empty corelation dataframe with first column dataframe
#     # df_corelation_genre_clusters = pd.concat([df_corelation_genre_clusters, df_first_col], axis=1)
#
#     # Loop from the first cluster column to the end
#     max_col = df.shape[1]
#     for i in range(3, max_col):
#         # Group by genre name and cluster columns
#         df = df_temp.groupby([genre_name, df_temp.columns[i]]).size()
#         print("df====== 222\n:", df)
#         # reset index column
#         df = df.to_frame(name = 'size').reset_index()
#         print("df_merge_id:\n", df)
#         colname = df.columns[1]
#         print("colname = df.columns[pos]:", colname)
#         df_genre_name_cluster = df.groupby([colname, 'genre_name']).agg({'size': 'sum'})
#         print("df_genre_name_cluster:\n", df_genre_name_cluster)
#
#         # Calculate percentage for each genres by cluster
#         cluster_percentages = df_genre_name_cluster.groupby(level=0).apply(lambda x:
#                                                  100 * x / float(x.sum()))
#         cluster_percentages = cluster_percentages.reset_index()
#         print("cluster_percentages:\n", cluster_percentages)
#         cluster_percentages = cluster_percentages.round({'size': 1}).fillna(0)
#
#         # Pivot table and fill nan value
#         cluster_percentages_pivot = cluster_percentages.pivot_table(index=colname, columns='genre_name', values='size', aggfunc='max')
#         cluster_percentages_pivot = cluster_percentages_pivot.fillna(0)
#         cluster_percentages_pivot = cluster_percentages_pivot.reset_index()
#         # Get list of column names level 0
#         # cluster_percentages_pivot = cluster_percentages_pivot.reset_index(drop=True,level=0)
#         print("cluster_percentages after using pivot:\n", cluster_percentages_pivot)
#
#
#         # # Groupby genre again to get maximum size of appearances clusters
#         # idx = df.groupby([genre_name])['size'].transform(max) == df['size']
#         # df = df[idx]
#         # print("df====== 333:\n", df)
#         #
#         # str_column_name = df.columns[1]
#         # # print("str_column_name:", str_column_name)
#         #
#         # df = df.groupby(genre_name)[str_column_name].apply(lambda x: ','.join(map(str, x))).reset_index()
#         # print("df====== 444 : \n", df)
#         # df = df[[str_column_name]]
#         # print("df====== 555 : \n", df)
#
#         # Concatenate each result dataframe from each hierachy clustering arguments group
#         df_corelation_genre_clusters = pd.concat([df_corelation_genre_clusters, cluster_percentages_pivot], axis=1)
#
#     # Concatenate a list of dataframes
#     df_corelation_genre_clusters= df_corelation_genre_clusters.reset_index()
#
#     return df_corelation_genre_clusters

# Method: Adding 4 more pivot columns
# input:
#   df_clustered: This dataframe contains all possible columns to calculate pivot table
# output:
#   df_clustered_pivot: Return new dataframe contains pivot table which has 4 flatted columns
#       pivot_labels, pivot_..._1st, pivot_..._2nd, pivot_..._3rd
def corelation_genre_clusters(df_clustered):
    # print("df_clustered:", df_clustered)
    # Iterating row by row
    for i, row in df_clustered.iterrows():
        # Create a new dataframe with first columns of X is store_id and it's labels
        X_first_column = df_clustered.iloc[i]['X_first_column']
        labels = df_clustered.iloc[i]['labels']
        labels = convert_labels_to_dataframe(labels)
        X_first_column = convert_Xfirstcol_to_dataframe(X_first_column)
        df_clustered_refined = pd.concat([X_first_column, labels], axis=1)

        # Get store_info_dataset and visit date in preparing further
        RES_DATASET_ARG = df_clustered.iloc[i]['RES_DATASET_ARG']
        store_info_dataset, df_vd = get_storeinfo_visitdata(df_asi, df_avd, df_hsi, df_hr, RES_DATASET_ARG)

        print("df_clustered_refined:\n", df_clustered_refined)
        print("store_info_dataset:\n", store_info_dataset)
        print("df_vd:\n", store_info_dataset)

        # Merging with store_info_dataset to get genre name and area name
        df_clustered_refined = pd.merge(store_info_dataset, df_clustered_refined, how='inner', on=[store_id])

        # Get only the first word of area_name
        df_clustered_refined = format_arename_col_first_word(df_clustered_refined).reset_index(drop=True)
        print("Final dataframe of time series and their clusters:\n", df_clustered_refined)

        # Create 2 new dataframe ass temporaries for upcoming
        df_temp = df_clustered_refined
        df = df_clustered_refined
        # Choosing between 2 type of split by genre or area
        if df_clustered.iloc[i]['SPLIT_FIRST_BY_ARG'] == 'genre':
            group_type = genre_name
        else: # groupby area name
            group_type = area_name

        # Get first column to form up corelation dataframe
        df_first_col = df_temp.groupby([group_type]).size()
        df_first_col = df_first_col.to_frame(name = 'size').reset_index()
        print("df_first_col: \n", df_first_col)

        # Create a new empty corelation dataframe
        df_corelation_genre_clusters = pd.DataFrame()
        # Concatenate the new empty corelation dataframe with first column dataframe
        df_corelation_genre_clusters = pd.concat([df_corelation_genre_clusters, df_first_col], axis=1)

        # Loop from the first cluster column to the end
        # print("df.shape[1] which contain number of columns", df.shape[1])
        max_col = df.shape[1] - 1
        # We group it by group_type and last column number 3.
        # We create pivot table by these columns.
        print("df====== 111:\n", df)

        # Group by genre name or area name and cluster columns
        df = df_temp.groupby([group_type, df_temp.columns[max_col]]).size()
        print("df_temp.columns[j]:\n", df_temp.columns[max_col])
        # reset index column
        df = df.to_frame(name = 'size').reset_index()
        print("df_merge_id:\n", df)
        colname = df.columns[1]
        print("colname = df.columns[pos]:", colname)
        df_genre_name_cluster = df.groupby([colname, group_type]).agg({'size': 'sum'})
        print("df_genre_name_cluster:\n", df_genre_name_cluster)

        # Calculate percentage for each genres by cluster
        cluster_percentages = df_genre_name_cluster.groupby(level=0).apply(lambda x:
                                                 100 * x / float(x.sum()))
        cluster_percentages = cluster_percentages.reset_index()
        print("cluster_percentages:\n", cluster_percentages)
        cluster_percentages = cluster_percentages.round({'size': 0}).fillna(0)

        # Pivot table and fill nan value
        cluster_percentages_pivot = cluster_percentages.pivot_table(index=colname, columns=group_type, values='size', aggfunc='max')
        cluster_percentages_pivot = cluster_percentages_pivot.fillna(0)
        cluster_percentages_pivot = cluster_percentages_pivot.reset_index()
        # Get list of column names level 0
        # cluster_percentages_pivot = cluster_percentages_pivot.reset_index(drop=True,level=0)

        print("cluster_percentages after using pivot:", "\n", cluster_percentages_pivot)
        print("list(my_dataframe.columns.values):", "\n", list(cluster_percentages_pivot.columns.values))
        p_labels,group_type_1st,group_type_2nd,group_type_3rd = list(cluster_percentages_pivot.columns.values)

        print("cluster_percentages_pivot[p_labels].values.tolist()      ", cluster_percentages_pivot[p_labels].values.tolist())
        print("cluster_percentages_pivot[group_type_1st].values.tolist()", cluster_percentages_pivot[group_type_1st].values.tolist().astype(int))
        print("cluster_percentages_pivot[group_type_2nd].values.tolist()", cluster_percentages_pivot[group_type_2nd].values.tolist().astype(int))
        print("cluster_percentages_pivot[group_type_3rd].values.tolist()", cluster_percentages_pivot[group_type_3rd].values.tolist().astype(int))
        s = cluster_percentages_pivot[group_type_1st].values.tolist()


        df_clustered.at[i, 'pivot_'+p_labels] = listToString(cluster_percentages_pivot[p_labels].values.tolist())
        df_clustered.at[i, 'pivot_'+group_type_1st] = listToString(cluster_percentages_pivot[group_type_1st].values.tolist())
        df_clustered.at[i, 'pivot_'+group_type_2nd] = listToString(cluster_percentages_pivot[group_type_2nd].values.tolist())
        df_clustered.at[i, 'pivot_'+group_type_3rd] = listToString(cluster_percentages_pivot[group_type_3rd].values.tolist())

    # print("df_clustered:", "\n", df_clustered)
    # df_clustered.to_csv("ttess.csv")
    df_clustered_pivot = df_clustered
    return df_clustered_pivot

# method: do the clustering for missing values of time series for 5 algorithms
# input:
#   df_imputation: dataframe contains first level of arguments:
#       X_first_column, X ALGORITHMS_ARG, RES_DATASET_ARG, SPLIT_FIRST_BY_ARG, RESAMPLING_METHOD_ARG,
#       IMPUTATION_METHOD_ARG, MAX_MISSING_PERCENTAGE_ARG
# output:
#   df_imputation_clustered: Dataframe contains imputation from multiple algorithms: dbscan, hierachy, .., .., ..
def missing_values_clustering(df_imputation):
    print("df_imputation_level input missing_values_clustering:\n", df_imputation)

    imputation_dbscan_index   = 0
    list_cols = list(df_imputation.columns.values)
    list_cols.extend(['METRIC_ARG', 'EPSILON_MIN_ARG', 'EPSILON_MAX_ARG', 'EPSILON_STEP_ARG', 'MINS_ARG'])
    df_imputation_dbscan = pd.DataFrame(columns=list_cols)

    imputation_optics_index   = 0
    list_cols_optics = list(df_imputation.columns.values)
    list_cols_optics.extend(['METRIC_ARG', 'MIN_SAMPLES_ARG', 'XI_ARG', 'MIN_CLUSTER_SIZE_ARG'])
    df_imputation_optics = pd.DataFrame(columns=list_cols_optics)

    imputation_hierachy_index = 0
    list_cols_hierachy = list(df_imputation.columns.values)
    list_cols_hierachy.extend(['NUM_OF_HC_CLUSTER', 'LINKAGE', 'AFFINITY'])
    df_imputation_hierachy = pd.DataFrame(columns=list_cols_hierachy)

    imputation_kmeans_index   = 0
    list_cols_kmeans = list(df_imputation.columns.values)
    list_cols_kmeans.extend(['NUM_CLUSTERS_ARG', 'ITERATIONS_ARG', 'WINDOW_SIZE_ARG'])
    df_imputation_kmeans = pd.DataFrame(columns=list_cols_kmeans)

    imputation_kmeans_auto_index   = 0
    list_cols_kmeans_auto = list(df_imputation.columns.values)
    list_cols_kmeans_auto.extend(['NUM_CLUSTERS_ARG', 'ITERATIONS_ARG'])
    df_imputation_kmeans_auto = pd.DataFrame(columns=list_cols_kmeans_auto)

    for i, row in df_imputation.iterrows():
        # Creating dataframe of arguments of Dbscan
        if df_imputation.iloc[i]['ALGORITHMS_ARG'] == 'DBSCAN':
            METRIC = ['euclidean', 'manhattan', 'l2', 'canberra', 'hamming',  'sqeuclidean']
            # METRIC = ['dice', 'hamming', 'jaccard', 'kulsinski']
            # 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath',
            # 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
            #  'sokalsneath', 'sqeuclidean', 'yule']
            EPSILON_MIN = [5]
            EPSILON_MAX = [16000]
            EPSILON_STEP = [10]
            MINS = [3]
            for index, tuple in enumerate(itertools.product(METRIC, EPSILON_MIN, EPSILON_MAX, EPSILON_STEP, MINS)):
                # print("tuple [", index,  "]:", tuple)
                METRIC_ARG, EPSILON_MIN_ARG, EPSILON_MAX_ARG, EPSILON_STEP_ARG, MINS_ARG = tuple
                df_imputation_dbscan.loc[imputation_dbscan_index] = [df_imputation.iloc[i]['X_first_column']] + [df_imputation.iloc[i]['X']]\
                                                + [df_imputation.iloc[i]['ALGORITHMS_ARG']] + [df_imputation.iloc[i]['RES_DATASET_ARG']]\
                                                + [df_imputation.iloc[i]['SPLIT_FIRST_BY_ARG']] + [df_imputation.iloc[i]['RESAMPLING_METHOD_ARG']]\
                                                + [df_imputation.iloc[i]['IMPUTATION_METHOD_ARG']] + [df_imputation.iloc[i]['MAX_MISSING_PERCENTAGE_ARG']]\
                                                + [METRIC_ARG] + [EPSILON_MIN_ARG] + [EPSILON_MAX_ARG] + [EPSILON_STEP_ARG] + [MINS_ARG]

                # print("imputation_dbscan_index:", imputation_dbscan_index)
                # print("df_imputation_dbscan.iloc[imputation_dbscan_index]['X']:\n", df_imputation_dbscan.iloc[imputation_dbscan_index]['X'])
                imputation_dbscan_index = imputation_dbscan_index + 1

        # Creating dataframe of arguments of Dbscan
        elif df_imputation.iloc[i]['ALGORITHMS_ARG'] == 'OPTICS':
            # METRIC = ['euclidean', 'manhattan']
            METRIC = ['DTWDistance']
            # METRIC = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
            # ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard',
            # 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
            # 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

            MIN_SAMPLES = [3]
            XI = [0.01] # Determines the minimum steepness on the reachability plot that constitutes a cluster boundary.
            MIN_CLUSTER_SIZE = [0.01]
            for index, tuple in enumerate(itertools.product(METRIC, MIN_SAMPLES, XI, MIN_CLUSTER_SIZE)):
                METRIC_ARG, MIN_SAMPLES_ARG, XI_ARG, MIN_CLUSTER_SIZE_ARG = tuple
                df_imputation_optics.loc[imputation_optics_index] = [df_imputation.iloc[i]['X_first_column']] + [df_imputation.iloc[i]['X']]\
                                                + [df_imputation.iloc[i]['ALGORITHMS_ARG']] + [df_imputation.iloc[i]['RES_DATASET_ARG']]\
                                                + [df_imputation.iloc[i]['SPLIT_FIRST_BY_ARG']] + [df_imputation.iloc[i]['RESAMPLING_METHOD_ARG']]\
                                                + [df_imputation.iloc[i]['IMPUTATION_METHOD_ARG']] + [df_imputation.iloc[i]['MAX_MISSING_PERCENTAGE_ARG']]\
                                                + [METRIC_ARG] + [MIN_SAMPLES_ARG] + [XI_ARG] + [MIN_CLUSTER_SIZE_ARG]

                print("imputation_optics_index:", imputation_optics_index)
                print("df_imputation_optics.iloc[imputation_optics_index]['X']:\n", df_imputation_optics.iloc[imputation_optics_index]['X'])
                imputation_optics_index = imputation_optics_index + 1

        elif df_imputation.iloc[i]['ALGORITHMS_ARG'] == 'HIERACHY':
            NUM_OF_HC_CLUSTER = [3, 9]
            LINKAGE = ['complete', 'average', 'single']
            AFFINITY = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
            LINKAGE_EXTENDED = ['ward']
            AFFINITY_EXTENDED = ['euclidean']

            for index, tuple in enumerate(itertools.product(NUM_OF_HC_CLUSTER, LINKAGE, AFFINITY)):
                NUM_OF_HC_CLUSTER_ARG, LINKAGE_ARG, AFFINITY_ARG = tuple
                df_imputation_hierachy.loc[imputation_hierachy_index] = [df_imputation.iloc[i]['X_first_column']] + [df_imputation.iloc[i]['X']]\
                                                    + [df_imputation.iloc[i]['ALGORITHMS_ARG']] + [df_imputation.iloc[i]['RES_DATASET_ARG']]\
                                                    + [df_imputation.iloc[i]['SPLIT_FIRST_BY_ARG']] + [df_imputation.iloc[i]['RESAMPLING_METHOD_ARG']]\
                                                    + [df_imputation.iloc[i]['IMPUTATION_METHOD_ARG']] + [df_imputation.iloc[i]['MAX_MISSING_PERCENTAGE_ARG']]\
                                                    + [NUM_OF_HC_CLUSTER_ARG] + [LINKAGE_ARG] + [AFFINITY_ARG]
                imputation_hierachy_index = imputation_hierachy_index + 1

            for index, tuple in enumerate(itertools.product(NUM_OF_HC_CLUSTER, LINKAGE_EXTENDED, AFFINITY_EXTENDED)):
                NUM_OF_HC_CLUSTER_ARG, LINKAGE_ARG, AFFINITY_ARG = tuple
                df_imputation_hierachy.loc[imputation_hierachy_index] = [df_imputation.iloc[i]['X_first_column']] + [df_imputation.iloc[i]['X']]\
                                                    + [df_imputation.iloc[i]['ALGORITHMS_ARG']] + [df_imputation.iloc[i]['RES_DATASET_ARG']]\
                                                    + [df_imputation.iloc[i]['SPLIT_FIRST_BY_ARG']] + [df_imputation.iloc[i]['RESAMPLING_METHOD_ARG']]\
                                                    + [df_imputation.iloc[i]['IMPUTATION_METHOD_ARG']] + [df_imputation.iloc[i]['MAX_MISSING_PERCENTAGE_ARG']]\
                                                    + [NUM_OF_HC_CLUSTER_ARG] + [LINKAGE_ARG] + [AFFINITY_ARG]
                imputation_hierachy_index = imputation_hierachy_index + 1

        elif df_imputation.iloc[i]['ALGORITHMS_ARG'] == 'KMEANS_DTW':
            NUM_CLUSTERS = [3, 9]
            ITERATIONS = [15]
            WINDOW_SIZE = [6, 9]
            # NUM_CLUSTERS = [3]
            # ITERATIONS = [10]
            # WINDOW_SIZE = [3]

            for index, tuple in enumerate(itertools.product(NUM_CLUSTERS, ITERATIONS, WINDOW_SIZE)):
                # print("tuple [", index,  "]:", tuple)
                NUM_CLUSTERS_ARG, ITERATIONS_ARG, WINDOW_SIZE_ARG = tuple
                df_imputation_kmeans.loc[imputation_kmeans_index] = [df_imputation.iloc[i]['X_first_column']] + [df_imputation.iloc[i]['X']]\
                                                + [df_imputation.iloc[i]['ALGORITHMS_ARG']] + [df_imputation.iloc[i]['RES_DATASET_ARG']]\
                                                + [df_imputation.iloc[i]['SPLIT_FIRST_BY_ARG']] + [df_imputation.iloc[i]['RESAMPLING_METHOD_ARG']]\
                                                + [df_imputation.iloc[i]['IMPUTATION_METHOD_ARG']] + [df_imputation.iloc[i]['MAX_MISSING_PERCENTAGE_ARG']]\
                                                + [NUM_CLUSTERS_ARG] + [ITERATIONS_ARG] + [WINDOW_SIZE_ARG]

                print("imputation_kmeans_index:", imputation_kmeans_index)
                print("df_imputation_kmeans.iloc[imputation_kmeans_index]['X']:\n", df_imputation_kmeans.iloc[imputation_kmeans_index]['X'])
                imputation_kmeans_index = imputation_kmeans_index + 1

        elif df_imputation.iloc[i]['ALGORITHMS_ARG'] == 'KMEANS_AUTO':
            NUM_CLUSTERS = [3]
            ITERATIONS = [100]
            # NUM_CLUSTERS = [3]
            # ITERATIONS = [10]

            for index, tuple in enumerate(itertools.product(NUM_CLUSTERS, ITERATIONS)):
                # print("tuple [", index,  "]:", tuple)
                NUM_CLUSTERS_ARG, ITERATIONS_ARG = tuple
                df_imputation_kmeans_auto.loc[imputation_kmeans_auto_index] = [df_imputation.iloc[i]['X_first_column']] + [df_imputation.iloc[i]['X']]\
                                                + [df_imputation.iloc[i]['ALGORITHMS_ARG']] + [df_imputation.iloc[i]['RES_DATASET_ARG']]\
                                                + [df_imputation.iloc[i]['SPLIT_FIRST_BY_ARG']] + [df_imputation.iloc[i]['RESAMPLING_METHOD_ARG']]\
                                                + [df_imputation.iloc[i]['IMPUTATION_METHOD_ARG']] + [df_imputation.iloc[i]['MAX_MISSING_PERCENTAGE_ARG']]\
                                                + [NUM_CLUSTERS_ARG] + [ITERATIONS_ARG]

                print("imputation_kmeans_auto_index:", imputation_kmeans_auto_index)
                print("df_imputation_kmeans.iloc[imputation_kmeans_auto_index]['X']:\n", df_imputation_kmeans_auto.iloc[imputation_kmeans_auto_index]['X'])
                imputation_kmeans_auto_index = imputation_kmeans_auto_index + 1

    # Get clustered from dbscan argumented dataframe
    df_imputation_dbscan_clustered = clustering_by_dbscan(df_imputation_dbscan)
    # print("DBSCAN ALGORITHMS:\n", df_imputation_dbscan_clustered)

    # Get clustered from optics argumented dataframe
    df_imputation_optics_clustered = clustering_by_optics(df_imputation_optics)
    # print("OPTICS ALGORITHMS:\n", df_imputation_optics_clustered)

    # Get clustered from hierachy argumented dataframe
    df_imputation_hierachy_clustered = clustering_by_hierachy(df_imputation_hierachy)
    # print("HIERACHY ALGORITHMS", df_imputation_hierachy_clustered)

    # Get clustered from kmeans DTW argumented dataframe
    df_imputation_kmeans_clustered = clustering_by_kmeans(df_imputation_kmeans)
    # print("KMEANS ALGORITHMS\n", df_imputation_kmeans_clustered)

    # Get clustered from kmeans auto argumented dataframe
    df_imputation_kmeans_auto_clustered = clustering_by_kmeans_auto(df_imputation_kmeans_auto)
    # print("KMEANS ALGORITHMS\n", df_imputation_kmeans_auto_clustered)

    # # This action will mergering in columns and in rows of multiple dataframes.
    # # It will create new columns of this dataframe and add it to another one.
    # df_all_algo_clustered = [df_imputation_dbscan_clustered, df_imputation_hierachy_clustered]
    # df_all_algo_clustered = pd.concat(df_all_algo_clustered).reset_index(drop=True)

    return df_imputation_dbscan_clustered, df_imputation_optics_clustered, df_imputation_hierachy_clustered, df_imputation_kmeans_clustered, df_imputation_kmeans_auto_clustered

# method: get imputation matrix, which is preparing for clustering
# input:
# output:
#   df_imputation_level: contain imputation level of input dataset arguments
#                         ['X_first_column', 'ALGORITHMS_ARG', 'RES_DATASET_ARG', 'SPLIT_FIRST_BY_ARG',
#                         'RESAMPLING_METHOD_ARG', 'IMPUTATION_METHOD_ARG', 'MAX_MISSING_PERCENTAGE_ARG'])
def get_df_imputation_level(ALGORITHMS, RES_DATASET, SPLIT_FIRST_BY, RESAMPLING_METHOD,
                                  IMPUTATION_METHOD, MAX_MISSING_PERCENTAGE):
    df_imputation = pd.DataFrame(columns=['X_first_column', 'X', 'ALGORITHMS_ARG', 'RES_DATASET_ARG', 'SPLIT_FIRST_BY_ARG',
                                              'RESAMPLING_METHOD_ARG', 'IMPUTATION_METHOD_ARG', 'MAX_MISSING_PERCENTAGE_ARG'])
    for index, tuple in enumerate(itertools.product(ALGORITHMS, RES_DATASET, SPLIT_FIRST_BY, RESAMPLING_METHOD,
                                  IMPUTATION_METHOD, MAX_MISSING_PERCENTAGE)):
        print("=================================== START ======================================================")
        print("tuple [", index,  "]:", tuple)
        str_tuple = "tuple [", index,  "]:", tuple

        ALGORITHMS_ARG, RES_DATASET_ARG, SPLIT_FIRST_BY_ARG, RESAMPLING_METHOD_ARG, \
        IMPUTATION_METHOD_ARG, MAX_MISSING_PERCENTAGE_ARG = tuple

        df_attrib_1_attrib_2, df_vd, df_si = filtering_splitting_merging(df_asi, df_avd, df_hsi, df_hr,
                                RES_DATASET_ARG, MAX_MISSING_PERCENTAGE_ARG,
                                SPLIT_FIRST_BY_ARG, RESAMPLING_METHOD_ARG)

        visitor_matrix_transposed = imputing_all_timeseries(df_attrib_1_attrib_2, IMPUTATION_METHOD_ARG)
        X_first_column, X = split_matrix_values(visitor_matrix_transposed)
        print("visitor_matrix_transposed:=====================\n", visitor_matrix_transposed)
        # print("X_first_column:\n", X_first_column)
        # print("type of X_first_column:\n", type(X_first_column))
        print("X:\n", X)
        print("X shape:\n", X.shape)
        # store_info_dataset, df_vd = get_storeinfo_visitdata(df_asi, df_avd, df_hsi, df_hr, RES_DATASET_ARG)

        df_imputation.loc[index] = [X_first_column] + [X] + [ALGORITHMS_ARG] + [RES_DATASET_ARG] + [SPLIT_FIRST_BY_ARG] + \
                                       [RESAMPLING_METHOD_ARG] + [IMPUTATION_METHOD_ARG] + [MAX_MISSING_PERCENTAGE_ARG]
        print("df_imputation_level 2:\n", df_imputation)
        # sys.exit()
    return df_imputation, df_vd, df_si


# # ============ Step 10: Find the corelation between genres and clusters ==============

# Python program to convert a list
# to string using join() function
# Function to convert
def listToString(s):
    listToStr = ' '.join([str(elem) for elem in s])
    return listToStr

def convert_labels_to_dataframe(labels):
    # print("labels 000000000:\n", labels)
    # print("labels 111111111:\n", labels)
    labels = pd.DataFrame(data=labels)
    labels = labels.astype(int)
    labels.rename(columns = {0:'labels'}, inplace=True)
    # print("labels 222222222:\n", labels)
    return labels

# Method:  Remove bracket [] characters
def convert_Xfirstcol_to_dataframe(X_first_column):
    X_first_column = pd.DataFrame(X_first_column)
    X_first_column.rename(columns = {0:'store_id'}, inplace=True)
    # print("X_first_column convert_Xfirstcol_to_dataframe:\n", X_first_column)
    return X_first_column

# Method: Adding 4 more pivot columns
# input:
#   df_clustered: This dataframe contains all possible columns to calculate pivot table
# output:
#   df_clustered_pivot: Return new dataframe contains pivot table which has 4 flatted columns
#       pivot_labels, pivot_..._1st, pivot_..._2nd, pivot_..._3rd
def correlation_clustered_pivoting(df_clustered, df_vd, df_si):
    # print("df_clustered:", df_clustered)
    # Iterating row by row
    for i, row in df_clustered.iterrows():
        # Create a new dataframe with first columns of X is store_id and it's labels
        X_first_column = df_clustered.iloc[i]['X_first_column']
        labels = df_clustered.iloc[i]['labels']
        labels = convert_labels_to_dataframe(labels)
        X_first_column = convert_Xfirstcol_to_dataframe(X_first_column)
        df_clustered_refined = pd.concat([X_first_column, labels], axis=1)

        print("df_si input :\n", df_si)
        df_si = store_info_format(df_si)
        df_si[genre_name] = df_si[genre_name].str.strip()
        df_si[area_name]  = df_si[area_name].str.strip()
        # df_si.to_csv("df_si.csv")
        # print("df_vd before taking original:\n", df_vd)
        #
        # # Get store_info_dataset and visit date in preparing further
        # RES_DATASET_ARG = df_clustered.iloc[i]['RES_DATASET_ARG']
        # store_info_dataset, df_vd = get_storeinfo_visitdata(df_asi, df_avd, df_hsi, df_hr, RES_DATASET_ARG)
        #
        print("df_clustered_refined:\n", df_clustered_refined)
        # df_clustered_refined.to_csv("df_clustered_refined.csv")

        # print("store_info_dataset:\n", store_info_dataset)
        # print("df_vd:\n", df_vd)

        # for df in (df_clustered_refined, df_si):
        #     # Strip the column(s) you're planning to join with
        #     df[store_id] = df[store_id].str.strip()

        # df_si[store_id] = df_si[store_id].astype(str)

        # col_si = df_si[store_id].tolist()
        # col_clustered_refined = df_clustered_refined[store_id].tolist()
        # print("col_si:", col_si)
        # print("length of col_si:", len(col_si))
        # print("col_clustered_refined:", col_clustered_refined)
        # print("length of col_clustered_refined:", len(col_clustered_refined))
        # result =  all(elem in col_si  for elem in col_clustered_refined)
        # if result:
        #     print("Yes, col_si contains all elements in col_clustered_refined")
        # else :
        #     print("No, col_si does not contains all elements in col_clustered_refined")

        # Merging with store_info_dataset to get genre name and area name
        df_clustered_refined = pd.merge(df_si, df_clustered_refined, how='inner', on=[store_id])
        # df_clustered_refined.to_csv("df_clustered_refined_merged.csv")
        print("df_clustered_refined after merging:\n", df_clustered_refined)
        # sys.exit()

        # Get only the first word of area_name
        df_clustered_refined = format_arename_col_first_word(df_clustered_refined).reset_index(drop=True)
        print("Final dataframe of time series and their clusters:\n", df_clustered_refined)

        # Create 2 new dataframe as temporaries for upcoming
        df_temp = df_clustered_refined
        df = df_clustered_refined
        # Choosing between 2 type of split by genre or area
        if df_clustered.iloc[i]['SPLIT_FIRST_BY_ARG'] == 'genre':
            group_type = genre_name
        else: # groupby area name
            group_type = area_name

        # Get first column to form up corelation dataframe
        df_first_col = df_temp.groupby([group_type]).size()
        df_first_col = df_first_col.to_frame(name = 'size').reset_index()
        print("df_first_col: \n", df_first_col)

        # Create a new empty corelation dataframe
        df_corelation_genre_clusters = pd.DataFrame()
        # Concatenate the new empty corelation dataframe with first column dataframe
        df_corelation_genre_clusters = pd.concat([df_corelation_genre_clusters, df_first_col], axis=1)

        # Loop from the first cluster column to the end
        # print("df.shape[1] which contain number of columns", df.shape[1])
        max_col = df.shape[1] - 1
        # We group it by group_type and last column number 3.
        # We create pivot table by these columns.
        print("df====== 111:\n", df)

        # Group by genre name or area name and cluster columns
        df = df_temp.groupby([group_type, df_temp.columns[max_col]]).size()
        print("df_temp.columns[j]:\n", df_temp.columns[max_col])
        # reset index column
        df = df.to_frame(name = 'size').reset_index()
        print("df_merge_id:\n", df)
        colname = df.columns[1]
        print("colname = df.columns[pos]:", colname)
        df_genre_name_cluster = df.groupby([colname, group_type]).agg({'size': 'sum'})
        print("df_genre_name_cluster:\n", df_genre_name_cluster)

        # Calculate percentage for each genres by cluster
        cluster_percentages = df_genre_name_cluster.groupby(level=0).apply(lambda x:
                                                 100 * x / float(x.sum()))
        cluster_percentages = cluster_percentages.reset_index()
        print("cluster_percentages:\n", cluster_percentages)
        cluster_percentages = cluster_percentages.round({'size': 0}).fillna(0)

        # Pivot table and fill nan value
        cluster_percentages_pivot = cluster_percentages.pivot_table(index=colname, columns=group_type, values='size', aggfunc='max')
        cluster_percentages_pivot = cluster_percentages_pivot.fillna(0).astype(int)
        cluster_percentages_pivot = cluster_percentages_pivot.reset_index()
        # Get list of column names level 0
        # cluster_percentages_pivot = cluster_percentages_pivot.reset_index(drop=True,level=0)

        print("cluster_percentages after using pivot:", "\n", cluster_percentages_pivot)
        print("list(my_dataframe.columns.values):", "\n", list(cluster_percentages_pivot.columns.values))
        p_labels,group_type_1st,group_type_2nd,group_type_3rd = list(cluster_percentages_pivot.columns.values)

        print("cluster_percentages_pivot[p_labels].values.tolist()      ", cluster_percentages_pivot[p_labels].values.tolist())
        print("cluster_percentages_pivot[group_type_1st].values.tolist()", cluster_percentages_pivot[group_type_1st].values.tolist())
        print("cluster_percentages_pivot[group_type_2nd].values.tolist()", cluster_percentages_pivot[group_type_2nd].values.tolist())
        print("cluster_percentages_pivot[group_type_3rd].values.tolist()", cluster_percentages_pivot[group_type_3rd].values.tolist())
        s = cluster_percentages_pivot[group_type_1st].values.tolist()


        df_clustered.at[i, 'pivot_'+p_labels] = listToString(cluster_percentages_pivot[p_labels].values.tolist())
        df_clustered.at[i, 'pivot_'+group_type_1st] = listToString(cluster_percentages_pivot[group_type_1st].values.tolist())
        df_clustered.at[i, 'pivot_'+group_type_2nd] = listToString(cluster_percentages_pivot[group_type_2nd].values.tolist())
        df_clustered.at[i, 'pivot_'+group_type_3rd] = listToString(cluster_percentages_pivot[group_type_3rd].values.tolist())

    # print("df_clustered:", "\n", df_clustered)
    # df_clustered.to_csv("ttess.csv")
    df_clustered_pivot = df_clustered
    return df_clustered_pivot


df_imputation, df_vd, df_si = get_df_imputation_level(ALGORITHMS, RES_DATASET, SPLIT_FIRST_BY, RESAMPLING_METHOD,
                                  IMPUTATION_METHOD, MAX_MISSING_PERCENTAGE)

# Get clustered by algrorithms
df_imputation_dbscan_clustered, df_imputation_optics_clustered, df_imputation_hierachy_clustered, \
df_imputation_kmeans_clustered, df_imputation_kmeans_auto_clustered = missing_values_clustering(df_imputation)

# df_dbscan_clustered_pivot   = correlation_clustered_pivoting(df_imputation_dbscan_clustered)
# df_optics_clustered_pivot = correlation_clustered_pivoting(df_imputation_optics_clustered)
# df_hierachy_clustered_pivot = correlation_clustered_pivoting(df_imputation_hierachy_clustered)
# df_kmeans_clustered_pivot = correlation_clustered_pivoting(df_imputation_kmeans_clustered)
df_kmeans_auto_clustered_pivot = correlation_clustered_pivoting(df_imputation_kmeans_auto_clustered, df_vd, df_si)


# print("df_kmeans_clustered_pivot:\n", df_kmeans_auto_clustered_pivot)
print("df_optics_clustered_pivot:\n", df_kmeans_auto_clustered_pivot)

df_kmeans_auto_clustered_pivot.to_csv("df_kmeans_auto_clustered_pivot.csv")




