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

ALGORITHMS = ['DBSCAN']
ALGORITHMS_ARG = ['HIERACHY', 'DBSCAN']
RES_DATASET = ['air', 'hpg']
# RES_DATASET = ['air']
RES_DATASET_ARG = 'air'
RESAMPLING_METHOD = ['under']
RESAMPLING_METHOD_ARG = 'under'
# SPLIT_GROUPS = [3, 9]
SPLIT_FIRST_BY = ['genre', 'area']
SPLIT_FIRST_BY_ARG = 'area'
# split groups arguments always is 9
# SPLIT_GROUPS_ARG = 9
# IMPUTATION_METHOD = ['median', 'mean', 'linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'barycentric',
#           'krogh', 'polynomial', 'spline', 'piecewise_polynomial', 'from_derivatives', 'pchip', 'akima']
IMPUTATION_METHOD = ['median', 'mean']
IMPUTATION_METHOD_ARG = 'median'
MAX_MISSING_PERCENTAGE = [28]
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



def store_info_format(df):
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

# This function will format and group hpg reserver file
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

def get_3_equal_attrib_1_size_sampling(df, attrib_1_1st, attrib_1_2nd, attrib_1_3rd, attrib_1, RESAMPLING_METHOD_ARG):
    df_attrib_1_1st = df.loc[df[attrib_1].isin([attrib_1_1st])].reset_index(drop=True)
    df_attrib_1_2nd = df.loc[df[attrib_1].isin([attrib_1_2nd])].reset_index(drop=True)
    df_attrib_1_3rd = df.loc[df[attrib_1].isin([attrib_1_3rd])].reset_index(drop=True)

    if RESAMPLING_METHOD_ARG == 'under':
        size = df_attrib_1_3rd.shape[0]
        df_attrib_1_1st = under_sampling(df_attrib_1_1st, size)
        # Due to randomly choosing in undersampling method, location or area of those series will be shuffle as well.
        # This action will affect in choosing 3 largest areas in the next step
        print("df_attrib_1_1st:\n", df_attrib_1_1st)
        print("df_attrib_1_1st shape 0 :", df_attrib_1_1st.shape[0])

        df_attrib_1_2nd = under_sampling(df_attrib_1_2nd, size)
        print("df_attrib_1_2nd:\n", df_attrib_1_2nd)
        print("df_attrib_1_2nd shape:", df_attrib_1_2nd.shape[0])

    else:
    #     sampling method == 'over'
    #     size = df_attrib_1_1st.shape[0]
    #     df_attrib_1_2nd = over_sampling(df_attrib_1_2nd, size)
    #     df_attrib_1_3rd = over_sampling(df_attrib_1_3rd, size)
        print("Oversampling data: temporary not implemented due to diversity of clusters")

    frames_after_sampling = [df_attrib_1_1st, df_attrib_1_2nd, df_attrib_1_3rd]
    df_attrib_1 = pd.concat(frames_after_sampling).reset_index(drop=True)

    return df_attrib_1, df_attrib_1_1st, df_attrib_1_2nd, df_attrib_1_3rd


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
    # print("Store id dataset before merging - store_info_dataset - store_info_dataset:\n", store_info_dataset)
    # print("Store id dataset and visitor data before merging - df_vd:\n", df_vd)

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
def split_store_visit_2nd(df_merged, RES_DATASET_ARG, RESAMPLING_METHOD_ARG):
    print("df_merged:\n", df_merged)
    df_groups_store_id = df_merged
    df_groups_store_id = df_groups_store_id.drop(['visit_date', 'visitors'], axis=1)
    df_groups_store_id = df_groups_store_id.drop_duplicates().reset_index()
    print("df_groups_store_id:\n", df_groups_store_id)

    df_groups = df_groups_store_id.groupby(genre_name).count().sort_values(store_id, ascending=False).reset_index()
    print('=== LIST OUT ALL GENRE GROUPS ASCENDING df_groups: === \n', df_groups)

    store_info_dataset, df_vd = get_storeinfo_visitdata(RES_DATASET_ARG)

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
def split_to_9_groups(df, SPLIT_FIRST_BY_ARG, RESAMPLING_METHOD_ARG):
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
    df_attrib_1, df_attrib_1st, df_attrib_2nd, df_attrib_3rd = get_3_equal_attrib_1_size_sampling(df,
                        attrib_1_1st, attrib_1_2nd, attrib_1_3rd, attrib_1, RESAMPLING_METHOD_ARG)

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
    df_attrib_1_attrib_2 = pd.concat(frames).reset_index(drop=True)
    print("df_attrib_1_attrib_2:===\n", df_attrib_1_attrib_2)
    print('Total number of restaurants in 3 main ', attrib_1,' with 3 main', attrib_2,':', df_attrib_1_attrib_2.shape[0], '\n')
    return df_attrib_1, df_attrib_1_attrib_2

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
    store_info_dataset, df_vd = get_storeinfo_visitdata(df_asi, df_avd, df_hsi, df_hr, RES_DATASET_ARG)

    # Filtering visistor data from dataframe base on MAX_MISSING_PERCENTAGE_ARG
    df_vd_high_missing_filtered = filtering_high_missing_rate_store(df_vd, MAX_MISSING_PERCENTAGE_ARG)

    # Merging store info dataset and visit data to one dataframe
    df_merged = merge_store_visit_2nd(store_info_dataset, df_vd_high_missing_filtered)

    # Splitting store infodataset by genres or by area
    df_attrib_1, df_attrib_1_attrib_2 = split_to_9_groups(df_merged, SPLIT_FIRST_BY_ARG, RESAMPLING_METHOD_ARG)

    # Merging df_attrib_1, df_attrib_1_attrib_2 with df_vd which preparing for imputation step
    print("df_attrib_1_attrib_2 before merging:\n", df_attrib_1_attrib_2)
    print("df_vd:\n", df_vd)
    df_attrib_1 = pd.merge(df_attrib_1, df_vd, on=[store_id], how='inner')
    df_attrib_1_attrib_2 = pd.merge(df_attrib_1_attrib_2, df_vd, on=[store_id], how='inner')
    print("df_attrib_1_attrib_2 after  merging:\n", df_attrib_1_attrib_2)

    # Delete multiple columns from the dataframe
    df_attrib_1 = df_attrib_1.drop(["genre_name", "area_name", "store_day_count", "store_percentage"], axis=1)
    df_attrib_1_attrib_2 = df_attrib_1_attrib_2.drop(["genre_name", "area_name", "store_day_count", "store_percentage"], axis=1)

    return df_attrib_1, df_attrib_1_attrib_2

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

# Create new visitor dataframe
# input:
# df - df_store_and_visit : dataframe contains the merge of all store and visits,
#   which is used as input to create distance matrix of all time series
# dr_idx            : date range of for the index
# method            : method for imputation -- we can use a lot of other types of method -- mean, ...
# floor_percentage  : create new dataframe base only on series has missing percentage less than floor_percentage
# output:
# list_removed_timeseries_index: list of timeseries which are removed for high missing percentage rate
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
    return all_imputed_timeseries_transposed


df_attrib_1, df_attrib_1_attrib_2 = filtering_splitting_merging(df_asi, df_avd, df_hsi, df_hr,
                                RES_DATASET_ARG, MAX_MISSING_PERCENTAGE_ARG,
                                SPLIT_FIRST_BY_ARG, RESAMPLING_METHOD_ARG)





# method: do the clustering for missing values of time series
def missing_values_clustering(X, ALGORITHMS_ARG, RES_DATASET_ARG, RESAMPLING_METHOD_ARG, SPLIT_GROUPS_ARG,
                              IMPUTATION_METHOD_ARG, MAX_MISSING_PERCENTAGE_ARG):

    if ALGORITHMS_ARG == 'HIERACHY':
    #     timeseries_hierachy_clustered = clustering_by_hierachy(visitor_matrix_formatted_values, visitor_matrix_formatted_first_column,
    #                                                             store_info_dataset, NUM_OF_HC_CLUSTER_ARG)
    #     print("============================== ARGUMENTS LIST ======================================================")
    #     print("Alogrithm:", ALGORITHMS_ARG,"-", "Resource dataset:", RES_DATASET_ARG,"-", "Resampling method:", RESAMPLING_METHOD_ARG,"-",
    #           "Split to 3 or 9 groups:", SPLIT_GROUPS_ARG,"-",
    #           "Imputation method:", IMPUTATION_METHOD_ARG,"-", "Max missing percentage:", MAX_MISSING_PERCENTAGE_ARG)
    #
    #     df = corelation_genre_clusters(timeseries_hierachy_clustered)
        print("HIERACHY")

    elif ALGORITHMS_ARG == 'DBSCAN':
        METRIC = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
        # ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
        # 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
        EPSILON_MIN = [5]
        EPSILON_MAX = [15000]
        EPSILON_STEP = [10]
        MINS = [15]
        for index, tuple in enumerate(itertools.product(METRIC, EPSILON_MIN, EPSILON_MAX, EPSILON_STEP, MINS)):
            print("======================= START ", ALGORITHMS_ARG, " ALGORITHMS ===============================")
            print("tuple [", index,  "]:", tuple)
            str_tuple = "tuple [", index,  "]:", tuple
            METRIC_ARG, EPSILON_MIN_ARG, EPSILON_MAX_ARG, EPSILON_STEP_ARG, MINS_ARG = tuple
            clustering_by_dbscan(X, METRIC_ARG, EPSILON_MIN_ARG, EPSILON_MAX_ARG, EPSILON_STEP_ARG, MINS_ARG)
    return 1

for index, tuple in enumerate(itertools.product(ALGORITHMS, RES_DATASET, SPLIT_FIRST_BY, RESAMPLING_METHOD,
                              IMPUTATION_METHOD, MAX_MISSING_PERCENTAGE)):
    print("=================================== START ======================================================")
    print("tuple [", index,  "]:", tuple)
    str_tuple = "tuple [", index,  "]:", tuple

    ALGORITHMS_ARG, RES_DATASET_ARG, SPLIT_FIRST_BY_ARG, RESAMPLING_METHOD_ARG, \
    IMPUTATION_METHOD_ARG, MAX_MISSING_PERCENTAGE_ARG = tuple

    visitor_matrix_transposed = imputing_all_timeseries(df_attrib_1_attrib_2, IMPUTATION_METHOD_ARG)
    X = visitor_matrix_transposed.values

    missing_values_clustering(X, ALGORITHMS_ARG, RES_DATASET_ARG, SPLIT_FIRST_BY_ARG, RESAMPLING_METHOD_ARG,
                              IMPUTATION_METHOD_ARG, MAX_MISSING_PERCENTAGE_ARG)












