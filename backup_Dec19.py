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



# TypeError: unsupported operand type(s) for +: 'int' and 'str'
# print(ts1)

# e_distance = euclid_dist(ts1, ts2)
# print("e_distance:", e_distance)

# x=np.linspace(0,50,100)
# ts1=pd.Series(3.1*np.sin(x/1.5)+3.5)
# ts2=pd.Series(2.2*np.sin(x/3.5+2.4)+3.2)
# ts3=pd.Series(0.04*x+3.0)
#
# ts1.plot()
# ts2.plot()
# ts3.plot()
#
# plt.ylim(-2,10)
# plt.legend(['ts1','ts2','ts3'])
# plt.show()
# ============ Experiements ==============
# # print(data['hr'].shape)
# # Get only additional hpg_reserve data which have id as in air_reserve id by innner join
# print("data['hpg_reserve'].shape - before merge:", data['hr'].shape)
# data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])
# print("data['store_id_relation'].shape:", data['id'].shape)
# print("data['hpg_reserve'].shape - after merge:", data['hr'].shape)
# print("data['hpg_reserve'].tail {nl}:", data['hr'].tail())
#
# # Removing hpg_store_id column in the new
# data['hr'].drop('hpg_store_id', axis=1, inplace=True)
# print("data['hpg_reserve'].tail {nl}:", data['hr'].tail())
#
# for df in ['ar','hr']:
#     data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
#     data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
#     data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
#     data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
#     # data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
#     # tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors':'rv1'})
#     # tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors':'rv2'})
#     # data[df] = pd.concat(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])
#
# # print(tmp1.shape)
# # print(tmp2.shape)
# data[df] = pd.concat([data['ar'], data['hr']], sort=True)
#
# # print(data[df].head())
# # print(data[df].tail())
#
#
# # Convert air_visit_data field visit_date to date type and split them to year, month, dayofweek
# data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
# data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
# data['tra']['year'] = data['tra']['visit_date'].dt.year
# data['tra']['month'] = data['tra']['visit_date'].dt.month
# data['tra']['visit_date'] = data['tra']['visit_date'].dt.date
#
# # Split the sample submission file to 2 main field air_store_id and date. Then split date to year, month, dayofweek
# data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
#
# # print(data['tes']['visit_date'].tail())
# data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
#
# # print(data['tes']['air_store_id'].tail())
# data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
# data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
# data['tes']['year'] = data['tes']['visit_date'].dt.year
# data['tes']['month'] = data['tes']['visit_date'].dt.month
# data['tes']['visit_date'] = data['tes']['visit_date'].dt.date
#
# # get the unique stores from sample submission file base on air_store_id, preparing for forecasting
# unique_stores = data['tes']['air_store_id'].unique()
# print("data['tes']['air_store_id'].tail():", data['tes']['air_store_id'].tail())
#
# # There are 821 unique store needed to forecast
# print("unique_stores.shape:", unique_stores.shape)
#
# stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)
# print("stores.tail():", stores.tail())

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

df_avd = data['avd']
df_ar = data['ar']
df_hr = data['hr']

df_sidr = data['sidr']


# print("df_avd.describe():\n", df_avd.describe())
# Total = df_avd['visitors'].sum()
# print("sumf of df_avd:", Total)
#
# print("df_ar.describe():\n", df_ar.describe())
# Total = df_ar['reserve_visitors'].sum()
# print("sumf of df_ar:", Total)
#
# print("df_hr.describe():\n", df_hr.describe())
# Total = df_hr['reserve_visitors'].sum()
# print("sumf of df_hr:", Total)

genre_name = 'genre_name'
area_name = 'area_name'
store_id = 'store_id'

# IMPUTATION_METHOD = ['median', 'mean', 'linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'barycentric',
#           'krogh', 'polynomial', 'spline', 'piecewise_polynomial', 'from_derivatives', 'pchip', 'akima']
IMPUTATION_METHOD = ['median', 'mean']
IMPUTATION_METHOD_ARG = 'median'
MAX_MISSING_PERCENTAGE = [25, 50, 75, 100]
RESAMPLING_METHOD = ['under']
NUM_OF_HC_CLUSTER = [3, 9]
NUM_OF_HC_CLUSTER_ARG = 3
SPLIT_GROUPS = [3, 9]


def store_info_format(df):
    df = df.drop(['latitude', 'longitude'], axis = 1)

    # convert non-ascii characters by ignore them
    df[area_name] = df[area_name].apply(lambda x: x.\
                                              encode('ascii', 'ignore').\
                                              decode('ascii').\
                                              strip())
    return df

# RES_DATASET = ['air', 'hpg']
RES_DATASET = ['air']
# This implementation is used for header of columns
for i in RES_DATASET:
    if i == 'air':
        store_info_dataset = data['asi']
        store_info_dataset = store_info_format(store_info_dataset)
        res = i
    elif i == 'hpg':
        store_info_dataset = data['hsi']
        store_info_dataset = store_info_format(store_info_dataset)
        res = i
        # visit_date_header = 'visit_datetime'
    else:
        # store_info_dataset = all_rsv_visitor
        res = 'air'



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
def hpg_reserve_visitor_format(df):
    df['visit_datetime'] = pd.to_datetime(df['visit_datetime'], format="%Y/%m/%d").dt.date
    df = (df.groupby(['hpg_store_id','visit_datetime'])
        .agg({'reserve_visitors': 'sum'})
        .reset_index()
        .rename(columns={'hpg_store_id':'store_id', 'visit_datetime':'visit_date', 'reserve_visitors':'visitors'})
    )
    Total = df['visitors'].sum()
    print("Sum of df hpg reserve visitors:\n", Total)
    print("hpg_reserve_visitor_format:\n", df)
    return df

# # This function will merge hpg reserver file with store id relationship with air file
# # output will be used to merge again with air visit date and air reserve
# # output:
# #   df - hpg visit date and it is merged with store id relationship. Which will be used to merge with air system
# # to get mixed dataset
# #   df_hvd: dataframe for hpg visit dataset.
# def hpg_reserve_visitor_format(df, df_sidr):
#     df['visit_datetime'] = pd.to_datetime(df['visit_datetime'], format="%Y/%m/%d").dt.date
#     df = (df.groupby(['hpg_store_id','visit_datetime'])
#         .agg({'reserve_visitors': 'sum'})
#         .reset_index()
#         .rename(columns={'hpg_store_id':'store_id', 'visit_datetime':'visit_date', 'reserve_visitors':'visitors'})
#     )
#     # print("df_sidr:", df_sidr)
#     print("df hpg reserve before merged:\n", df)
#     Total = df['visitors'].sum()
#     print("sum of df hpg reserve visitors before merged:\n", Total)
#     # This is the dataframe of only hpg reserve system without merging with air system
#     df_hvd = df
#
#     df = pd.merge(df, df_sidr, on=['hpg_store_id'], how='inner')
#     print("hpg_reserve_visitor_format merge with store id relation with air:\n", df)
#     return df, df_hvd
#
# # Method use: merge air visit date and air reserve after groupby and hpg reserve after groupby
# def all_reserve_visitor(df, air, hpg):
#     print("df:", df)
#     print("air:", air)
#     df['visit_date'] = pd.to_datetime(df['visit_date'])
#     air['visit_date'] = pd.to_datetime(air['visit_date'])
#     hpg['visit_date'] = pd.to_datetime(hpg['visit_date'])
#     df = pd.merge(df, air, on=['air_store_id', 'visit_date'], how='inner')
#     df = pd.merge(df, hpg, on=['air_store_id', 'visit_date'], how='inner')
#     df['reserve_visitors'] = df['reserve_visitors_air'] + df['reserve_visitors_hpg']
#     df = df[['air_store_id', 'hpg_store_id', 'visit_date', 'reserve_visitors_air',
#              'reserve_visitors_hpg', 'visitors', 'reserve_visitors']]
#     print("df:", df)
#     return df


# df_ar = air_reserve_visitor_format(df_ar)
df_hr_format = hpg_reserve_visitor_format(df_hr)
# hpg_rvg, df_hvd = hpg_reserve_visitor_format(df_hr, df_sidr)
# all_rsv_visitor = all_reserve_visitor(df_avd, df_ar, hpg_rvg)



# ============ Step 01: Split dataframe air - hpg store info to 9 groups ==============
# Get 3 first values of the first column in a dataframe
def get_first_3values_from_df(df):
    location_1st = df.iloc[0][0]
    location_2nd = df.iloc[1][0]
    location_3rd = df.iloc[2][0]
    return location_1st, location_2nd, location_3rd

# Method: Undersampling
# input: dataframe of time series which are needed to reduce number of samples.
# size: size of the new sample
# Due to randomly choosing in undersampling method, location or area of those series will be shuffle as well.
# This action will affect in choosing 3 largest areas in the next step
def under_sampling(df, size):
    # Shuffle dataset in row
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.iloc[0:size] # first size rows of dataframe
    return df

def format_arename_col_first_word(df):
    # Get first letter from area name column
    df['new_col'] = df[area_name].astype(str).str.split().str.get(0)
    df[area_name] = df['new_col']
    df.drop('new_col', axis=1, inplace=True)
    return df

# Get a new dataframe which contain equal size of restaurants for 3 genres
# by resampling dataset by undersampling or oversampling
# Input:
# df: full dataset of 3 genres
# gerne_1st, gerne_2nd, gerne_3rd: 3 max gernes of the full input dataframe
# Output:
# df_genre: Full dataframe with 3 gernes after sampling,
# df_genre_1st, df_genre_2nd, df_genre_3rd: and 3 dataframes for 3 types of genres for further processing on area
def get_3_equal_genre_size_sampling(df, gerne_1st, gerne_2nd, gerne_3rd):
    # df_genre = df.loc[df[genre_name].isin([gerne_1st, gerne_2nd, gerne_3rd])].reset_index(drop=True)
    # Full store id list
    df_genre_1st = df.loc[df[genre_name].isin([gerne_1st])].reset_index(drop=True)
    df_genre_2nd = df.loc[df[genre_name].isin([gerne_2nd])].reset_index(drop=True)
    df_genre_3rd = df.loc[df[genre_name].isin([gerne_3rd])].reset_index(drop=True)

    if RESAMPLING_METHOD[0] == 'under':
        size = df_genre_3rd.shape[0]
        df_genre_1st = under_sampling(df_genre_1st, size)
        # Due to randomly choosing in undersampling method, location or area of those series will be shuffle as well.
        # This action will affect in choosing 3 largest areas in the next step
        print("df_genre_1st:\n", df_genre_1st)
        print("df_genre_1st shape 0 :", df_genre_1st.shape[0])

        df_genre_2nd = under_sampling(df_genre_2nd, size)
        print("df_genre_2nd:\n", df_genre_2nd)
        print("df_genre_2nd shape:", df_genre_2nd.shape[0])

    else:
    #     sampling method == 'over'
    #     size = df_genre_1st.shape[0]
    #     df_genre_2nd = over_sampling(df_genre_2nd, size)
    #     df_genre_3rd = over_sampling(df_genre_3rd, size)
        print("Oversampling data: temporary not implemented due to diversity of clusters")

    frames_after_sampling = [df_genre_1st, df_genre_2nd, df_genre_3rd]
    df_genre = pd.concat(frames_after_sampling).reset_index(drop=True)

    df_genre = format_arename_col_first_word(df_genre).reset_index(drop=True)
    df_genre_1st = format_arename_col_first_word(df_genre_1st)
    df_genre_2nd = format_arename_col_first_word(df_genre_2nd)
    df_genre_3rd = format_arename_col_first_word(df_genre_3rd)

    return df_genre, df_genre_1st, df_genre_2nd, df_genre_3rd

# Get a new dataframe which contain equal size of restaurants for 3 areas
# by resampling dataset by undersampling or oversampling
# Input:
# df: full dataset of 1 genre
# location_1st, location_2nd, location_3rd: 3 max gernes of the full input dataframe
# size: size of the least number of restaurant of 3 areas. By this we have 3 equal-sized areas
# Output:
def get_3_equal_location_size_sampling(df, location_1st, location_2nd, location_3rd, size):
    df_location_1st = df.loc[df[area_name].isin([location_1st])].reset_index(drop=True)
    df_location_2nd = df.loc[df[area_name].isin([location_2nd])].reset_index(drop=True)
    df_location_3rd = df.loc[df[area_name].isin([location_3rd])].reset_index(drop=True)

    print("Test df_location_1st:\n", df_location_1st)
    print("Default size need to be cut of from the main part:", size)

    if RESAMPLING_METHOD[0] == 'under':
        df_location_1st = under_sampling(df_location_1st, size)
        print("df_location_1st:\n", df_location_1st)
        print("df_location_1st shape 0 :", df_location_1st.shape[0])

        df_location_2nd = under_sampling(df_location_2nd, size)
        print("df_location_2nd:\n", df_location_2nd)
        print("df_location_2nd shape:", df_location_2nd.shape[0])

        df_location_3rd = under_sampling(df_location_3rd, size)
        print("df_location_3rd:", df_location_3rd)
        print("df_location_3rd shape:", df_location_3rd.shape[0])

    # else:
    # #     sampling method == 'over'
    #     size = df_location_1st.shape[0]
    #     df_location_2nd = over_sampling(df_location_2nd, size)
    #     df_location_3rd = over_sampling(df_location_3rd, size)

    frames_after_sampling = [df_location_1st, df_location_2nd, df_location_3rd]
    df_location = pd.concat(frames_after_sampling).reset_index(drop=True)
    print("Full df_location for 1 genre:\n", df_location)

    return df_location

# Left join 3 dataframes to get a new dataframe with maximium rows.
def left_join_df(df_left, df_middle, df_right):
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

    df_left_join = df_left.merge(df_middle,how='left', left_on=area_name, right_on=area_name)
    df_left_join = df_left_join.merge(df_right,how='left', left_on=area_name, right_on=area_name)
    df_left_join = df_left_join.fillna(0)
    print("df_left_join df_left_join:\n", df_left_join)
    df = df_left_join[area_name]
    print("df df:\n", df)
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

# Method to split dataset to 3 genres groups and 9 genre-area groups.
# input: df - input store info file with full genre, area. This file did not include time series
# output:  df_genre, df_3genres_3locations
# This 2 outputs will be used to verify cluster number twice
def split_asi_9_groups(df):
    # Group by air_genre_name or hpg_genre_name and count unique values with pandas per groups for air_store_info.csv,
    # air_store_info.csv or mixing between air and hpg
    # df_groups = df.groupby(genre_name)[store_id].nunique().sort_values(store_id)
    df_groups = df.groupby(genre_name).count().sort_values(store_id, ascending=False).reset_index()
    print('\n=== LIST OUT ALL GENRE GROUPS ASCENDING df_groups: === \n', df_groups)

    # ========= Overall of 3 genres and 3 locations =========
    # Filtering air_store_info df to 3 main genres
    gerne_1st, gerne_2nd, gerne_3rd = get_first_3values_from_df(df_groups)
    print("gerne_1st:", gerne_1st)
    print("gerne_2nd:", gerne_2nd)
    print("gerne_3rd:", gerne_3rd)

    # df_genre: full of 3 equal genre dataframe
    df_genre, df_genre_1st, df_genre_2nd, df_genre_3rd = get_3_equal_genre_size_sampling(df, gerne_1st, gerne_2nd, gerne_3rd)

    print("=== ALL RESTAURANTS OF 3 MAIN GENRES - BEGIN: ===\n")
    print(df_genre)
    print('Total number of restaurants in three equalled-size main genres:', df_genre.shape[0])
    print("=== ALL RESTAURANTS OF 3 MAIN GENRES - END:   ===\n")

    # 01 Back up this implementation line for further debugging
    # df_genre_area = df_genre_1st.groupby(area_name).count().sort_values(store_id, ascending=False).reset_index()

    # Group by area for 3 genres df_genre_1st, df_genre_2nd, df_genre_area_3rd then pipe it to input for
    # equallizing size of area
    df_genre_area_1st = df_genre_1st.groupby(area_name).count().sort_values(store_id, ascending=False).reset_index()
    print("df_genre_area_1st:\n", df_genre_area_1st)
    df_genre_area_2nd = df_genre_2nd.groupby(area_name).count().sort_values(store_id, ascending=False).reset_index()
    print("df_genre_area_2nd:\n", df_genre_area_2nd)
    df_genre_area_3rd = df_genre_3rd.groupby(area_name).count().sort_values(store_id, ascending=False).reset_index()
    print("df_genre_area_3rd:\n", df_genre_area_3rd)

    # Create left join dataframe to get merge of 3 genres and 3 areas.
    # With this dataframe, we can identify 3 top areas with the least number of restaurants of the third place
    df_left_join = left_join_df(df_genre_area_1st, df_genre_area_2nd, df_genre_area_3rd)
    # print("df_left_join:\n", df_left_join)

    # 01 Back up this implementation line for further debugging
    # location_1st, location_2nd, location_3rd = get_first_3values_from_df(df_genre_area)
    location_1st, location_2nd, location_3rd = get_first_3values_from_df(df_left_join)
    print("location_1st:", location_1st)
    print("location_2nd:", location_2nd)
    print("location_3rd:", location_3rd)

    # Get the size of the least areas in top 3 most areas
    genre_area_size = df_left_join.iloc[2][1]
    print("genre_area_size:", genre_area_size)

    # ---Filtering df_genre to 3 main genres---

    # Method 01: search all locations
    # search_for_loc = [location_1st, location_2nd, location_3rd]df_location_2nd shape
    # df_genre_location = df_genre.loc[df_genre[area_name].str.contains('|'.join(search_for_loc))].reset_index(drop=True)

    # total_num_res_gerne_location = df_genre_location.shape[0]
    # print('Total number of restaurants in three main genres with three main locations:', total_num_res_gerne_location)
    # print("df_genre_location:====\n", df_genre_location)
    #
    # print('\n=== LIST OUT 3 LARGEST GROUP NAMES: ===')
    # # Split air_store_info to 3 main genres: Italian/French, Izakaya, Cafe/Sweets
    # df_gerne3rd = df[df[genre_name] == gerne_3rd]
    # df_gerne1st = df[df[genre_name] == gerne_1st]
    # df_gerne2nd = df[df[genre_name] == gerne_2nd]
    #
    # # Split each genre to 3 main locations
    # mask_location1st = df_gerne1st[area_name].str.contains(location_1st)
    # mask_location2nd = df_gerne1st[area_name].str.contains(location_2nd)
    # mask_location3rd = df_gerne1st[area_name].str.contains(location_3rd)
    # df_gerne1st_location1st = df_gerne1st[mask_location1st].reset_index(drop=True)
    # df_gerne1st_location2nd = df_gerne1st[mask_location2nd].reset_index(drop=True)
    # df_gerne1st_location3rd = df_gerne1st[mask_location3rd].reset_index(drop=True)
    # print("Total number of restaurants from df_gerne1st_location1st, df_gerne1st_location2nd and df_gerne1st_location3rd:",
    #       df_gerne1st_location1st.shape[0], df_gerne1st_location2nd.shape[0], df_gerne1st_location3rd.shape[0])
    #
    # # Split each genre to 3 main locations
    # mask_location1st = df_gerne2nd[area_name].str.contains(location_1st)
    # mask_location2nd = df_gerne2nd[area_name].str.contains(location_2nd)
    # mask_location3rd = df_gerne2nd[area_name].str.contains(location_3rd)
    # df_gerne2nd_location1st = df_gerne2nd[mask_location1st].reset_index(drop=True)
    # df_gerne2nd_location2nd = df_gerne2nd[mask_location2nd].reset_index(drop=True)
    # df_gerne2nd_location3rd = df_gerne2nd[mask_location3rd].reset_index(drop=True)
    # print("Total number of restaurants from df_gerne2nd_location1st, df_gerne2nd_location2nd and df_gerne2nd_location3rd:",
    #       df_gerne2nd_location1st.shape[0], df_gerne2nd_location2nd.shape[0], df_gerne2nd_location3rd.shape[0])
    #
    # # Split each genre to 3 main locations
    # mask_location1st = df_gerne3rd[area_name].str.contains(location_1st)
    # mask_location2nd = df_gerne3rd[area_name].str.contains(location_2nd)
    # mask_location3rd = df_gerne3rd[area_name].str.contains(location_3rd)
    # df_gerne3rd_location1st = df_gerne3rd[mask_location1st].reset_index(drop=True)
    # df_gerne3rd_location2nd = df_gerne3rd[mask_location2nd].reset_index(drop=True)
    # df_gerne3rd_location3rd = df_gerne3rd[mask_location3rd].reset_index(drop=True)
    # print("Total number of restaurants from df_gerne3rd_location1st, df_gerne3rd_location2nd and df_gerne3rd_location3rd:",
    #       df_gerne3rd_location1st.shape[0], df_gerne3rd_location2nd.shape[0], df_gerne3rd_location3rd.shape[0])
    #
    # # Full store id list
    # frames = [df_gerne3rd_location1st, df_gerne3rd_location2nd, df_gerne3rd_location3rd,
    #           df_gerne1st_location1st, df_gerne1st_location2nd, df_gerne1st_location3rd,
    #           df_gerne2nd_location1st, df_gerne2nd_location2nd, df_gerne2nd_location3rd]
    # df_3genres_3locations = pd.concat(frames)
    # print("df_3genres_3locations:===\n", df_3genres_3locations)
    # print('Total number of restaurants in 3 main genres with 3 main locations:', df_3genres_3locations.shape[0], '\n')

    # return df_genre_location, df_3genres_3locations

    # Method 02: search each of them, equalize 3 proportions
    df_genre_location_1st = get_3_equal_location_size_sampling(df_genre_1st, location_1st, location_2nd, location_3rd, genre_area_size)
    df_genre_location_2nd = get_3_equal_location_size_sampling(df_genre_2nd, location_1st, location_2nd, location_3rd, genre_area_size)
    df_genre_location_3rd = get_3_equal_location_size_sampling(df_genre_3rd, location_1st, location_2nd, location_3rd, genre_area_size)

    # Full store id list
    frames = [df_genre_location_1st, df_genre_location_2nd, df_genre_location_3rd]
    df_3genres_3locations = pd.concat(frames).reset_index(drop=True)
    # print("df_3genres_3locations:===\n", df_3genres_3locations)
    print('Total number of restaurants in 3 main genres with 3 main locations:', df_3genres_3locations.shape[0], '\n')

    return df_genre, df_3genres_3locations

df_3genres, df_3genres_3locations = split_asi_9_groups(store_info_dataset)
print("Dataframe contains all  3 genres of 3 locations - df_3genres_3locations:\n", df_3genres_3locations)
print("Dataframe contains only 3 genres - df_3genres:\n", df_3genres)
# print("df_genre_location:\n", df_genre_location)

# ============ Step 02: Check missing values ==============
# Method use:
# input:
# df_3genres_3locations - list of store names of 3 genres and 3 locations
# df_hr_format - list of all visit of visitor of all stores
# output:
# df_ts_visits - List of all chosen timeseries and their visitors
def get_avd_3genres_3locations(df_3genres, df_3genres_3locations, df_avd):
    # print("df_3genres_3locations - df_3genres_3locations:\n", df_3genres_3locations)
    # air_visit_data.csv
    print("Full dataframe of air visit data rows - df_avd:\n", df_avd)
    if SPLIT_GROUPS == 3:
        df = df_3genres
    else: # SPLIT_GROUPS == 9
        df = df_3genres_3locations

    print("dataframe from 3 genres or 3genres/3locations:\n", df)
    # Filtering dataframe air_visit_data based on the column value of another dataframe df_3genres_3locations
    mask_air_store_id = df_avd.store_id.isin(df.store_id)
    df_ts_visits = df_avd[mask_air_store_id].reset_index(drop=True)
    print("Dataframe of air visit data from chosen time series:", df_ts_visits.shape[0])
    # print("df_ts_visits:\n", df_ts_visits)
    return df_ts_visits

# Method use: Filtering only time series base on 3 genres / 3 genres and 3 locations
# input:
# df_3genres_3locations - list of store names of 3 genres and 3 locations
# df_hr_format - list of all visit of visitor of all stores
# output:
# df_ts_visits - List of all chosen timeseries and their visitors
def get_hvd_3genres_3locations(df_3genres, df_3genres_3locations, df_hr_format):
    print("Dataframe contains all  3 genres of 3 locations - df_3genres_3locations:\n", df_3genres_3locations)
    print("Standard format of HPG restaurants time series - visit date is grouped - df_hr_format:\n", df_hr_format)

    if SPLIT_GROUPS == 3:
        df = df_3genres
    else: # SPLIT_GROUPS == 9
        df = df_3genres_3locations

    # Filtering dataframe hpg_visit_data based on the column value of another dataframe df_3genres_3locations
    # df_3genres_3locations: Dataframe contains a list of serie store names which are in 3 main genres and 3 main areas
    # Create a mask of hpg_store_id between df.hpg_store_id and df_3genres_3locations.hpg_store_id
    mask_hpg_store_id = df_hr_format.store_id.isin(df.store_id)
    # Get all series with the accordingly series name
    df_ts_visits = df_hr_format[mask_hpg_store_id].reset_index(drop=True)

    print("====================================================")
    print("Filtering HPG restaurants time series only chosen restaurants:", df_ts_visits.shape[0])


    # print("df_ts_visits:\n", df_ts_visits)
    return df_ts_visits

if res == 'air':
    df_store_and_visit = get_avd_3genres_3locations(df_3genres, df_3genres_3locations, df_avd)
    # This line below is used to test for 3 genres only without spliting it in to 9 groups.
    # This line should be checked again for input and output
    # df_vd_3genres_3locations = get_avd_3genres_3locations(df_3genres)
    print("df_store_and_visit:\n", df_store_and_visit)
    print("====================================================")
else:
    df_store_and_visit = get_hvd_3genres_3locations(df_3genres, df_3genres_3locations, df_hr_format)
    print("df_store_and_visit:\n", df_store_and_visit)
    print("====================================================")

# air = df.loc[df['air_store_id'] == 'air_25e9888d30b386df']
# print(air['visit_date'].min(), air['visit_date'].max())
# print(air)

# At this point of coding, we can find that we have missing values on each time series

# # Find the list of all starting moment of all series
# def data_moment_list(store_id, j):
#     series = df.loc[df['air_store_id'] == store_id]
#     # series = series.set_index('visit_date')
#     # series.index = pd.DatetimeIndex(series.index)
#     # series = series.reindex(index)
#     first_moment = series['visit_date'].iloc[0]
#     # str1 = str(store_id)
#     # str1 = str1 + ' ' + str(j) +' visitors'
#     # print('type:', type(first_moment))
#     return first_moment
#
# list_first_moments=[]
# for i, j in zip(store_id_list, range(len(store_id_list))):
#     list_first_moments.append(data_moment_list(i, j))
# # print('list_first_moments:', list_first_moments)
# # statistics.median(list_first_moments)
# # pd.Timestamp.fromordinal(int(list_first_moments.apply(lambda x: x.toordinal()).median()))

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


first_moment, last_moment = first_last_moments(df_store_and_visit)

# # Format datetime column to date column
# def format_datetime_df(df):
#     df['visit_date'] = pd.to_datetime(df['visit_date'], format="%Y/%m/%d").dt.date
#     # df.rename(columns={'visit_datetime':'visit_date'}, inplace=True)
#     return df

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

average_first_moment_timepoint, average_last_moment_timepoint = average_first_last_moments(first_moment, last_moment)

# ============ Step 03 and 04: Data Imputation and start at the same moments ==============

def percentage(part, whole, digits):
    val = float(part)/float(whole)
    val *= 10 ** (digits + 2)
    return (floor(val) / 10 ** digits)

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
    # print("imputing_one_timeserie - df:111\n", df)
    # print("imputing_one_timeserie - sid:", sid)
    series = df.loc[df[store_id] == sid]

    series = series.set_index('visit_date')
    series.index = pd.DatetimeIndex(series.index)
    # print("imputing_one_timeserie - series 111:\n", series)
    series = series.reindex(dr_idx)
    # print("imputing_one_timeserie - series 222:\n", series)

    if method == 'mean':
        # Mean values imputation method
        print("series['visitors'] before:\n",series['visitors'])
        series[column] = series[column].fillna(series[column].mean())
        # roundup and convert to int
        series['visitors'] = series['visitors'].apply(lambda x: round(x, 0)).astype(int)
        print("series['visitors'] after:\n",series['visitors'])
    elif method == 'median':
        # Median values imputation method
        series['visitors'] = series['visitors'].fillna(series['visitors'].median())
    # Implemenation by interpolate dataframe, this is important step.
    elif method == 'linear':
        # Please note that only method='linear' is supported for DataFrame
        print("series['visitors'] before:\n", series['visitors'])
        upsampled = series['visitors']
        interpolated = upsampled.interpolate(method='linear', limit=None, limit_direction='both')
        series['visitors'] = interpolated
        series['visitors'] = series['visitors'].astype(np.int64)
        print("series['visitors'] after:\n", series['visitors'])
    else:
        # By choosing method, we can conclude that, dataset can be filled in order:
        # 1. The method
        # 2. forward fill
        # 3. backward fill
        # Method list: time, index, values, nearest, zero, slinear, cubic, barycentric, krogh, polynomial, spline
        # piecewise_polynomial, from_derivatives, pchip, akima
        # ‘ffill’ stands for ‘forward fill’ and will propagate last valid observation forward.
        # Pandas dataframe.bfill() is used to backward fill the missing values in the dataset.
        print("series['visitors'] before:\n", series['visitors'])
        # sum0 = series['visitors'].isna().sum()
        # series['visitors'] = series['visitors'].interpolate(method='polynomial', order=1, limit=None, limit_direction='both')
        # sum1 = series['visitors'].isna().sum()
        # series['visitors'] = series['visitors'].interpolate(method='polynomial', order=1, limit=None, limit_direction='both').ffill()
        # sum2 = series['visitors'].isna().sum()
        series['visitors'] = series['visitors'].interpolate(method='akima', order=1, limit=None, limit_direction='both').ffill().bfill()
        # sum3 = series['visitors'].isna().sum()
        # print("sum0, sum1 and sum2 sum3:", sum0, sum1, sum2, sum3)

        series['visitors'] = series['visitors'].astype(np.int64)
        print("series['visitors'] after:\n", series['visitors'])

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

# Method: finding missing values of
# input:
#   df - df_store_and_visit:
#   sid: store id of the specific restaurant
#   df_standard_range: dataframe with standard range. This range is extracted
#       to cage all timeseries from first to last time point
def finding_missing_values(df_store_and_visit, sid, df_standard_range):
    df_missing_values_series = df_store_and_visit.loc[df_store_and_visit[store_id] == sid]
    df_missing_values_series = df_missing_values_series.set_index('visit_date')
    #check for missing datetimeindex values based on reference index (with all values)
    missing_dates = df_standard_range.index[~df_standard_range.index.isin(df_missing_values_series.index)]
    return len(missing_dates)

# Python program to get average of a list
def Average(lst):
    return sum(lst) / len(lst)

# Create new visitor dataframe
# input:
# df - df_store_and_visit : dataframe contains the merge of all store and visits,
#   which is used as input to create distance matrix of all time series
# dr_idx            : date range of for the index
# method            : method for imputation -- we can use a lot of other types of method -- mean, ...
# floor_percentage  : create new dataframe base only on series has missing percentage less than floor_percentage
# output:
# list_removed_timeseries_index: list of timeseries which are removed for high missing percentage rate
def imputing_all_timeseries(df_store_and_visit, dr_idx, method, floor_percentage):
    print("Step 03: data imputation - imputing_all_timeseries\n")
    print("df_store_and_visit\n", df_store_and_visit)

    # store_id_list     : store id list of all series
    store_id_list = df_store_and_visit[store_id].tolist()
    # Removing duplicate values
    store_id_list = list(set(store_id_list))
    print('Length of Store id list of input dataframe:', len(store_id_list))
    store_id_list = np.asarray(store_id_list)

    # Create an empty dataframe in creating new dataframe by adding one by one of imputed time series
    all_imputed_timeseries = pd.DataFrame()
    total_num_missing_values = 0
    # Create an empty dataframe to be standard of date range index
    # print("dr_idx:", dr_idx)
    df_standard_range = pd.DataFrame(index=dr_idx)
    # Length of standard dataframe
    len_std_df = len(df_standard_range)

    # Create a list of one-timeseries dataframe. Then I concatenate all of them.
    list_timeseries = []
    list_removed_timeseries_index = []
    list_percents = []
    for i, j in zip(store_id_list, range(len(store_id_list))):
        # print("List item:", i)
        # Proceeding data imputation for each of timeseries by the chosen imputation method
        one_imputed_timeseries = imputing_one_timeseries(df_store_and_visit, i, dr_idx, method, 'visitors', j)
        num_missing_values     = finding_missing_values(df_store_and_visit, i, df_standard_range)
        # Adding only series with missing percentage less than the input percentage
        percent = percentage(num_missing_values, len_std_df, 2)
        list_percents.append(percent)
        # print("num_missing_values_df of serie with percentage:", num_missing_values, i, percent)

        if percent < floor_percentage:
            total_num_missing_values = total_num_missing_values + num_missing_values
            # Concatenate a list of of series to form up an dataframe
            all_imputed_timeseries = pd.concat([all_imputed_timeseries, one_imputed_timeseries], axis=1)
            list_timeseries.append(one_imputed_timeseries)
            # Just to show the last id list
            # if i==store_id_list[-1]:
        else:
            list_removed_timeseries_index.append(int(j))

    # del(store_id_list[list_removed_timeseries_index])
    average_percent = Average(list_percents)
    print("Average percent of the list of missing percents:", round(average_percent, 2))
    print("List of removed timeseries - list_removed_timeseries_index:", list_removed_timeseries_index)
    print("Total number of timeseries before removing (store_id_list)        :", len(store_id_list))
    print("Total number of removed timeseries (list_removed_timeseries_index):", len(list_removed_timeseries_index))

    num_remained_series = len(store_id_list) - len(list_removed_timeseries_index)
    print('Total number of restaurants in three main genres with three main locations after removing high missing rate series:',
          num_remained_series)

    print("Length of standard dataframe:", len_std_df)
    print('Total number of restaurants in three main genres:', df_3genres.shape[0])
    print('Total number of restaurants in three main genres with three main locations:', df_3genres_3locations.shape[0])

    # # It seems this del command is used to remove those time series which are in the removed list
    # # It seems there is no need for using it right now. There is an error in deleting: ValueError: cannot delete array elements
    #     # for index in sorted(list_removed_timeseries_index, reverse=True):
    #     #     del store_id_list[index]
    # print("store_id_list after removing:", store_id_list)

    print("Total number of missing values:", total_num_missing_values)
    # Calculate the maxium of all time points in all time series.
    # It is calculated by multiply the standard dataframe with number of remained series after removing
    maximum_num_values_of_all_remained_series = len_std_df*num_remained_series
    print("Maximum number of times series points of all remained series:", maximum_num_values_of_all_remained_series)
    print("Percentage of missing values:", percentage(total_num_missing_values, maximum_num_values_of_all_remained_series, 2))

    return all_imputed_timeseries

date_range_idx = pd.date_range(average_first_moment_timepoint, average_last_moment_timepoint)



# Method used: Find the missing percentage of df_store_and_visit
# input:
#   df_store_and_visit: contains all stores and all visits
#   fm: first moment dataframe of all stores
#   lm: last  moment dataframe of all stores
# output:
#   missing percentage of all stores according to first and last moment
def missing_percentage(df, first_timepoint, last_timepoint):
    # print("---------------df", df)
    current_num_store_and_visit = df.shape[0]
    print("---------------total_num_rows ", current_num_store_and_visit)

    total_num_store = df[store_id].nunique()
    print("---------------total_num_store", total_num_store)
    # Finding the min and max day of a column
    # d1 = df.visit_date.min()
    # d2 = df.visit_date.max()

    last_timepoint  = datetime.strptime(last_timepoint, "%m-%d-%Y")
    first_timepoint = datetime.strptime(first_timepoint, "%m-%d-%Y")
    print("type last_timepoint", type(last_timepoint))
    days_diff = abs((last_timepoint - first_timepoint).days)
    print("days_diff:", days_diff)

    # Total number of possible day for all stores
    total_num_possible_day = total_num_store*days_diff
    return percentage(current_num_store_and_visit, total_num_possible_day, 2)

miss_percent = missing_percentage(df_store_and_visit, average_first_moment_timepoint, average_last_moment_timepoint)
print("miss_percent miss_percent", miss_percent)

# Check value of concatenated dataframe and list of series
# ============== Using multiple method to control the imputation steps
# ============== We can use others method like mean, median or
# ============== method : {‘linear’, ‘time’, ‘index’, ‘values’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’,
# ============== ‘barycentric’, ‘krogh’, ‘polynomial’, ‘spline’, ‘piecewise_polynomial’, ‘from_derivatives’, ‘pchip’, ‘akima’}
imputation_method = ['linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'barycentric',
          'krogh', 'polynomial', 'spline', 'piecewise_polynomial', 'from_derivatives', 'pchip', 'akima']

# Visitor dataframe is used to collect all timeseries as a matrix.
# This matrix is used as input for cluster technique
# or it can be used to create distance matrix.
visitor_df = imputing_all_timeseries(df_store_and_visit, date_range_idx,
                               IMPUTATION_METHOD_ARG, MAX_MISSING_PERCENTAGE[3])
# Draw out variables and arguements and theirs functions

print('Visitor dataframe after data imputation - visitor_df head:\n {} \n'.format(visitor_df.head()))
print('Visitor dataframe after data imputation - visitor_df tail:\n {} \n'.format(visitor_df.tail()))
# Overview of all series
print("visitor_df.describe():\n", visitor_df.describe())

# ============================================= For testing hierachy =============================================
# visitor_df_transposed : is the transposed version of visitor_df
visitor_df_transposed = visitor_df.transpose()
# visitor_df_transposed.to_csv('output_visitor.csv')
# print("visitor_df_transposed:--------------", visitor_df_transposed)

# Plot example dataframe =============================================
def plot_visitor(df):
    df.plot(legend=False)
    plt.show()

# plot_visitor(visitor_df)
# # ============ Step 05: Clustering using DBSCAN and Pairwise Similarity Evaluation ==============
#
# visitor_matrix = visitor_df.values
# original_visitor_df = visitor_df
# # print(type(visitor_array))
# print('visitor_matrix X :\n {} \n'.format(visitor_matrix))
#
# # Transpose the matrix for timeseries as row and timestamps as column
# visitor_matrix = visitor_matrix.transpose()
# print('visitor_matrix after transposed :\n {} \n'.format(visitor_matrix))
#
#
#
# # pairwise_manhattan = pairwise_distances(visitor_matrix, metric='manhattan')
# # print('pairwise_distances manhattan:\n {} \n'.format(pairwise_manhattan))
# #
# # pairwise_euclidean = pairwise_distances(visitor_matrix, metric='euclidean')
# # print('pairwise_distances euclidean:\n {} \n'.format(pairwise_euclidean))
# #
# # similarities = cosine_similarity(visitor_matrix)
# # print('pairwise cosine_similarity:\n {}\n'.format(similarities))
# #
# # # pairwise_DTW = pairwise_distances(visitor_matrix, metric='euclidean')
# # # print('pairwise_distances euclidean:\n {} \n'.format(pairwise_euclidean))

# #############################################################################
def cluster_dbscan(matrix, distance_measure, eps, minS):
    dbs = DBSCAN(eps=eps, metric=distance_measure, min_samples=minS)
    cluster_labels = dbs.fit_predict(matrix)
    core_samples_mask = np.zeros_like(dbs.labels_, dtype=bool)
    #
    core_samples_mask[dbs.core_sample_indices_] = True
    # print("core_samples_mask=== :", core_samples_mask)
    return cluster_labels, core_samples_mask

def run_cluster(matrix, distance_measure, param_min = 0.01, param_max = 1, param_step = 0.01, minS = 1):
    nrows = matrix.shape[0]
    if nrows <= 1:
        raise ValueError("Time-series matrix contains no information. " \
                         "Was all of your data filtered out?")

    normal_matrix = matrix
    max_nclusters = 0
    max_eps = 0
    prev_nclusters = 0
    break_out = False
    parameter_range = np.arange(param_min, param_max, param_step)
    actual_parameters = []
    cluster_label_matrix = np.empty(shape = (nrows, len(parameter_range)), dtype=int)
    for ind, eps in enumerate(parameter_range):
        actual_parameters.append(eps)
        cluster_labels, core_samples_mask = cluster_dbscan(normal_matrix,
                                            distance_measure,
                                            eps, minS)

        nclusters = len(list(np.unique(cluster_labels)))
        n_noise_ = list(cluster_labels).count(-1)

        cluster_label_matrix[:, ind] = cluster_labels
        if nclusters > 1:
            break_out = True
        if (prev_nclusters != nclusters):
            # print('cluster_labels index: {}'.format(ind))
            # print('cluster_labels list : \n {}'.format(cluster_labels))
            print('cluster_labels eps     : {}'.format(eps))
            print('number of the clusters : {}'.format(nclusters))
            print('Noise points           : {}'.format(n_noise_))
            print('=====================================')
        if (prev_nclusters == 1) & (nclusters == 1) & break_out:
          param_max = eps
          break
        else:
          prev_nclusters = nclusters
    #Print out the clusters with their sequence IDs
    # print('cluster_label_matrix:\n {} \n'.format(cluster_label_matrix))
    for i in range(0, cluster_label_matrix.shape[0]):
        encoded_labels = [ str(x).encode() for x \
                in cluster_label_matrix[i, 0:len(actual_parameters)] ]
    return cluster_labels, nclusters, core_samples_mask
    # print("encoded_labels:", encoded_labels)

def clustering_by_dbscan(X, pairwise_distance_array):
    # labels, nclusters = run_cluster(pairwise_manhattan, 'euclidean', param_min = 18040, param_max = 18051, param_step = 10, minS = 1)
    # cluster_labels eps  : 18040
    # number of the clusters : 7

    # labels, nclusters = run_cluster(pairwise_manhattan, 'euclidean', param_min = 16000, param_max = 18000, param_step = 50, minS = 2)
    # labels, nclusters = run_cluster(pairwise_manhattan, 'euclidean', param_min = 3000, param_max = 6000, param_step = 300, minS = 5)
    # labels, nclusters = run_cluster(pairwise_manhattan, 'manhattan', param_min = 90000, param_max = 150000, param_step = 300, minS = 5)
    # labels, nclusters = run_cluster(pairwise_euclidean, 'euclidean', param_min = 280, param_max = 300, param_step = 20, minS = 5)
    # labels, nclusters = run_cluster(pairwise_euclidean, 'euclidean', param_min = 500, param_max = 1100, param_step = 20, minS = 2)

    # labels, nclusters = run_cluster(pairwise_euclidean, 'euclidean', param_min = 100, param_max = 1200, param_step = 20, minS = 1)
    # labels, nclusters = run_cluster(pairwise_euclidean, 'euclidean', param_min = 100, param_max = 1200, param_step = 20, minS = 2)
    # labels, nclusters = run_cluster(pairwise_euclidean, 'euclidean', param_min = 100, param_max = 1200, param_step = 20, minS = 3)

    # labels, nclusters = run_cluster(pairwise_distance_array, 'euclidean', param_min = 960, param_max = 961, param_step = 20, minS = 1)
    # labels, nclusters, core_samples_mask = run_cluster(pairwise_distance_array, 'euclidean', param_min = 1190, param_max = 1191, param_step = 5, minS = 2)

    labels, nclusters, core_samples_mask = run_cluster(pairwise_distance_array, 'euclidean', param_min = 1190, param_max = 1191, param_step = 5, minS = 2)
    check_central_majority(df_3genres_3locations, labels)

    plot_all_ts(X, labels)
    plot_each_group_ts(X, labels, core_samples_mask)

def check_central_majority(df_genre_location, labels):
    df = df_genre_location
    nclusters = len(list(np.unique(labels)))
    unique_labels = set(labels)
    print("Type of nclusters    :", type(nclusters))
    print("Type of unique_labels:", type(unique_labels))
    print("Type   of labels:", type(labels))
    print("Length of labels:", len(labels))
    df = df.copy()
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


# clustering_by_dbscan(original_visitor_df, pairwise_euclidean)

# # ============ Step 06: Clustering using Pairwise Similarity Evaluation DWT ==============
# # DTW Distance between 2 time series with fully window size complexity of O(nm)
# def DTWDistance(s1, s2):
#     DTW={}
#
#     for i in range(len(s1)):
#         DTW[(i, -1)] = float('inf')
#     for i in range(len(s2)):
#         DTW[(-1, i)] = float('inf')
#     DTW[(-1, -1)] = 0
#
#     for i in range(len(s1)):
#         for j in range(len(s2)):
#             dist= (s1[i]-s2[j])**2
#             DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
#
#     return math.sqrt(DTW[len(s1)-1, len(s2)-1])
#
# # DTW Distance between 2 time series with specific window size w to increase speed
# def DTWDistance(s1, s2,w):
#     DTW={}
#
#     w = max(w, abs(len(s1)-len(s2)))
#
#     for i in range(-1,len(s1)):
#         for j in range(-1,len(s2)):
#             DTW[(i, j)] = float('inf')
#     DTW[(-1, -1)] = 0
#
#     for i in range(len(s1)):
#         for j in range(max(0, i-w), min(len(s2), i+w)):
#             dist= (s1[i]-s2[j])**2
#             DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
#
#     return math.sqrt(DTW[len(s1)-1, len(s2)-1])
#
# #  LB Keogh lower bound of dynamic time warping to increase speed
# def LB_Keogh(s1,s2,r):
#     LB_sum=0
#     for ind,i in enumerate(s1):
#
#         lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
#         upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
#
#         if i>upper_bound:
#             LB_sum=LB_sum+(i-upper_bound)**2
#         elif i<lower_bound:
#             LB_sum=LB_sum+(i-lower_bound)**2
#
#     return math.sqrt(LB_sum)
#
#
#
# #  k-means clustering
# def k_means_clust(data,num_clust,num_iter,w=5):
#     centroids=random.sample(data,num_clust)
#     counter=0
#     for n in range(num_iter):
#         counter+=1
#         print(counter)
#         assignments={}
#         #assign data points to clusters
#         for ind,i in enumerate(data):
#             min_dist=float('inf')
#             closest_clust=None
#             for c_ind,j in enumerate(centroids):
#                 if LB_Keogh(i,j,5)<min_dist:
#                     cur_dist=DTWDistance(i,j,w)
#                     if cur_dist<min_dist:
#                         min_dist=cur_dist
#                         closest_clust=c_ind
#             if closest_clust in assignments:
#                 assignments[closest_clust].append(ind)
#             else:
#                 assignments[closest_clust]=[]
#
#         #recalculate centroids of clusters
#         for key in assignments:
#             clust_sum=0
#             for k in assignments[key]:
#                 clust_sum=clust_sum+data[k]
#             centroids[key]=[m/len(assignments[key]) for m in clust_sum]
#
#     return centroids
#
# def clustering_by_kmean(X, num_cluster, iteration):
#     centroids=k_means_clust(X,num_cluster,iteration,4)
#     for i in centroids:
#         plt.plot(i)
#     plt.show()
#
# # clustering_by_kmean(visitor_matrix, 9, 5)
#
# # ============ Step 07: Clustering using Pairwise Euclidean distance ==============
# def clustering_by_kmedoids(vm, D, num_clusters):
#     M, C = kmedoids.kMedoids(D, num_clusters)
#     # print("type of D:", type(D))
#
#     print("M:", M)
#     print("Type(M):", type(M))
#     # print("C==================:", C)
#     # print("C==================:", C[0])
#     for i in C:
#         print('----  :\n {} \n'.format(C[i]))
#     # print("Type(C):", type(C))
#
#     # print('medoids:')
#     # for point_idx in M:
#     #     print("vm[point_idx]:", vm[[point_idx], :])
#
#     # print('')
#     # print('clustering result:')
#     # for label in C:
#     #     for point_idx in C[label]:
#     #         print("label {0}:　{1}".format(label, vm[[point_idx], :]))
#     #         print("==============")
#
#
#
#
# # ============ Step 08: Clustering using OPTICS algorithms ==============
#
#
# def clustering_by_optics(vm):
#     min_of = 2
#     skuscores = pd.DataFrame(columns = ['Name', 'Perc_noise', 'Numclusters','Outliers' , 'DBCV'])
#     skuclust = {}
#
#     # print('vm  :\n {} \n'.format(vm))
#     pairwise_euclidean = pairwise_distances(vm, metric='euclidean')
#     # optics = OPTICS(min_cluster_size=.03)
#     optics = OPTICS(min_cluster_size=2, metric='euclidean')
#     optics.fit(pairwise_euclidean)
#
#     # pc = pd.DataFrame(vm)
#     # print('pc : {}'.format(pc))
#
#     score = {}
#     score['Perc_noise'] = len(np.where(optics.labels_ == -1)[0]) / optics.labels_.shape[0]
#     score['Numclusters']  = len(np.unique(optics.labels_)) - 1
#
#     # # score['Name'] = sku.transformation
#     # score['DBCV'] = optics.validity
#     # print('optics.labels_ : {}'.format(optics.labels_))
#     print('optics.labels_  :\n {} \n'.format(optics.labels_))
#     # print('optics.reachability_  :\n {} \n'.format(optics.reachability_))
#     #
#     # print('core_distances_  :\n {} \n'.format(optics.core_distances_))
#
#     # pc['cluster'] = optics.labels_
#     # outlier_factor = optics.outlier_factor_
#     # pc['of'] = outlier_factor
#     # reachability = optics.reachability_
#     # pc['reach'] = reachability
#     # outliers = np.where(outlier_factor > min_of)
#     # outliers = list(outliers[0].ravel())
#     # # pc['cluster'][outliers] = -2
#     # # pc.set_index(sku.indices, 'SKU',  inplace = True)
#     # score['Outliers'] = len(outliers)
#     # # score['ID'] = str(iterator)
#     # score = pd.DataFrame(score, index = ['Name'])
#     # # score['columns'] = [sku.columns]
#     # score['num_components'] = pc.shape[1]
#     # skuscores = skuscores.append(score)
#     # # skuclust[str(iterator)] = pc
#     print(score)
#     # print('pc : {}'.format(pc))
#
#
# # clustering_by_optics(visitor_matrix)
#
# clustering_by_dbscan(original_visitor_df, pairwise_euclidean)
# # clustering_by_kmean(visitor_matrix, 9, 5)
# # clustering_by_kmedoids(visitor_matrix, pairwise_euclidean, 9)



# # ============ Step 09: Clustering using Hierachy algorithms ==============
from sklearn.cluster import AgglomerativeClustering


# Method used: Get values of all time series as matrix values
# input:
#   visitor_matrix_transposed - Matrix of store id and their visitors after transposed with columns and rows name
# output:
#   matrix_values: Contain values of the input matrix
def split_matrix_values(vmf):
    print("vmf:\n", vmf)
    first_column_values = vmf.iloc[:, 0:1].values
    matrix_values = vmf.iloc[:, 1:].values

    # print("First column values:\n", first_column_values)
    print("Stacking all timeseries and use it as distance matrix values:\n", matrix_values)
    return first_column_values, matrix_values

## ======== We need to remove this section of coding, because it was processed in the first part of coding =====

# Method used: get labels from a dataset according to affinity and linkage
# input:
#   dataset_ts_arg  : dataset of time series as argument
#   affinity_arg    :
#   linkage_arg     :
# output:
#   labels_hc       : labels of hierachy cluster
def labeling_hierachy_cluster(dataset_ts_arg, affinity_arg, linkage_arg):
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
    print("type of one element in dataset_ts_arg:", dataset_ts_arg[0][0])
    print("type of one element in dataset_ts_arg:", type(dataset_ts_arg[0][0]))
    labels_hc = hc.fit_predict(dataset_ts_arg)
    # print("labels_hc:\n", labels_hc)
    # print("labels_hc type:", type(labels_hc))
    return labels_hc

# # Method use: Attaching first 3 columns of store id, genres, locations and its labels.
# # input:
# #   df      : dataframe contains 3 columns of store id, genres, locations
# #   labels  : array of labels
# # output:
# #   df      : dataframe contains 3 columns and its labels
# def main_labelling_clustering(df, labels):
#     labels = labels.tolist()
#     print("df====== 000 :\n", df)
#     df.loc[:,'hc'] = labels
#
#     print("df====== 111 :\n", df)
#     # print("df_merge_id:", df.head())
#     # df_merge_id = df_merge_id.groupby([genre_name]).count()
#     # df_merge_id= df_merge_id.groupby(genre_name).air_genre_name.count()
#
#     df = df.groupby([genre_name, df.columns[3]]).size()
#     print("df====== 222 :", df)
#     df = df.to_frame(name = 'size').reset_index()
#     print("df_merge_id:", df)
#
#     idx = df.groupby([genre_name])['size'].transform(max) == df['size']
#     df = df[idx]
#     print("df======: \n", df)
#     df = df.groupby(genre_name)['hc'].apply(lambda x: ','.join(map(str, x))).reset_index()
#     print("df_merge_id:", df)
#     df = df[['hc']]
#     print("df_merge_id:", df)
#     return df

# Method use: Attaching first 3 columns of store id, genres, locations and its labels.
# input:
#   df      : dataframe contains 3 columns of store id, genres, locations
#   labels  : array of labels
# output:
#   df      : dataframe contains 3 columns and its labels
def main_labelling_clustering(df_empty, labels, aff, lnk):
    # print("type of labels:", type(labels))
    # print("labels:-----------", labels)
    df_empty['hc_' + lnk + '_' + aff] = labels
    # df_empty['hc'] = labels
    # print("df_empty:-----------\n", df_empty)
    return df_empty

# Method used: Clustering by hierachy method
# input:
#   vm_values : visitor matrix after formatted and get values
#   df_merge  :
# output:
#
def clustering_by_hierachy(vm_values, vm_first_col_values, store_info_dataset):
    # If linkage is “ward”, only “euclidean” is accepted. If “precomputed”,
    # a distance matrix (instead of a similarity matrix) is needed as input for the fit method.
    # linkage_list = ['ward']
    linkage_list = ['ward', 'complete', 'average', 'single']

    print("vm_values input: \n", vm_values)

    # print("df_merge  input: \n", df_merge)
    # first_three_columns = df_merge
    # first_three_columns = first_three_columns[[store_id, genre_name, area_name]]
    # print("First 3 columns of store_id, genre and area after removing: \n", first_three_columns)
    # df_genre_clusters = first_three_columns.groupby([genre_name]).size().reset_index()
    # # print("list(my_dataframe.columns.values):", list(df_genre_clusters.columns.values))
    # print("df_genre_clusters after  :============== \n", df_genre_clusters.head())

    df_empty = pd.DataFrame({store_id: vm_first_col_values[:, 0]})
    # print("First column of stord_id:\n", df_empty)

    for lnk in linkage_list:
        if lnk=='ward':
            affinity_list = ['euclidean']
        else:
            affinity_list = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']

        for aff in affinity_list:
            lables = labeling_hierachy_cluster(vm_values, aff, lnk)
            df_genre_clusters = main_labelling_clustering(df_empty, lables, aff, lnk)
            # main_labels_df = main_labelling_clustering(first_three_columns, lables)
            # df_genre_clusters['hc_' + lnk + '_' + aff] = main_labels_df

            # main_labels_df = main_labelling_clustering(df_empty, lables)
            # # main_labels_df = main_labelling_clustering(first_three_columns, lables)
            # df_genre_clusters['hc_' + lnk + '_' + aff] = main_labels_df

    # print("df_genre_clusters: \n", df_genre_clusters)


    # Merging with store_info_dataset to get genre name and area name
    df_genre_clusters = pd.merge(store_info_dataset, df_genre_clusters, how='inner', on=[store_id])

    # Get only the first word of area_name
    df_genre_clusters = format_arename_col_first_word(df_genre_clusters).reset_index(drop=True)
    print("Final dataframe of time series and their clusters:\n", df_genre_clusters)
    # df_genre_clusters.to_csv('df_genre_clusters.csv')
    return df_genre_clusters

# Method use: format visitor matrix by adding name "store_id" and resetting index column
# input:
#   vm: vistor_matrix
# output:
#   vm: visitor_matrix after formatting
def format_visitor_matrix(vmt):
    # Adding name "store_id" for visitor matrix
    # print("visitor matrix transposed header:\n", vmt.head())
    # print("list(visitor_matrix.columns.values):", list(vmt.columns.values))
    vmt.index.name = store_id
    # Resetting index coulumn for visitor matrix
    vmt = vmt.reset_index()
    # print("list(visitor_matrix.columns.values):", list(visitor_matrix_transposed.columns.values))
    print("visitor_matrix transposed after formatting: ---- \n", vmt)
    return vmt

# Method use: Merging store id from air_store_id/hpg_store_id and visitor matrix.
#   By this we can get genres and locations, then we will attach it with labels in the later steps.
# This is be used to identify which genre and location belong to which cluster.
# input:
#   visitor_matrix: Matrix of visitors from chosen, filtered time series
#
def get_df_merge_id(store_info_dataset, df_id):
    # Merging
    df_merge_id = pd.merge(store_info_dataset, df_id, how='inner', on=[store_id])

    print("Merging visitor matrix and store id dataset - df_merge_id:\n", df_merge_id)
    return df_merge_id

visitor_matrix_formatted = format_visitor_matrix(visitor_df_transposed)
visitor_matrix_formatted_first_column, visitor_matrix_formatted_values = split_matrix_values(visitor_matrix_formatted)
timeseries_hierachy_clustered = clustering_by_hierachy(visitor_matrix_formatted_values, visitor_matrix_formatted_first_column, store_info_dataset)

# # ============ Step 10: Find the corelation between genres and clusters ==============

# Method use: Find the corelation between genres and clusters
# input:
#   timeseries_hierachy_clustered      : dataframe contains 3 columns of store id, genres, locations and clusters
# output:
#   df      : dataframe contains corelation between genre groups and clusters
def corelation_genre_clusters(df):
    # print("Input dataframe with clustered time series:\n", df)

    df_temp = df

    # Get first column to form up corelation dataframe
    df_first_col = df_temp.groupby([genre_name]).size()
    df_first_col = df_first_col.to_frame(name = 'size').reset_index()
    # print("df_first_col: \n", df_first_col)

    # Create a new empty corelation dataframe
    df_corelation_genre_clusters = pd.DataFrame()
    # Concatenate the new empty corelation dataframe with first column dataframe
    df_corelation_genre_clusters = pd.concat([df_corelation_genre_clusters, df_first_col], axis=1)

    # Loop from the first cluster column to the end
    max_col = df.shape[1]
    for i in range(3, max_col):
        # Group by genre name and cluster columns
        df = df_temp.groupby([genre_name, df_temp.columns[i]]).size()
        # print("df====== 222\n:", df)
        # reset index column
        df = df.to_frame(name = 'size').reset_index()
        # print("df_merge_id:\n", df)

        # Groupby genre again to get maximum size of appearances clusters
        idx = df.groupby([genre_name])['size'].transform(max) == df['size']
        df = df[idx]
        # print("df====== 333:\n", df)

        str_column_name = df.columns[1]
        # print("str_column_name:", str_column_name)

        df = df.groupby(genre_name)[str_column_name].apply(lambda x: ','.join(map(str, x))).reset_index()
        # print("df====== 444 : \n", df)
        df = df[[str_column_name]]
        # print("df====== 555 : \n", df)
        df_corelation_genre_clusters = pd.concat([df_corelation_genre_clusters, df], axis=1)

    print("df_corelation_genre_clusters:\n", df_corelation_genre_clusters)
    return df

for tuple in itertools.product(IMPUTATION_METHOD, MAX_MISSING_PERCENTAGE, RESAMPLING_METHOD, NUM_OF_HC_CLUSTER, SPLIT_GROUPS):
    # print("tuple:", tuple)
    # for item in tuple:
    IMPUTATION_METHOD_ARG, MAX_MISSING_PERCENTAGE_ARG, RESAMPLING_METHOD_ARG, NUM_OF_HC_CLUSTER_ARG, SPLIT_GROUPS_ARG = tuple
    # print(IMPUTATION_METHOD_ARG, MAX_MISSING_PERCENTAGE_ARG, RESAMPLING_METHOD_ARG, NUM_OF_HC_CLUSTER_ARG, SPLIT_GROUPS_ARG)
    # print("tuple:", tuple[0], tuple[1])

    corelation_genre_clusters(timeseries_hierachy_clustered, tuple)
