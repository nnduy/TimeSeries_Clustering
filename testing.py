from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import cosine_similarity
import glob, re
import numpy as np
import pandas as pd
from pandas import Timestamp
from sklearn import *
from datetime import datetime
from xgboost import XGBRegressor
import math
import statistics

from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plt

# import sklearn
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import gmean
from scipy.stats.mstats import gmean
# from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.metrics import pairwise_distances
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# # nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
# # distances, indices = nbrs.kneighbors(X)
# # print(indices)
# # print(distances)
# # # array([[0, 1],
# # #        [1, 0],
# # #        [2, 1],
# # #        [3, 4],
# # #        [4, 3],
# # #        [5, 4]]...)
# # # distances
# # # array([[0.        , 1.        ],
# # #        [0.        , 1.        ],
# # #        [0.        , 1.41421356],
# # #        [0.        , 1.        ],
# # #        [0.        , 1.        ],
# # #        [0.        , 1.41421356]])
#
# # X = np.array([[2, 3], [3, 5], [5, 8]])
# print('matrix X :\n {} \n'.format(X))
#
# pairwise = pairwise_distances(X, metric='manhattan')
# print('pairwise_distances :\n {} \n'.format(pairwise))
#
# similarities = cosine_similarity(X)
# print('pairwise cosine_similarity:\n {}\n'.format(similarities))

# import matplotlib.patches as mpatches
# red_patch = mpatches.Patch(color='red', label='The red data')
# blue_patch = mpatches.Patch(color='blue', label='The blue data')
# handles1=[red_patch, blue_patch]
# print(type(handles1))

    # df = pd.DataFrame({'a': [2, 4, 8, 0],
    #                     'b': [2, 0, 0, 0],
    #                     'c': [10, 2, 1, 8]},
    #                    index=['falcon', 'dog', 'spider', 'fish'])
    # # df1 = df[['a','b']]
    # class_member_mask = [True, False, True]
    # current_group_index = df.columns[class_member_mask]
    # current_group_columns = df[current_group_index]
    # print(current_group_index)
    # print(current_group_columns)

# scenarios = ['scen-1', 'scen-2']
#
# fig, ax = plt.subplots()
#
# for index, item in enumerate(scenarios):
#     df = pd.DataFrame({'A' : np.random.randn(4)})
#     print df
#     df.plot(ax=ax)
#
# plt.ylabel('y-label')
# plt.xlabel('x-label')
# plt.title('Title')
# plt.show()

import random

def DTWDistance(s1, s2,w):
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

def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):

        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2

    return math.sqrt(LB_sum)

def k_means_clust(data,num_clust,num_iter,w=5):
    centroids=random.sample(data,num_clust)
    # print("centroids:", centroids)
    counter=0
    for n in range(num_iter):
        counter+=1
        print (counter)
        assignments={}
        #assign data points to clusters
        for ind,i in enumerate(data):
            print("======================",i)
            print("length",len(i))
            # It acts as an unbounded upper value for comparison.
            # This is useful for finding lowest values for something.
            min_dist=float('inf')
            closest_clust=None
            for c_ind,j in enumerate(centroids):
                if LB_Keogh(i,j,5)<min_dist:
                    cur_dist=DTWDistance(i,j,w)
                    if cur_dist<min_dist:
                        min_dist=cur_dist
                        closest_clust=c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust]=[]

        #recalculate centroids of clusters
        for key in assignments:
            clust_sum=0
            for k in assignments[key]:
                clust_sum=clust_sum+data[k]
            centroids[key]=[m/len(assignments[key]) for m in clust_sum]

    return centroids

train = np.genfromtxt('datasets/train.csv', delimiter='\t')
test = np.genfromtxt('datasets/test.csv', delimiter='\t')
data=np.vstack((train[:,:-1],test[:,:-1]))

print("test:", test)
print("type of data test:", type(test))

centroids=k_means_clust(data,4,5,4)
print("type of centroids:", type(centroids))
print("centroids:", centroids)
print("centroids length:", len(centroids))

# for i in centroids:
    # print("======================",i)
    # print("length",len(i))
    # plt.plot(i)

# plt.show()
