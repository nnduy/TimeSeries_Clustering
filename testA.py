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



ALGORITHMS = ['DBSCAN','HC']
ALGORITHMS_ARG = ['HC']
# RES_DATASET = ['air', 'hpg']
RES_DATASET_ARG = ['air']
RESAMPLING_METHOD_ARG = ['under']
# SPLIT_GROUPS = [3, 9]
SPLIT_GROUPS_ARG = [9]
# IMPUTATION_METHOD = ['median', 'mean', 'linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'barycentric',
#           'krogh', 'polynomial', 'spline', 'piecewise_polynomial', 'from_derivatives', 'pchip', 'akima']
IMPUTATION_METHOD = ['median', 'mean']
IMPUTATION_METHOD_ARG = ['median']
# MAX_MISSING_PERCENTAGE = [25, 50, 75, 100]
MAX_MISSING_PERCENTAGE_ARG = [100]

# NUM_OF_HC_CLUSTER = [3, 9]
NUM_OF_HC_CLUSTER_ARG = [3]

for tuple in itertools.product(ALGORITHMS_ARG, RES_DATASET_ARG, RESAMPLING_METHOD_ARG, SPLIT_GROUPS_ARG,
                              IMPUTATION_METHOD_ARG, MAX_MISSING_PERCENTAGE_ARG):
    # print("tuple:", tuple)
    # for item in tuple:
    # IMPUTATION_METHOD_ARG, MAX_MISSING_PERCENTAGE_ARG, RESAMPLING_METHOD_ARG, NUM_OF_HC_CLUSTER_ARG, SPLIT_GROUPS_ARG = tuple
    # print(IMPUTATION_METHOD_ARG, MAX_MISSING_PERCENTAGE_ARG, RESAMPLING_METHOD_ARG, NUM_OF_HC_CLUSTER_ARG, SPLIT_GROUPS_ARG)
    print("tuple:", tuple[0], tuple[1])

# print("type tuple:", type(tuple))


