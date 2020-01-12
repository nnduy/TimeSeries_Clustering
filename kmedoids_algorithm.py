# from sklearn.neighbors import NearestNeighbors
# import numpy as np
# from sklearn.metrics import pairwise_distances
# from sklearn.metrics.pairwise import pairwise_kernels
# from sklearn.metrics.pairwise import cosine_similarity
# import glob, re
# import numpy as np
# import pandas as pd
# import math
# import statistics
#
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
#
#
#
# # clustering_by_optics(visitor_matrix)
#
# # # clustering_by_kmean(visitor_matrix, 9, 5)
# # # clustering_by_kmedoids(visitor_matrix, pairwise_euclidean, 9)
