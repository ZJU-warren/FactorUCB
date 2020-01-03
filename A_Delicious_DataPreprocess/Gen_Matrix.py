import sys; sys.path.append('../')
from tools import *
import pandas as pd
import DataSetLink as DLSet
from sklearn.cluster import SpectralClustering


def graph_cluster(mat, k=200):
    # spectral cluster
    model = SpectralClustering(n_clusters=k, gamma=0.1, affinity='precomputed')
    model.fit(mat)
    y_pred = model.labels_
    for i in range(k):
        print(i, ':', np.sum(y_pred == i))
    # store the cluster result
    with open(DLSet.user_clusterID_link, 'w') as f:
        f.write('userID, clusterID\n')
        for i in range(y_pred.shape[0]):
            f.write('{}, {}\n'.format(i, y_pred[i]))

    # generate the new matrix
    social_mat = np.zeros((k, k))
    org_num_user = mat.shape[0]
    for i in range(org_num_user):
        for j in range(org_num_user):
            social_mat[y_pred[i], y_pred[j]] += mat[i, j]
    social_mat = social_mat / (social_mat.sum(axis=0))
    return social_mat


def main():
    # load social data
    social = load_data(DLSet.social_filename, '\t')[['A', 'B']]
    map_user = load_data(DLSet.map_link % 'userID')
    user_num = len(set(social['A'].values) | set(social['B'].values))

    # replace the userID by map_userID and generate the social matrix
    social_mat = gen_social_matrix(social, map_user, user_num)
    social_mat = graph_cluster(social_mat)
    store_obj(social_mat, DLSet.social_mat_link)


if __name__ == '__main__':
    main()
