from tools import *
from MathTools import *
import DataSetLink as DLSet
import Constant
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
import sys; sys.path.append('../')


def gen_ut_it_matrix(df, data_count):
    ut = np.zeros((data_count['userID'], data_count['tagID']))
    it = np.zeros((data_count['itemID'], data_count['tagID']))
    for u, i, t, time in df.values:
        ut[u][t] += 1
        it[i][t] += 1
    return ut, it


def gen_context(df, data_count, ratio, n_components):
    sub_df = df.sample(frac=ratio)
    ut, it = gen_ut_it_matrix(df, data_count)
    sub_ut, sub_it = gen_ut_it_matrix(sub_df, data_count)

    # choose the first n p-components by pca
    pca = TruncatedSVD(n_components=n_components)
    pca.fit(join(sub_ut, sub_it))
    context = pca.transform(join(ut, it))

    return sub_df, context[:data_count['userID']], context[data_count['userID']:]


def main():
    # load data and filter the errors
    logs = load_data(DLSet.logs_filename, '\t').sample(frac=0.0002)
    logs.columns = ['userID', 'itemID', 'tagID', 'timestamp']
    logs = logs[logs['timestamp'] > 0]

    # reset the keyword field
    for each in logs.columns:
        logs = map_id(logs, each, DLSet.map_link % each)
    logs = logs.sort_values('timestamp', ascending=True)

    # count the data shape
    data_count = cal_data_count(logs)

    # generate the sub_df and context
    sub_logs, user_context, item_context =\
        gen_context(logs, data_count, Constant.train_size, Constant.pca_component)

    # generate user_selected = {user : set(selected item)}
    user_selected = defaultdict(set)
    for u, i, t, time in logs.values:
        user_selected[u].add(i)

    # store
    print(data_count)
    logs.to_csv(DLSet.logs_link, index=False)
    sub_logs.to_csv(DLSet.sub_logs_link, index=False)
    store_obj(user_context, DLSet.user_context_link)
    store_obj(item_context, DLSet.item_context_link)
    store_obj(user_selected, DLSet.user_selected_link)


if __name__ == '__main__':
    main()
