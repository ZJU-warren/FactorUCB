import pandas as pd
import os
from BanditData import BanditData
import chardet
import random
import numpy as np
from scipy.sparse.linalg import svds
import re
import scipy
import DataSetLink as DLSet
from Tools.DataTools import *
from sklearn.decomposition import TruncatedSVD


def get_encoding(file):
    # 二进制方式读取，获取字节数据，检测类型
    with open(file, 'rb') as f:
        return chardet.detect(f.read())['encoding']


def init_variable(user_tag_df):
    '''
    :param user_tag_df: dataframe of file: '/user_taggedartists-timestamps.dat'
    :return: user_map: 不连续的user_id到连续的user_id的映射
    :return: item_map: 不连续的item_id到连续的item_id的映射
    :return: user_selected: dict : user_selected[user_id]: user点击过的item
    '''
    # 生成 user_map, item_map
    user_map = {}
    item_map = {}
    tag_map = {}
    uid = 0
    iid = 0
    tid = 0

    n_round = 0
    for index, row in user_tag_df.iterrows():
        n_round += 1
        user_id = int(row[0])
        item_id = int(row[1])
        tag_id = row[2]
        # print(user_id, item_id, tag_id)
        if user_id not in user_map.keys():
            user_map[user_id] = uid
            uid += 1
        if item_id not in item_map.keys():
            item_map[item_id] = iid
            iid += 1
        if tag_id not in tag_map.keys():
            tag_map[tag_id] = tid
            tid += 1

    # 生成user_selected_set
    user_selected = {}  # 每个用户点击过的商品
    for index, row in user_tag_df.iterrows():
        user_id = user_map[int(row[0])]
        item_id = item_map[int(row[1])]
        if user_id not in user_selected.keys():
            user_selected[user_id] = []
        user_selected[user_id].append(item_id)
    for user, items in user_selected.items():
        user_selected[user] = set(user_selected[user])

    return user_map, item_map, tag_map, user_selected


def get_context(user_tag_df, n_tag, user_map, item_map, tag_map, n_train, k=25):
    # 初始化tag矩阵：
    n_user = len(user_map)
    n_item = len(item_map)
    user_tags = np.zeros((n_user, n_tag))
    item_tags = np.zeros((n_item, n_tag))
    i = 0
    for index, row in user_tag_df.iterrows():
        i += 1
        print("%d is handle" % i)
        if i > n_train:
            break
        user_id = user_map[int(row[0])]
        item_id = item_map[int(row[1])]
        tags = tag_map[row[2]]
        user_tags[user_id][tags] += 1
        item_tags[item_id][tags] += 1

    # 对user_tags和item_tags共同进行tf-idf
    tags = np.concatenate((user_tags, item_tags), axis=0)
    # tags_sparse = scipy.sparse.csc_matrix(tags)                           # modify 1
    #
    tags_sparse = scipy.sparse.csc_matrix(item_tags)
    context, sigma, rig = scipy.sparse.linalg.svds(tags_sparse, k=k)
    # print(context)
    # transformer = TfidfTransformer()
    # tfidf = transformer.fit_transform(tags)
    # PCA
    # pca = TruncatedSVD(n_components=pca_component)
    # context = pca.fit_transform(item_tags)

    return context, context


if __name__ == '__main__':

    '''不同数据集的设置'''
    arg_name = 'delicious'
    logs_filename = DLSet.logs_filename
    social_filename = DLSet.social_filename
    tags_filename = DLSet.tags_filename
    tag_df_col_name = 'value'

    split_tags = False
    n_arm_set = 25          # 每个log数据的arm_set大小
    train_size = 0.2        # 用于热启动(计算context)的数据集比例
    pca_component = 25      # context的维度
    batch_size = 100000
    compute_social = False

    '''读取logs.dat：'''
    user_tag_df = load_data(logs_filename, '\t')
    pd.DataFrame.sort_values(user_tag_df, by='timestamp', inplace=True)

    '''初始化变量'''
    user_map, item_map, tag_map_origin, user_selected = init_variable(user_tag_df)
    print('_________finish____________')
    pd.DataFrame.from_dict(tag_map_origin, orient='index', columns=['tagID_new']).to_csv(
        DLSet.map_link % 'tagID')
    pd.DataFrame.from_dict(user_map, orient='index', columns=['userID_new']).to_csv(
        DLSet.map_link % 'userID')
    pd.DataFrame.from_dict(item_map, orient='index', columns=['itemID_new']).to_csv(
        DLSet.map_link % 'itemID')
    n_user = len(user_map)
    n_item = len(item_map)
    n_tag_origin = len(tag_map_origin)
    n_train = round(len(user_tag_df) * train_size)

    user_context, item_context \
        = get_context(user_tag_df, n_tag_origin, user_map, item_map, tag_map_origin, n_train, pca_component)
    store_obj(item_context, DLSet.item_context_link)

    '''生成bandit data'''
    logs = []
    i = 0
    total_item = set(item_map.values())
    for index, row in user_tag_df.iterrows():
        i += 1
        if i < n_train:
            continue
        if i >= n_train and i % 200 == 0:
            print("log [%d]" % (i - n_train))

        # 读取dataframe内容
        arm_context = {}
        arm_true_reward = {}
        rewards = {}
        bandit_context = {'tags': {}, 'tags_reward': {}}
        user_id = user_map[int(row[0])]
        selected_item_id = item_map[int(row[1])]
        t = int(row[3])

        # 构建arm信息
        # bandit_context['context'] = list(user_context[user_id])
        arm_set = list(random.sample(total_item - user_selected[user_id], n_arm_set-1))
        arm_set.append(selected_item_id)

        for item in arm_set:
            # arm_context[item] = list(item_context[item])
            # arm_context[item] = list(np.zeros(pca_component))
            if item == selected_item_id:
                rewards[item] = 1
                arm_true_reward[item] = 1
            else:
                rewards[item] = 0
                arm_true_reward[item] = 0

        tags = tag_map_origin[row[2]]
        bandit_context['tags'][selected_item_id] = [tags]
        bandit_context['tags_reward'][selected_item_id] = 1.0
        bandit_data = BanditData(timestamp=t, arm_reward=rewards, arm_context=arm_context,
                                 arm_true_reward=arm_true_reward, bandit_id=user_id, bandit_context=bandit_context)
        logs.append(str(bandit_data.__dict__) + '\n')

    # write data
    print('len logs : ', len(logs))
    batch_write(logs, DLSet.bandit_data_link)


