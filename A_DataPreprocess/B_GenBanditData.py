import sys; sys.path.append('../')
import DataSetLink as DLSet
from Tools.DataTools import *
import random
import Constant
from BanditData import BanditData


# load data
def load_basic_data():
    logs = load_data(DLSet.logs_link)
    sub_logs = load_data(DLSet.sub_logs_link)
    user_context = load_obj(DLSet.user_context_link)
    item_context = load_obj(DLSet.item_context_link)
    user_selected = load_obj(DLSet.user_selected_link)
    return logs, sub_logs, user_context, item_context, user_selected


# generate logs
def gen_data(diff_logs, data_count, user_context, item_context, user_selected):
    result_logs = []
    diff_logs = diff_logs.sort_values(['timestamp'], ascending=True)

    # generate bandit data
    items_pool = set([i for i in range(data_count['itemID'])])
    cnt = 0
    for u, i, t, time in diff_logs.values:
        arm_context = {}
        bandit_context = {}
        arm_true_reward = {}
        rewards = {}

        # set info and sample arm_set
        bandit_context['context'] = list(user_context[u])
        bandit_context['tags'] = t
        bandit_context['tag_reward'] = 1
        arm_set = list(random.sample(items_pool - user_selected[u], Constant.n_arm_set - 1))
        arm_set.append(i)

        # load the item context and set the reward
        for item in arm_set:
            # arm_context[item] = list(item_context[item])
            arm_true_reward[item] = 1 if item == i else 0
            rewards[item] = 1 if item == i else 0

        # format as banditData
        bandit_data = BanditData(timestamp=t, arm_reward=rewards, arm_context=arm_context,
                                 arm_true_reward=arm_true_reward, bandit_id=u, bandit_context=bandit_context)
        result_logs.append(str(bandit_data.__dict__) + '\n')
        cnt += 1
        if cnt % 1000 == 0:
            print("%d / %d" % (cnt, diff_logs.shape[0]))
    return result_logs


def main():
    # load basic data
    logs, sub_logs, user_context, item_context, user_selected = load_basic_data()

    # calculate the diff_logs and data_count
    diff_logs = gen_diff_set(logs, sub_logs)
    data_count = cal_data_count(logs)

    # generate the bandit data
    result_logs = gen_data(diff_logs, data_count, user_context, item_context, user_selected)
    batch_write(result_logs, DLSet.bandit_data_link)


if __name__ == '__main__':
    main()


# {'userID': 1867, 'itemID': 69223, 'tagID': 40897, 'timestamp': 104093}
