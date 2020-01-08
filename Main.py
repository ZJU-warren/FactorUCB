import sys; sys.path.append('../')
from Tools.DataTools import *
import DataSetLink as DLSet
import Constant
import random
import time
from B_FactorUCB.ProxyAgent import ProxyAgent


def main():
    n_cluster = 50

    # load data
    W = load_obj(DLSet.social_mat_link % n_cluster)
    user_cluster_df = load_data(DLSet.user_clusterID_link).values
    item_context = load_obj(DLSet.item_context_link)

    # state the agent
    agent = ProxyAgent(
        d=Constant.pca_component, l=0, N=W.shape[0], W=W,
        lambda_1=1, lambda_2=1, alpha_u=0.1, alpha_a=0.1, item_context=item_context,
    )

    # measure
    total_reward = 0
    random_reward = 0
    t_round = 0

    pre_time = time.time()
    avg_time = 0
    batch_size = 100

    log_file_path = DLSet.result_link % 'coLin'
    with open(log_file_path, 'w') as wf:
        wf.write('n_round,ratio' + '\n')

    for i in range(1, 5):
        with open(DLSet.bandit_data_link % i, 'r') as f:
            for line in f:
                log = eval(line)
                if log['bandit_id'] >= user_cluster_df.shape[0]:
                    continue

                # take one log
                t_round += 1
                with open(log_file_path, 'a') as wf:

                    log['bandit_id'] = user_cluster_df[log['bandit_id'], 1]

                    # random choose
                    random_reward += log['arm_reward'][random.choice(list(log['arm_reward'].keys()))]
                    # agent choose
                    total_reward += agent.one_step(log)

                    # record
                    if random_reward != 0:
                        wf.write(str(t_round) + ',' + str(total_reward / random_reward) + '\n')
                    else:
                        wf.write(str(t_round) + ',0' + '\n')

                # print
                if t_round % batch_size == 0:
                    cost_time = time.time() - pre_time
                    pre_time = time.time()
                    n = t_round // batch_size
                    avg_time += (cost_time - avg_time) / (n + 1)
                    print("t_round = {}/437593, cost_time = {}: total_reward = {}, random_reward = {}, ratio = {}"
                          .format(t_round, cost_time, total_reward, random_reward,
                                  total_reward / random_reward))


if __name__ == '__main__':
    main()
