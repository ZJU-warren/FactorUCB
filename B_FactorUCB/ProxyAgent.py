import sys; sys.path.append('../')
from tools import *
from B_FactorUCB.FactorUCB import FactorUCB


class ProxyAgent:
    def __init__(self, d, l, N, W, lambda_1, lambda_2, alpha_u, alpha_a):
        self.model = FactorUCB(d, l, N, W, lambda_1, lambda_2, alpha_u, alpha_a)
        # self.item_context = item_context
        # self.item_pool = set()

    def one_step(self, data):
        item_list = []
        item_xs = []
        for arm in data['arm_reward'].keys():
            item_list.append(arm)
            item_xs.append(np.array(data['arm_context'][arm]))
        rec = self.model.decide(item_list, item_xs, data['bandit_id'])
        self.model.update(data['arm_reward'][rec])
        reward = data['arm_reward'][rec]
        return reward
