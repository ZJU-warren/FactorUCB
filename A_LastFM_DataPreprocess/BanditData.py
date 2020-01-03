import sys; sys.path.append('../')

class BanditData(object):
    def __init__(self, timestamp=-1, bandit_id=-1, arm_reward=None,
                 arm_true_reward=None, arm_context=None, bandit_context=None):

        self.timestamp = timestamp
        self.bandit_id = bandit_id
        self.arm_reward = arm_reward
        self.arm_true_reward = arm_true_reward
        self.arm_context = arm_context
        self.bandit_context = bandit_context
