import sys; sys.path.append('../')
from Tools.MathTools import *


class FactorItem:
    def __init__(self, lambda_2, d_, l, x):
        # basic setting
        self.d_ = d_; self.l = l
        self.lambda_2 = lambda_2
        self.x = x

        # init the parameters                                       # line 7
        self.C = lambda_2 * np.identity(l)
        self.d = np.zeros(l)
        self.v = np.zeros(l)

        # opt the calculation
        self.AI = None
        self.vec_0X0V_WT = None
        self.CI = np.linalg.inv(self.C)

    def calculate(self, user, WT, N):
        temp = np.zeros((self.d_ + self.l, N))
        temp[:, user] = join(self.x, self.v)
        self.vec_0X0V_WT = vec(temp.dot(WT))
        return join(self.x, self.v).T, self.vec_0X0V_WT, self.CI

    def update(self, txw, tvw, r):
        self.C += tvw.dot(tvw.T)                                    # line 14
        self.d += tvw.dot(r - self.x.T.dot(txw))                    # line 15

        self.CI = np.linalg.inv(self.C)
        # self.CI = fast_inverse(self.CI, tvw)
        self.v = self.CI.dot(self.d)                                # line 16

        self.v = normalize(join(self.x, self.v))[self.d_:]          # line 17
