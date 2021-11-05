# @Author: Li
# @FileName: GA_sko_opt.py
# @Time: 2021/10/30 19:48


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sko.GA import GA

# 适应度函数,求取最大值
# 因为GA函数是求最小值，所以我在适应度函数上加一个负号
# GA要求输入维度2维及其以上，所以我传入2个参数，第二维x2不用
a = 20
b = 0.2
c = 2 * np.pi


def schaffer(p):
    """
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    """
    x1, x2 = p
    return -a * np.exp(-b * np.sqrt(sum(np.power(p, 2)) / len(p))) - np.exp(
        sum(np.cos([c * i for i in p])) / len(p)) + a + np.e


# 个体类
ga = GA(func=schaffer, n_dim=2, size_pop=50, max_iter=100, lb=-32.768, ub=32.768, prob_mut=0.001, precision=1e-7)
best_x, best_y = ga.run()
print('best_x:', best_x[0], '\n', 'best_y:', -best_y)

Y_history = pd.DataFrame(ga.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()

# plt.scatter(best_x[0], -best_y, c='r', label='best point')
#
# plt.legend()
# plt.show()
