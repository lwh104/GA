# @Author: Li
# @FileName: GA_self_work.py
# @Time: 2021/10/31 20:24
import math
import random

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pop_size = 300  # 种群数量
low = -32.768  # 自变量下限
up = 32.768  # 自变量上限
chrom_length = 40  # 染色体长度
epoch = 100  # 迭代次数
pc = 0.6  # 交配概率
pm = 0.01  # 变异概率


# 种群：二维列表
# x_y数据点：使用二维列表存储，(N, 2) 2分别代表 x 和 y

# 目标函数
# p: [x, y]
def fun(p: list, a=20, b=0.2, c=2 * np.pi):
    return -a * np.exp(-b * np.sqrt(sum(np.power(p, 2)) / len(p))) - np.exp(
        sum(np.cos([c * i for i in p])) / len(p)) + a + np.e


# 绘制3D动态图
def draw_3D(fig, xy_data, epoch):
    # 清除当前图像
    fig.clf()

    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    ax = Axes3D(fig)
    X, Y = np.mgrid[-40:40:45j, -40:40:45j]
    Z = fun([X, Y])
    # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, alpha=0.5)

    x_data = []
    y_data = []
    z_data = []
    for data in xy_data:
        z_data.append(fun(data))
        x_data.append(data[0])
        y_data.append(data[1])

    ax.scatter3D(x_data, y_data, z_data, c='#FF00FF')
    ax.set_xlabel('X', color='b')
    ax.set_ylabel('Y', color='r')
    ax.set_zlabel('Z', color='g')

    plt.suptitle(f'种群数量：300 染色体长度：40 第{epoch}代', color='g')
    print('z_data: \n', z_data)

    plt.pause(0.2)
    return z_data


# 迭代完成后，绘制散点图
def draw2D(epoch, z_data):
    plt.figure()
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(np.arange(epoch), z_data, '.', color='red')
    plt.title('ACKLEY函数优化过程')
    plt.xlabel('迭代次数')
    plt.ylabel('函数值')
    plt.show()


# 种群初始化，编码
def species_origin(population_size, chromosome_len):
    # 种群二维列表，包含个体与染色体两维
    population = [[]]

    for i in range(population_size):
        # 染色体暂存
        temp = []
        for j in range(chromosome_len):
            # 生成由二进制数组成的染色体
            temp.append(random.randint(0, 1))
        # 染色体添加到种群中
        population.append(temp)
    # 返回种群，种群是二维列表，个体与染色体两维
    return population[1:]


# 测试 species_origin
# back = species_origin(2, 3)
# print()


# 解码，从二进制编码成十进制表现型
def get_decode(population: list, chromosome_len, low, up):
    part_size = chromosome_len // 2
    decode = []
    for i in range(len(population)):
        total_x = 0
        total_y = 0
        for j in range(part_size):
            total_x += population[i][j] * math.pow(2, part_size - 1 - j)
            total_y += population[i][j + part_size] * math.pow(2, part_size - 1 - j)
        # 对 total 进行映射到区间[-32.768, 32.768]
        total_x = total_x / (np.power(2, part_size) - 1) * abs(low - up) - abs(low - up) / 2
        total_y = total_y / (np.power(2, part_size) - 1) * abs(low - up) - abs(low - up) / 2
        decode.append([total_x, total_y])

    # 返回解码后的二维列表 点集
    return decode


# 测试 get_decode
# back = species_origin(2, 10)
# encode = get_decode(back, 10)
# print()


# 遗传算法依据原则：适应度越高，被选择的机会越高，而适应度越低，被选择的机会越低
# 目标函数是求最小值，因此应该是目标函数值越小，越容易选中，因此反转目标函数值，使得函数值小的适应度大
def get_fitness(decode, obj=fun):
    """

    :param decode:
    :param obj:
    :return: 返回fitness
    """
    # i: [x, y]
    pred = [obj(i) for i in decode]
    max_fit = np.max(pred)
    return max_fit - pred + 1e-3


def select(fitness, binary_population, decode, pop_size):
    """
    随机选择序列，使用choice函数有放回的选取

    :param fitness: 适应度
    :param pop_size: 种群大小
    :return: 新种群适应度，新种群基因型，新种群表现型
    """
    idx = np.random.choice(np.arange(len(fitness)), size=pop_size, replace=True,
                           p=list(fitness / np.array(fitness).sum()))
    new_population_fitess = []
    new_binary_population = []
    new_decode = []
    for i in idx:
        new_population_fitess.append(fitness[i])
        new_binary_population.append(binary_population[i])
        new_decode.append(decode[i])
    return new_population_fitess, new_binary_population, new_decode


# 测试 cal_obj_value、select
# back = species_origin(pop_size, chrom_length)
# population_decode = get_decode(back, 10)
# pred = get_fitness(population_decode, fun)
# new_population = select(pred, pop_size)
# print()


# 通过随机选取基因交配点来生成交叉子代基因，交配点前的基因来自与父本基因，交配点后的基因来自与母本基因
# 变异通常是将交叉子代中的某一基因发生随机改变，通常是直接改变DNA的一个二进制位(0-1)
def crossover_and_mutation(population, chrom_length, pop_size, pc, pm):
    """

    :param population: 种群
    :param pop_size: 种群规模
    :param pc: 交叉概率
    :param pm: 变异概率
    :return: 新种群
    """
    origin_population = population.copy()
    child_population = []
    for index, father in enumerate(population):
        child = father.copy()
        if np.random.rand() < pc:
            # print(f'第{index}个索引交叉')
            new_select = np.random.randint(pop_size)
            while index == new_select:
                new_select = np.random.randint(pop_size)
            mother = list(origin_population[new_select])
            cross_points = np.random.randint(0, chrom_length)
            child[cross_points:] = mother[cross_points:]
            child = mutate(child, pm)
        child_population.append(child)
    for i in child_population:
        origin_population.append(i)

    return origin_population


# 变异操作
def mutate(child, pm):
    if np.random.rand() < pm:
        # 随机产生变异索引
        mutate_index = np.random.randint(0, chrom_length)
        # print(f'第{mutate_index}个索引变异')
        # 变异点二进制反转
        child[mutate_index] = child[mutate_index] ^ 1
        return child
    return child


# 测试 crossover_and_mutation
# origin = species_origin(4, 10)
# origin2 = crossover_and_mutation(origin, 4, pc, pm)
# decode = get_decode(back, 10)
# fitness, new_select = get_fitness_and_select(encode, fn, pop_size)
# new_population = crossover_and_mutation(pre, pop_size, pc, pm)
# print()

def train(binary_population, chrom_length, epoch, pc, pm, low, up):
    global epoch_xy_population
    z_data = []
    fig = plt.figure()
    # 打开交互模式
    plt.ion()
    for i in range(epoch):
        print(f'第{i}次迭代')

        # binary_population decode fitness 顺序一一对应
        # 解码，二进制转为十进制，并映射到指定区间 (1000,2)
        decode = get_decode(binary_population, chrom_length, low, up)
        # 计算表现型适应度
        fitness = get_fitness(decode, fun)
        # 根据适应度概率选择新的种群
        new_population_fitness, binary_population, epoch_xy_population = select(fitness, binary_population, decode,
                                                                                pop_size)
        # 交叉和变异
        binary_population = crossover_and_mutation(binary_population, chrom_length, pop_size, pc, pm)

        z_epoch = draw_3D(fig, epoch_xy_population, i)
        z_data.append(z_epoch)
        print('x_y坐标: \n', epoch_xy_population)

    # 关闭交互模式
    plt.ioff()
    plt.show()

    return epoch_xy_population, z_data


if __name__ == '__main__':
    population = species_origin(pop_size, chrom_length)
    last_x_y, z_data = train(population, chrom_length, epoch, pc, pm, low, up)
    draw2D(epoch, z_data)
    print('最终结果: \n', last_x_y)
