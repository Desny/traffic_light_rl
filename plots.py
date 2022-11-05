import matplotlib.pyplot as plt
import numpy as np


def plot_max_q_change(max_q_change_list):
    period_max_q = []
    period = 1000
    i = 0
    while i+period <= len(max_q_change_list):
        s = np.mean(max_q_change_list[i:i+period])
        period_max_q.append(s)
        i += 1
    # x = [i+period for i in range(len(max_q_change_list)-period)]
    x = [i for i in range(len(period_max_q))]
    fig, ax = plt.subplots()
    ax.plot(x, period_max_q)
    # plt.title('100 episodes')
    plt.show()


def plot_max_q_change_reward(max_q_change_list, reward_list):
    period_max_q = []
    period_reward = []
    period = 1000
    i = 0
    while i+period <= len(max_q_change_list):
        s = np.mean(max_q_change_list[i:i+period])
        period_max_q.append(s)
        r = np.mean(reward_list[i:i+period])
        period_reward.append(r)
        i += 1
    # x = [i+period for i in range(len(max_q_change_list)-period)]
    x = [i for i in range(len(period_max_q))]
    fig, ax = plt.subplots()
    ax.plot(x, period_max_q, label='max_q')
    ax.plot(x, period_reward, label='max_reward')
    # ax.scatter(target_update_x, target_update_y, marker='*', color='r')
    plt.legend(loc='best')
    # plt.title('100 episodes')
    plt.show()


def plot_average_queue(avg_q, episode, time):
    plt.figure(1)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Average Queue')
    plt.plot(avg_q)
    plt.savefig('record/queue_{0}_episodes{1}_095_gamma_099_SGD.png'.format(time, episode))
    # plt.show()
