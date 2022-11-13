import matplotlib.pyplot as plt


def plot_average_queue(avg_q, episode, time):
    plt.figure(1)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Average Queue')
    plt.plot(avg_q)
    plt.savefig('record/queue_{0}_episodes{1}.png'.format(time, episode))
