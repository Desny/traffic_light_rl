from absl import app
from absl import flags
from environment.env import SumoEnv
from agents.dqn import DqnAgent
from replay import ReplayBuffer
import torch
from datetime import datetime
import pandas as pd
import pickle
import math
from plots import plot_max_q_change, plot_max_q_change_reward, plot_average_queue

FLAGS = flags.FLAGS
flags.DEFINE_integer('skip_range', 50, 'time(seconds) range for skip randomly at the beginning')
flags.DEFINE_float('simulation_time', 10000, 'time for simulation')
flags.DEFINE_integer('yellow_time', 2, 'time for yellow phase')
flags.DEFINE_integer('delta_rs_update_time', 10, 'time for calculate reward')
flags.DEFINE_string('reward_fn', 'choose-min-waiting-time', '')
flags.DEFINE_string('net_file', 'nets/2way-single-intersection/single-intersection.net.xml', '')
flags.DEFINE_string('route_file', 'nets/2way-single-intersection/single-intersection-vhvh.rou.xml', '')
flags.DEFINE_bool('use_gui', False, 'use sumo-gui instead of sumo')
flags.DEFINE_integer('num_episodes', 701, '')
flags.DEFINE_string('network', 'dqn', '')
flags.DEFINE_string('mode', 'train', '')
flags.DEFINE_float('eps_start', 1.0, '')
flags.DEFINE_float('eps_end', 0.1, '')
flags.DEFINE_integer('eps_decay', 83000, '')
flags.DEFINE_integer('target_update', 3000, '')
flags.DEFINE_string('network_file', '', '')
flags.DEFINE_float('gamma', 0.95, '')
flags.DEFINE_integer('batch_size', 32, '')

device = "cuda" if torch.cuda.is_available() else "cpu"

time = str(datetime.now()).split('.')[0].split(' ')[0]
time = time.replace('-', '')


def main(argv):
    del argv
    env = SumoEnv(net_file=FLAGS.net_file,
                  route_file=FLAGS.route_file,
                  skip_range=FLAGS.skip_range,
                  simulation_time=FLAGS.simulation_time,
                  yellow_time=FLAGS.yellow_time,
                  delta_rs_update_time=FLAGS.delta_rs_update_time,
                  reward_fn=FLAGS.reward_fn,
                  use_gui=FLAGS.use_gui
                  )
    replay_buffer = ReplayBuffer(capacity=20000)

    agent = None
    if FLAGS.network == 'dqn':
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n
        agent = DqnAgent(FLAGS.mode, replay_buffer, FLAGS.target_update, FLAGS.gamma, FLAGS.eps_start, FLAGS.eps_end,
                         FLAGS.eps_decay, input_dim, output_dim, FLAGS.batch_size, FLAGS.network_file)

        # record
        record_episode = []
        record_learn_step = []
        record_gamma = []
        record_queue = []
        record_q_value = []
        data = {'episode': record_episode,
                'learn_step': record_learn_step,
                'gamma': record_gamma,
                'queue': record_queue,
                'q_value': record_q_value}
        data = pd.DataFrame(data)
        data.to_csv('record/exp_{}.csv'.format(time), index=False)

    for episode in range(FLAGS.num_episodes):
        initial_state = env.reset()
        # agent.learn_steps = 0
        env.train_state = initial_state
        done = False
        invalid_action = False
        while not done:
            state = env.compute_state()
            action = agent.select_action(state, replay_buffer.steps_done, invalid_action)
            next_state, reward, done, info = env.step(action)
            if info['do_action'] is None:
                invalid_action = True
                continue
            invalid_action = False
            # print('action:', action)
            # print()
            # print(env.sumo.simulation.getTime(), state, next_state, reward, done, info)

            replay_buffer.add(env.train_state, env.next_state, reward, info['do_action'])
            if not agent.update_gamma:
                agent.learn()
                if agent.learn_steps > 0:
                    record_episode.append(episode)
                    record_learn_step.append(agent.learn_steps)
                    record_gamma.append(agent.gamma)
                    record_queue.append(env.traffic_signal.compute_queue())
                    record_q_value.append(agent.q_value_batch_avg)
            else:
                agent.learn_gamma()

            if FLAGS.mode == 'eval':
                with open('record/eval_state_values_{}.txt'.format(time), 'a') as f:
                    f.writelines('time:' + str(env.sumo.simulation.getTime()))
                    f.writelines('\nstate:' + str(state))
                    state = torch.from_numpy(state).unsqueeze(0).to(device)
                    f.writelines('\nstate_action_values:' + str(agent.policy_net(state).tolist()) + '\n\n')

            if reward is not None:
                with torch.no_grad():
                    with open('record/record_{0}.txt'.format(time), 'a') as f:
                        f.writelines('episode:{0}\n'.format(episode))
                        f.writelines('simulation step:{0}\n'.format(env.sumo.simulation.getTime()))
                        f.writelines('all_state_action_values(policy):' + str(agent.policy_net(torch.from_numpy(env.train_state)).tolist()) + '\n')
                        f.writelines('state:' + str(env.train_state) + '\n')
                        f.writelines('next_state:' + str(env.next_state) + '\n')
                        f.writelines('action:' + str(info['do_action']) + '\n')
                        f.writelines('reward:' + str(reward) + '\n')
                        f.writelines('--------------\n')

        env.close()
        if FLAGS.mode == 'train':
            if episode != 0 and episode % 10 == 0:
                torch.save(agent.policy_net.state_dict(), 'weights/weights_{0}_{1}.pth'.format(time, episode))

            # record experiment data
            data = {'episode': record_episode,
                    'learn_step': record_learn_step,
                    'gamma': record_gamma,
                    'queue': record_queue,
                    'q_value': record_q_value}
            data = pd.DataFrame(data)
            data.to_csv('record/exp_{}.csv'.format(time), mode='a', index=False, header=False)
            # clear record per episode
            record_episode = []
            record_learn_step = []
            record_gamma = []
            record_queue = []
            record_q_value = []

        # with open('record/record_{0}_params.txt'.format(time), 'a') as f:
        #     f.writelines('i_episode:{0}'.format(episode))
        #     f.writelines('\nl1.weight:\n')
        #     f.writelines(str(agent.policy_net.state_dict()['l1.weight'].tolist()))
        #     f.writelines('\nl1.bias:\n')
        #     f.writelines(str(agent.policy_net.state_dict()['l1.bias'].tolist()))
        #     f.writelines('\nl2.weight:\n')
        #     f.writelines(str(agent.policy_net.state_dict()['l2.weight'].tolist()))
        #     f.writelines('\nl2.bias:\n')
        #     f.writelines(str(agent.policy_net.state_dict()['l2.bias'].tolist()))

        print('i_episode:', episode)
        print('eps_threshold = :', FLAGS.eps_end + (FLAGS.eps_start - FLAGS.eps_end) *
              math.exp(-1. * replay_buffer.steps_done / FLAGS.eps_decay))
        print('learn_steps:', agent.learn_steps)
        print('gamma:', agent.gamma)

        if FLAGS.mode == 'train' and episode != 0 and episode % 100 == 0:
            plot_average_queue(env.avg_queue, episode, time)

        # if episode % 100 == 0:
        #     plot_max_q_change_reward(agent.max_q_change_list, agent.max_reward_list)

    # if FLAGS.mode == 'train':
    #     with open('record/replay_{0}.pkl'.format(time), 'wb') as f:
    #         pickle.dump(replay_buffer.storage, f)
    # plot_max_q_change(agent.max_q_change_list)


if __name__ == '__main__':
    app.run(main)
