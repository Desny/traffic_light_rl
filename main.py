from absl import app
from absl import flags
from environment.env import SumoEnv
from agents.dqn import DqnAgent
from replay import ReplayBuffer
import torch
from datetime import datetime
import math
from plots import plot_average_queue

FLAGS = flags.FLAGS
flags.DEFINE_integer('skip_range', 10, 'time(seconds) range for skip randomly at the beginning')
flags.DEFINE_float('simulation_time', 10000, 'time for simulation')
flags.DEFINE_integer('yellow_time', 2, 'time for yellow phase')
flags.DEFINE_integer('delta_rs_update_time', 10, 'time for calculate reward')
flags.DEFINE_string('reward_fn', 'choose-min-waiting-time', '')
flags.DEFINE_string('net_file', 'nets/2way-single-intersection/single-intersection.net.xml', '')
flags.DEFINE_string('route_file', 'nets/2way-single-intersection/single-intersection-vhvh.rou.xml', '')
flags.DEFINE_bool('use_gui', False, 'use sumo-gui instead of sumo')
flags.DEFINE_integer('num_episodes', 601, '')
flags.DEFINE_string('network', 'dqn', '')
flags.DEFINE_string('mode', 'train', '')
flags.DEFINE_float('eps_start', 1.0, '')
flags.DEFINE_float('eps_end', 0.1, '')
flags.DEFINE_integer('eps_decay', 83000, '')
flags.DEFINE_integer('target_update', 3000, '')
flags.DEFINE_string('network_file', '', '')
flags.DEFINE_float('gamma', 0.95, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_bool('use_sgd', True, 'Training with the optimizer SGD or RMSprop')

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
                  mode=FLAGS.mode,
                  use_gui=FLAGS.use_gui,
                  )
    replay_buffer = ReplayBuffer(capacity=20000)

    agent = None
    if FLAGS.network == 'dqn':
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n
        agent = DqnAgent(FLAGS.mode, replay_buffer, FLAGS.target_update, FLAGS.gamma, FLAGS.use_sgd, FLAGS.eps_start,
                         FLAGS.eps_end, FLAGS.eps_decay, input_dim, output_dim, FLAGS.batch_size, FLAGS.network_file)

    for episode in range(FLAGS.num_episodes):
        initial_state = env.reset()
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

            if FLAGS.mode == 'train':
                replay_buffer.add(env.train_state, env.next_state, reward, info['do_action'])
                if not agent.update_gamma:
                    agent.learn()
                else:
                    agent.learn_gamma()

        env.close()
        if FLAGS.mode == 'train':
            if episode != 0 and episode % 100 == 0:
                torch.save(agent.policy_net.state_dict(), 'weights/weights_{0}_{1}.pth'.format(time, episode))

        print('i_episode:', episode)
        print('eps_threshold = :', FLAGS.eps_end + (FLAGS.eps_start - FLAGS.eps_end) *
              math.exp(-1. * replay_buffer.steps_done / FLAGS.eps_decay))
        print('learn_steps:', agent.learn_steps)
        print('gamma:', agent.gamma)

        if FLAGS.mode == 'train' and episode != 0 and episode % 100 == 0:
            plot_average_queue(env.avg_queue, episode, time)


if __name__ == '__main__':
    app.run(main)
