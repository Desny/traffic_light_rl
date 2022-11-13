import random
import torch
import torch.nn as nn
import networks
from collections import namedtuple
import copy
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DqnAgent:
    def __init__(
        self,
        mode: str,
        replay,
        target_update: int,
        gamma: float,
        use_sgd: bool,
        eps_start: float,
        eps_end: float,
        eps_decay: int,
        input_dim: int,
        output_dim: int,
        batch_size: int,
        network_file: str = ''
    ):
        self.mode = mode
        self.replay = replay
        self.target_update = target_update
        self.gamma = gamma
        self.use_sgd = use_sgd
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.n_actions = output_dim
        self.batch_size = batch_size

        self.network_file = network_file
        self.policy_net = networks.DqnNetwork(input_dim, output_dim).to(device)
        self.target_net = networks.DqnNetwork(input_dim, output_dim).to(device)
        self.policy_net_copy = networks.DqnNetwork(input_dim, output_dim).to(device)
        if network_file:
            self.policy_net.load_state_dict(torch.load(network_file, map_location=torch.device(device)))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.learn_steps = 0
        self.z = None
        self.fixed_gamma = copy.deepcopy(gamma)
        self.update_gamma = False
        self.q_value_batch_avg = 0

    def select_action(self, state, steps_done, invalid_action):
        original_state = state
        state = torch.from_numpy(state)
        if self.mode == 'train':
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * steps_done / self.eps_decay)
            if sample > eps_threshold:
                with torch.no_grad():
                    _, sorted_indices = torch.sort(self.policy_net(state), descending=True)
                    if invalid_action:
                        return sorted_indices[1]
                    else:
                        return sorted_indices[0]
            else:
                decrease_state = [(original_state[0] + original_state[4]) / 2,
                                  (original_state[1] + original_state[5]) / 2,
                                  (original_state[2] + original_state[6]) / 2,
                                  (original_state[3] + original_state[7]) / 2]
                congest_phase = [i for i, s in enumerate(decrease_state) if abs(s-1) < 1e-2]
                if len(congest_phase) > 0 and invalid_action is False:
                    return random.choice(congest_phase)
                else:
                    return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                _, sorted_indices = torch.sort(self.policy_net(state), descending=True)
                if invalid_action:
                    return sorted_indices[1]
                else:
                    return sorted_indices[0]

    def learn(self):
        if self.mode == 'train':
            if self.replay.steps_done <= 10000:
                return
            loss_fn = nn.MSELoss()
            if self.use_sgd:
                optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=0.00025)
            else:
                optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=0.00025)

            transitions = self.replay.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action).view(self.batch_size, 1)
            next_state_batch = torch.cat(batch.next_state)
            reward_batch = torch.cat(batch.reward).view(self.batch_size, 1)
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)
            with torch.no_grad():
                argmax_action = self.policy_net(next_state_batch).max(1)[1].view(self.batch_size, 1)
                q_max = self.target_net(next_state_batch).gather(1, argmax_action)
                expected_state_action_values = reward_batch + self.gamma * q_max
                # for plot
                self.q_value_batch_avg = torch.mean(state_action_values).item()

            loss = loss_fn(state_action_values, expected_state_action_values)
            optimizer.zero_grad()
            loss.backward()

            self.cal_z(state_batch, action_batch, q_max)

            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
            self.learn_steps += 1
            self.update_gamma = True

    def cal_z(self, state_batch, action_batch, q_max):
        self.policy_net_copy.load_state_dict(self.policy_net.state_dict())
        z_optimizer = torch.optim.SGD(self.policy_net_copy.parameters(), lr=0.0001)
        state_action_copy_values = self.policy_net_copy(state_batch).gather(1, action_batch)
        z_optimizer.zero_grad()
        f_gamma_grad = torch.mean(0.00025 * q_max * state_action_copy_values)
        f_gamma_grad.backward()
        self.z = {'l1.weight': self.policy_net_copy.l1.weight.grad,
                  'l1.bias': self.policy_net_copy.l1.bias.grad,
                  'l2.weight': self.policy_net_copy.l2.weight.grad,
                  'l2.bias': self.policy_net_copy.l2.bias.grad}

    def learn_gamma(self):
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=0.00025)

        transitions = self.replay.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).view(self.batch_size, 1)
        next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward).view(self.batch_size, 1)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            argmax_action = self.policy_net(next_state_batch).max(1)[1].view(self.batch_size, 1)
            q_max = self.target_net(next_state_batch).gather(1, argmax_action)
            expected_state_action_values = reward_batch + self.fixed_gamma * q_max

        loss = loss_fn(state_action_values, expected_state_action_values)
        optimizer.zero_grad()
        loss.backward()

        l1_weight = self.policy_net.l1.weight.grad * self.z['l1.weight']
        l1_bias = self.policy_net.l1.bias.grad * self.z['l1.bias']
        l2_weight = self.policy_net.l2.weight.grad * self.z['l2.weight']
        l2_bias = self.policy_net.l2.bias.grad * self.z['l2.bias']

        gamma_grad = -0.99 * torch.mean(torch.cat((l1_weight.view(-1), l1_bias.view(-1), l2_weight.view(-1), l2_bias.view(-1))))
        self.gamma += gamma_grad
        self.update_gamma = False
