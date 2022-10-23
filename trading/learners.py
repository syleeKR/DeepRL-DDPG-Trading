
import torch
import torch.nn as nn
from torch.nn import MSELoss
import torch.nn.functional as F
import copy
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import Adam
from buffer import ReplayBuffer
from utils import save_snapshot, recover_snapshot, load_model
from environment import *
#from visualizer import *
import matplotlib.pyplot as plt



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('current device =', device)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 1)

        self.fc6 = nn.Linear(obs_dim + act_dim, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 128)
        self.fc9 = nn.Linear(128, 128)
        self.fc10 = nn.Linear(128, 1)


    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)

        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = F.relu(self.fc3(q1))
        q1 = F.relu(self.fc4(q1))
        q1 = self.fc5(q1)

        q2 = F.relu(self.fc1(x))
        q2 = F.relu(self.fc2(q2))
        q2 = F.relu(self.fc3(q2))
        q2 = F.relu(self.fc4(q2))
        q2 = self.fc5(q2)

        return q1, q2


# actor network definition
# multi-layer perceptron (with 2 hidden layers)
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, ctrl_range):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, act_dim)

        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.1)
        self.drop3 = nn.Dropout(p=0.1)
        self.drop4 = nn.Dropout(p=0.1)

        self.bn1 = nn.BatchNorm1d(obs_dim)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(128)

        self.ctrl_range = ctrl_range

    def forward(self, obs,type):
        #x = self.bn1(obs)
        x = F.relu(self.fc1(obs))
        #x = self.bn2(x)
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        #x = self.bn3(x)
        x = self.drop2(x)
        x = F.relu(self.fc3(x))
        #x = self.bn4(x)
        x = self.drop3(x)
        x = F.relu(self.fc4(x))
        #x = self.bn5(x)
        x = self.drop4(x)

        if(type==1):
            return self.ctrl_range/2 * torch.tanh(self.fc5(x))
        else:
            return self.ctrl_range * torch.tanh(self.fc5(x))


class DDPGAgent:
    def __init__(self, obs_dim, act_dim, ctrl_range ):
        # networks
        self.actor = Actor(obs_dim, act_dim, ctrl_range).to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)

    def act(self, obs,type):
        # numpy ndarray to torch tensor
        # we first add an extra dimension
        obs = np.array(obs)
        obs = obs[np.newaxis, ...]
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).to(device)
            # TODO : get agent action from policy network (self.actor)
            act_tensor = self.actor(obs_tensor,type)

        # torch tensor to numpy ndarray
        # remove extra dimension
        action = act_tensor.cpu().detach().numpy()
        action = np.squeeze(action, axis=0)

        return action

def update(agent, replay_buf, gamma, actor_optim, critic_optim, target_actor, target_critic, tau, batch_size):
        # agent : agent with networks to be trained
        # replay_buf : replay buf from which we sample a batch
        # actor_optim / critic_optim : torch optimizers
        # tau : parameter for soft target update

    batch = replay_buf.sample_batch(batch_size=batch_size)

        # target construction does not need backward ftns
    with torch.no_grad():
            # unroll batch
        obs = torch.Tensor(batch.obs).to(device)
        act = torch.Tensor(batch.act).to(device)
        next_obs = torch.Tensor(batch.next_obs).to(device)
        rew = torch.Tensor(batch.rew).to(device)
        done = torch.Tensor(batch.done).to(device)

            ################
            # train critic #
            ################
        mask = 1. - done
        tq1, tq2 = target_critic(next_obs, target_actor(next_obs,2))
        target = rew + gamma * mask * torch.min(tq1, tq2)

    q1, q2 = agent.critic(obs, act)
    critic_loss = torch.mean((q1 - target) ** 2 + (q2 - target) ** 2)

    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

        ###############
        # train actor #
        ###############
        # freeze critic during actor training (why?)
    for p in agent.critic.parameters():
        p.requires_grad_(False)

        # TODO : actor loss construction! (Warning: sign of the loss?)
    q1, q2 = agent.critic(obs, agent.actor(obs,2))
    actor_loss = -torch.mean(torch.min(q1, q2))

    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

        # unfreeze critic after actor training
    for p in agent.critic.parameters():
        p.requires_grad_(True)

        # soft target update (both actor & critic network)
    for p, targ_p in zip(agent.actor.parameters(), target_actor.parameters()):
        targ_p.data.copy_((1. - tau) * targ_p + tau * p)
    for p, targ_p in zip(agent.critic.parameters(), target_critic.parameters()):
        targ_p.data.copy_((1. - tau) * targ_p + tau * p)

    return

def evaluate(agent, env, num_episodes):

    sum_scores = 0.

    for i in range(num_episodes):
        obs = env.reset()
        done = False
        score = 0.

        while not done:
            action = agent.act(obs,2)
            obs, rew, done, _ = env.step(action)
            score += rew



        sum_scores += score
    avg_score = sum_scores / num_episodes

    return avg_score

def train(agent, env, gamma,
              actor_lr, critic_lr, tau, noise_std,
              ep_len, num_updates, batch_size,
              init_buffer, buffer_size,
              start_train, train_interval,
              eval_interval, snapshot_interval, warmup,path=None):

    target_actor = copy.deepcopy(agent.actor)
    target_critic = copy.deepcopy(agent.critic)

        # freeze target networks
    for p in target_actor.parameters():
        p.requires_grad_(False)
    for p in target_critic.parameters():
        p.requires_grad_(False)

    actor_optim = Adam(agent.actor.parameters(), lr=actor_lr)
    critic_optim = Adam(agent.critic.parameters(), lr=critic_lr)

    if path is not None:
        recover_snapshot(path, agent.actor, agent.critic,
                             target_actor, target_critic,
                             actor_optim, critic_optim,
                             device=device
                             )
            # load snapshot

    obs_dim = env.observation_space_dim
    act_dim = env.action_space_dim
    ctrl_range = env.action_space_high

    replay_buf = ReplayBuffer(obs_dim, act_dim, buffer_size)

    save_path = './snapshots/'
    os.makedirs(save_path, exist_ok=True)

    test_env = copy.deepcopy(env)

        # main loop
    obs = env.reset()
    done = False
    step_count = 0
    ep = 0
    store =[]
    for t in range(num_updates + 1):
        if t < init_buffer:
                # perform random action until we collect sufficiently many samples
                # this is for exploration purpose

            action = [env.random_action()[0]/2]


        else:
                # executes noisy action - similar to epsilon-greedy
                # a_t = \pi(s_t) + N(0, \sigma^2)
            if (t<warmup):
                action = agent.act(obs,1) + noise_std * np.random.randn(act_dim)
                action = np.clip(action, -ctrl_range, ctrl_range)
            else:
                action = agent.act(obs,2) + noise_std * np.random.randn(act_dim)
                action = np.clip(action , -ctrl_range , ctrl_range )


        next_obs, rew, done, _ = env.step(action)
        #print(next_obs, agent.act(next_obs))
        step_count += 1
        if step_count == ep_len:
                # if the next_state is not terminal but done is set to True by gym env wrapper
            done = False

        replay_buf.append(obs, action, next_obs, rew, done)
        obs = next_obs

        if done == True or step_count == ep_len:
                # reset environment if current environment reaches a terminal state
                # or step count reaches predefined length



            obs = env.reset()
            done = False
            step_count = 0
            ep += 1


        if t % eval_interval == 0:
            avg_score = evaluate(agent, test_env, num_episodes=1)
            store.append(avg_score)
            print('[iter {} / ep {}] average score = {:.4f} (over 1 episodes)'.format(t, ep, avg_score))

        if t > start_train and t % train_interval == 0:
                # start training after fixed number of steps
                # this may mitigate overfitting of networks to the
                # small number of samples collected during the initial stage of training
            for _ in range(train_interval):
                update(agent,
                           replay_buf,
                           gamma,
                           actor_optim,
                           critic_optim,
                           target_actor,
                           target_critic,
                           tau,
                           batch_size
                           )
        if t % snapshot_interval == 0:
            snapshot_path = save_path + 'iter{}_'.format(t)
                # save weight & training progress
            save_snapshot(snapshot_path, agent.actor, agent.critic,
                              target_actor, target_critic,
                              actor_optim, critic_optim)

    plt.plot(store)
    plt.show()

