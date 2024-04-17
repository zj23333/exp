import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from multiprocessing import Process, Pipe
import argparse
import gym
import datetime

# 获取当前时间
current_time = datetime.datetime.now()
# 格式化时间字符串
time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

# 建立Actor和Critic网络
class PolicyNet(torch.nn.Module):  # actor
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
        
class ValueNet(torch.nn.Module):  # critic
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma
        self.device = device
    """
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()"""
    def take_action(self, state_list):
        # 将状态列表转换为 PyTorch 张量，并移动到指定的设备上
        state_tensor = torch.tensor(state_list, dtype=torch.float).to(self.device)
        
        # 使用 actor 神经网络模型来获取动作概率分布
        probs = self.actor(state_tensor)
        
        # 对每个状态进行动作采样
        action_list = []
        for probs_i in probs:
            action_dist = torch.distributions.Categorical(probs_i)
            action = action_dist.sample()
            action_list.append(action.item())
        
        return action_list


    def update(self, transition_dict):
        #print(transition_dict['states'])
        #print(transition_dict['actions'])
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        #print(states)
        #print(actions)

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)  # 时序差分误差
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数

def test_env(env,agent): #,vis=False):
    state = env.reset()
    # if vis: env.render()
    done = [False] * env.num_envs
    total_reward = np.zeros(env.num_envs)
    while not done[0]:
        action = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        # if vis: env.render()
        total_reward += np.array(reward)
    return total_reward

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def train(cfg,envs):
    print('Start training!')
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')
    # env = gym.make(cfg.env_name)   ## 初始化一个测试用 env
    # env.reset(seed=10)
    env_eval_parameters = EnvironmentParameters(trace_start_index=120,
                                                num_traces=30,
                                                server_frequency=128.0,  # GHz
                                                num_base_station=number_of_base_state,
                                                optical_fiber_trans_rate=500.0,
                                                backhaul_coefficient=0.02,
                                                migration_coefficient_low=1.0,
                                                migration_coefficient_high=3.0,
                                                server_poisson_rate=possion_rate_vector,
                                                client_poisson_rate=2,
                                                server_task_data_lower_bound=(0.05 * 1000.0 * 1000.0 * 8),
                                                server_task_data_higher_bound=(5 * 1000.0 * 1000.0 * 8),
                                                client_task_data_lower_bound=(0.05 * 1000.0 * 1000.0 * 8),
                                                client_task_data_higher_bound=(5 * 1000.0 * 1000.0 * 8),
                                                migration_size_low=0.5,
                                                migration_size_high=100.0,
                                                ratio_lower_bound=200.0,
                                                ratio_higher_bound=10000.0,
                                                map_width=8000.0, map_height=8000.0,
                                                num_horizon_servers=x_base_state, num_vertical_servers=y_base_state,
                                                traces_file_path='./environment/san_traces_coordinate.txt',
                                                transmission_rates=[60.0, 48.0, 36.0, 24.0, 12.0],  # Mbps
                                                trace_length=100,
                                                trace_interval=3,
                                                is_full_observation=False,
                                                is_full_action=True)
    env = BatchMigrationEnv(env_eval_parameters)  # envs是训练用的，env是测试用的
    ### 
    n_states  = envs.observation_space  # 在我的env里，这个observation_space和action_space都是普通的int
    n_actions = envs.action_space
    # def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
    agent = ActorCritic(n_states, cfg.hidden_dim, n_actions, 1e-3, 1e-2, 0.8, cfg.device)
    step_idx    = 0
    test_rewards = []
    test_ma_rewards = []
    state = envs.reset()    # 这里！
    while step_idx < cfg.max_steps:
        print('training step: ',step_idx)
        states= []
        actions= []
        next_states= []
        rewards   = []
        dones     = []
        # rollout trajectory
        for _ in range(cfg.n_steps):
            action = agent.take_action(state)
            next_state, reward, done, _ = envs.step(action) ##
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            state = next_state
            step_idx += 1
            if step_idx % 10 == 0:  # 每200个step，测试一下
                test_reward = np.mean(test_env(env,agent)) # np.mean([test_env(env,model) for _ in range(10)])
                print(f"step_idx:{step_idx}, test_reward:{test_reward}")
                test_rewards.append(test_reward)
                if test_ma_rewards:
                    test_ma_rewards.append(0.9*test_ma_rewards[-1]+0.1*test_reward)
                else:
                    test_ma_rewards.append(test_reward) 
                # 勿动，这行本来就注释掉了 plot(step_idx, test_rewards)
        transition_dict = {'states': np.vstack(states), 'actions': np.concatenate(actions), 'next_states': np.vstack(next_states), 'rewards': np.concatenate(rewards), 'dones': np.concatenate(dones)}
        agent.update(transition_dict)
    print('Finish training！')
    torch.save(agent, time_str+"_agent.pt")
    return test_rewards, test_ma_rewards
    

import matplotlib.pyplot as plt
import seaborn as sns 
def plot_rewards(rewards, ma_rewards, cfg, time_str, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(
        cfg.device, cfg.algo_name, cfg.env_name))
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    plt.savefig(time_str + '.png')
    plt.show()
    
def save_rewards(rewards, ma_rewards, cfg, time_str, tag='train'):
    arr_re = np.array(rewards)
    np.savetxt(time_str + '_rewards.txt', arr_re)
    arr_ma = np.array(rewards)
    np.savetxt(time_str + '_ma_rewards.txt', arr_ma)
    
import easydict
# from common.multiprocessing_env import SubprocVecEnv
import batch_migration_env
from batch_migration_env import EnvironmentParameters
from batch_migration_env import BatchMigrationEnv

if __name__ == '__main__':
    cfg = easydict.EasyDict({
            "algo_name": 'A2C',
            "env_name": 'BatchMigrationEnv-v0',
            "n_envs": 100,
            "max_steps": 2000,
            "n_steps":5,
            "gamma":0.99,
            "lr": 1e-4,  # Tensor 里有nan，所以降低学习率的数量级，如果还报错，就再降  ValueError: Expected parameter probs (Tensor of shape (1, 30, 3)) of distribution Categorical(probs: torch.Size([1, 30, 3])) to satisfy the constraint Simplex(), but found invalid values:
            "hidden_dim": 256,
            "device":torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
    })
    
    number_of_base_state = 64
    x_base_state = 8
    y_base_state = 8

    possion_rate_vector = [11,  8, 20,  9, 18, 18,  9, 17, 12, 17,  9, 17, 14, 10,  5,  7, 12,
            8, 20, 10, 14, 12, 20, 14,  8,  6, 15,  7, 18,  9,  8, 18, 17,  7,
           11, 11, 13, 14,  8, 18, 13, 17,  6, 18, 17, 18, 18,  7,  9,  6, 12,
           10,  9,  8, 20, 14, 11, 15, 14,  6,  6, 15, 16, 20]

    env_default_parameters = EnvironmentParameters(trace_start_index=0,
                                                num_traces=120,
                                                server_frequency=128.0,  # GHz
                                                num_base_station=number_of_base_state,
                                                optical_fiber_trans_rate=500.0,
                                                backhaul_coefficient=0.02,
                                                migration_coefficient_low=1.0,
                                                migration_coefficient_high =3.0,
                                                server_poisson_rate=possion_rate_vector, client_poisson_rate=2,
                                                server_task_data_lower_bound=(0.05 * 1000.0 * 1000.0 * 8),
                                                server_task_data_higher_bound=(5 * 1000.0 * 1000.0 * 8),
                                                client_task_data_lower_bound=(0.05 * 1000.0 * 1000.0 * 8),
                                                client_task_data_higher_bound=(5 * 1000.0 * 1000.0 * 8),
                                                migration_size_low=0.5,
                                                migration_size_high=100.0,
                                                ratio_lower_bound=200.0,
                                                ratio_higher_bound=10000.0,
                                                map_width=8000.0, map_height=8000.0,
                                                num_horizon_servers=8, num_vertical_servers=8,
                                                traces_file_path='./environment/san_traces_coordinate.txt',
                                                transmission_rates=[60.0, 48.0, 36.0, 24.0, 12.0],  # Mbps
                                                trace_length=100,
                                                trace_interval=3,
                                                is_full_observation=False,
                                                is_full_action=True)
    # envs = [make_envs(cfg.env_name) for i in range(cfg.n_envs)]
    envs = BatchMigrationEnv(env_default_parameters)# SubprocVecEnv(envs) 
    rewards,ma_rewards = train(cfg,envs)
    
    save_rewards(rewards, ma_rewards, cfg, time_str, tag="train") # 存文件
    plot_rewards(rewards, ma_rewards, cfg, time_str, tag="train") # 画出结果

