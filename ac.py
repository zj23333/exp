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
class ActorCritic(nn.Module):
    ''' A2C网络模型，包含一个Actor和Critic
    '''
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        print("------------- x -------------")
        print(x)
        print("------------- value -------------")
        print(value)
        print("------------- probs -------------")
        print(probs)
        dist  = Categorical(probs)
        return dist, value

class A2C:
    ''' A2C算法
    '''
    def __init__(self,n_states,n_actions,cfg) -> None:
        self.gamma = cfg.gamma
        self.device = cfg.device
        self.model = ActorCritic(n_states, n_actions, cfg.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())

    def compute_returns(self,next_value, rewards, masks):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)
        return returns
        
def make_envs(env_name):
    def _thunk():
        env = gym.make(env_name)
        env.reset(seed=2)
        return env
    return _thunk
    
def test_env(env,model): #,vis=False):
    state = env.reset()
    # if vis: env.render()
    done = [False] * env.num_envs
    total_reward = np.zeros(env.num_envs)
    while not done[0]:
        state = torch.FloatTensor(state).unsqueeze(0).to(cfg.device)
        dist, _ = model(state)
        #print(dist.sample().cpu().numpy())
        #print(type(dist.sample().cpu().numpy()))
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
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
                                                is_full_observation=True,
                                                is_full_action=False)
    env = BatchMigrationEnv(env_eval_parameters)  # envs是训练用的，env是测试用的
    ### 
    n_states  = envs.observation_space  # 在我的env里，这个observation_space和action_space都是普通的int
    n_actions = envs.action_space
    model = ActorCritic(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
    optimizer = optim.Adam(model.parameters())
    step_idx    = 0
    test_rewards = []
    test_ma_rewards = []
    state = envs.reset()    # 这里！
    while step_idx < cfg.max_steps:
        print('training step: ',step_idx)
        log_probs = []
        values    = []
        rewards   = []
        masks     = []
        entropy = 0
        # rollout trajectory
        for _ in range(cfg.n_steps):
            state = torch.FloatTensor(state).to(cfg.device)
            dist, value = model(state)
            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy()) ##
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(cfg.device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(cfg.device))
            state = next_state
            step_idx += 1
            if step_idx % 200 == 0:  # 每200个step，测试一下
                test_reward = np.mean(test_env(env,model)) # np.mean([test_env(env,model) for _ in range(10)])
                print(f"step_idx:{step_idx}, test_reward:{test_reward}")
                test_rewards.append(test_reward)
                if test_ma_rewards:
                    test_ma_rewards.append(0.9*test_ma_rewards[-1]+0.1*test_reward)
                else:
                    test_ma_rewards.append(test_reward) 
                # 勿动，这行本来就注释掉了 plot(step_idx, test_rewards)
        next_state = torch.FloatTensor(next_state).to(cfg.device)
        _, next_value = model(next_state)
        returns = compute_returns(next_value, rewards, masks)
        log_probs = torch.cat(log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(values)
        advantage = returns - values
        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Finish training！')
    torch.save(model, time_str+"_model.pt")
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
            "max_steps": 20000,
            "n_steps":5,
            "gamma":0.99,
            "lr": 1e-8,  # Tensor 里有nan，所以降低学习率的数量级，如果还报错，就再降  ValueError: Expected parameter probs (Tensor of shape (1, 30, 3)) of distribution Categorical(probs: torch.Size([1, 30, 3])) to satisfy the constraint Simplex(), but found invalid values:
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
                                                num_traces=100,
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
                                                is_full_observation=True,
                                                is_full_action=False)
    # envs = [make_envs(cfg.env_name) for i in range(cfg.n_envs)]
    envs = BatchMigrationEnv(env_default_parameters)# SubprocVecEnv(envs) 
    rewards,ma_rewards = train(cfg,envs)
    
    save_rewards(rewards, ma_rewards, cfg, time_str, tag="train") # 存文件
    plot_rewards(rewards, ma_rewards, cfg, time_str, tag="train") # 画出结果

