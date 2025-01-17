from environment.batch_migration_env import EnvironmentParameters
from environment.batch_migration_env import BatchMigrationEnv
from baselines.linear_baseline import LinearTimeBaseline
from baselines.rnn_critic_network_baseline import RNNCriticNetworkBaseline
from policies.rnn_policy_with_action_input import RNNPolicy
from policies.rnn_policy_with_action_input import RNNValueNet
from policies.rnn_policy_with_action_input import RNNPolicyWithValue
from policies.optimal_solution import optimal_solution_for_batch_system_infos

from policies.fc_categorical_policy import FCCategoricalPolicy
from policies.fc_categorical_policy import FCCategoricalPolicyWithValue
from policies.fc_categorical_policy import FCValueNetwork
from policies.rnn_critic_network import RNNValueNetwork
from policies.random_migrate_policy import RandomMigratePolicy
from policies.always_migrate_policy import AlwaysMigratePolicy

from sampler.migration_sampler import MigrationSamplerProcess
from sampler.migration_sampler import MigrationSampler
from sampler.migration_sampler import EvaluationSampler
from algorithms.dracm import DRACM
from dracm_trainer import Trainer

import itertools
import numpy as np
import tensorflow as tf

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from utils import logger


if __name__ == "__main__":
    number_of_base_state = 64
    x_base_state = 8
    y_base_state = 8

    # possion_rate_vector = np.random.randint(10, 31, size=number_of_base_state)
    # print("possion_rate_vector is: ", repr(possion_rate_vector))

    # 40.0, 36.0, 32.0, 28.0, 24.0,
    logger.configure(dir="./log/ppo-rome-with-optimal-100-bs-64-new", format_strs=['stdout', 'log', 'csv'])

    # bs number = 64
    possion_rate_vector = [7, 10, 8, 14, 15, 6, 20, 18, 11, 17, 20, 9, 8, 14, 9, 15, 8, 17, 9, 9, 10, 7, 17, 10,
                           13, 12, 5, 8, 10, 13, 19, 15, 10, 9, 10, 18, 12, 13, 5, 11, 7, 8, 8, 19, 15, 15, 6, 10,
                           5, 20, 17, 5, 5, 16, 5, 19, 19, 19, 9, 20, 17, 14, 17, 17]

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
                                                num_horizon_servers=x_base_state, num_vertical_servers=y_base_state,
                                                traces_file_path='./environment/rome_traces_coordinate.txt',
                                                transmission_rates=[60.0, 48.0, 36.0, 24.0, 12.0],  # Mbps
                                                trace_length=100,
                                                trace_interval=12,
                                                is_full_observation=False,
                                                is_full_action=True)

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
                                                traces_file_path='./environment/rome_traces_coordinate.txt',
                                                transmission_rates=[60.0, 48.0, 36.0, 24.0, 12.0],  # Mbps
                                                trace_length=100,
                                                trace_interval=12,
                                                is_full_observation=False,
                                                is_full_action=True)

    env = BatchMigrationEnv(env_default_parameters)
    eval_env = BatchMigrationEnv(env_eval_parameters)

    print("action dim of the environment: ", env._action_dim)

    rnn_policy = RNNPolicyWithValue(observation_dim=env._state_dim,
                                    action_dim=env._action_dim,
                                    rnn_parameter=256,
                                    embbeding_size=2)
    vf_baseline = RNNCriticNetworkBaseline(rnn_policy)

    sampler = MigrationSampler(env,
                               policy=rnn_policy,
                               batch_size=4800,
                               num_environment_per_core=2,
                               max_path_length=100,
                               parallel=True,
                               num_process=8,
                               is_norm_reward=True) # 2 * 4 * 30

    eval_sampler = EvaluationSampler(eval_env,
                                     policy=rnn_policy,
                                     batch_size=10,
                                     max_path_length=100)

    sampler_process = MigrationSamplerProcess(baseline=vf_baseline,
                                              discount=0.99,
                                              gae_lambda=0.95,
                                              normalize_adv=True,
                                              positive_adv=False)
    algo = DRACM(policy=rnn_policy,
                 value_function=rnn_policy,
                 policy_optimizer=tf.keras.optimizers.Adam(1e-3),
                 value_optimizer=tf.keras.optimizers.Adam(1e-3),
                 is_rnn=True,
                 is_shared_critic_net=True,
                 num_inner_grad_steps=4,
                 clip_value=0.2,
                 vf_coef=0.5,
                 max_grad_norm=1.0,
                 entropy_coef=0.03)

    trainer = Trainer(train_env=env,
                      eval_env=eval_env,
                      algo=algo,
                      sampler=sampler,
                      sample_processor=sampler_process,
                      update_batch_size=240,
                      policy=rnn_policy,
                      n_itr=120,
                      save_interval=5,
                      eval_sampler=eval_sampler,
                      test_interval=10,
                      save_path = 'checkpoints_ppo_64-bs-new-2\\model_checkpoint_epoch_')

    trainer.train(rnn_policy=True, is_test=False)
