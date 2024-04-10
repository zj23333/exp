from batch_migration_env import EnvironmentParameters
from batch_migration_env import BatchMigrationEnv

import numpy as np
import os


number_of_base_state = 64
x_base_state = 8
y_base_state = 8

possion_rate_vector = [11,  8, 20,  9, 18, 18,  9, 17, 12, 17,  9, 17, 14, 10,  5,  7, 12,
        8, 20, 10, 14, 12, 20, 14,  8,  6, 15,  7, 18,  9,  8, 18, 17,  7,
       11, 11, 13, 14,  8, 18, 13, 17,  6, 18, 17, 18, 18,  7,  9,  6, 12,
       10,  9,  8, 20, 14, 11, 15, 14,  6,  6, 15, 16, 20]

env_default_parameters = EnvironmentParameters(trace_start_index=50,
                                                num_traces=10,
                                                server_frequency=128.0,  # GHz
                                                num_base_station=64,
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


env = BatchMigrationEnv(env_default_parameters)
eval_env = BatchMigrationEnv(env_eval_parameters)

act0=[0]*10
act1=[1]*10
act2=[2]*10

state = env.reset()
state, reward, done, _ = env.step(act1)

state = eval_env.reset()
