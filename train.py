from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure
import gym
from gym.wrappers import FilterObservation, FlattenObservation
import panda_gym
import torch as th
import os
import numpy as np


def check_and_rename(original_path):
    i = 1
    path = original_path + '_' + str(i)
    while os.path.exists(path):
        path = original_path + '_' + str(i)
        i += 1

    return path


def learnPPO(policy, env, models_dir, logs_dir):
    # Train an agent

    # Custom actor (pi) and value function (vf) networks
    # Note: an extra linear layer will be added on top of the pi and the vf nets, respectively

    # policy_kwargs = dict(activation_fn=th.nn.ReLU,
    #                      net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]))

    TIMESTEPS = 100_000  # cada cuánto se guarda el modelo
    TOTAL_TIMESTEPS = 5_000_000  # múltiplo de TIMESTEPS
    CYCLES = TOTAL_TIMESTEPS // TIMESTEPS

    model = PPO(policy, env, verbose=1, tensorboard_log=logs_dir,
                learning_rate=0.000104019,
                n_steps=512,
                batch_size=32,
                n_epochs=5,
                gamma=0.9,
                gae_lambda=1,
                clip_range=0.3,
                ent_coef=0.0000000752585,
                vf_coef=1,
                max_grad_norm=0.9)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    logs_dir_PPO = check_and_rename(logs_dir + '/PPO')

    new_logger = configure(f'{logs_dir_PPO}',
                           ['stdout', 'csv', 'tensorboard'])  # imprime log en consola y guarda en csv y tensorboard
    model.set_logger(new_logger)

    print(model.policy)

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    models_dir_PPO = check_and_rename(models_dir + '/PPO')

    for i in range(1, CYCLES + 1):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir_PPO}/{TIMESTEPS * i}")

    # return model


def learnSAC(policy, env, models_dir, logs_dir):
    # Train an agent

    # Custom actor (pi) and critic (qf) networks
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]))

    TIMESTEPS = 20_000  # cada cuánto se guarda el modelo
    TOTAL_TIMESTEPS = 200_000  # múltiplo de TIMESTEPS
    CYCLES = TOTAL_TIMESTEPS // TIMESTEPS

    model = SAC(policy, env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=logs_dir,
                learning_rate=0.001,
                tau=0.95,  # the soft update coefficient (“Polyak update”, between 0 and 1)
                learning_starts=50,
                batch_size=256,
                buffer_size=1_000_000,
                ent_coef=0.2
                )

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    logs_dir_SAC = check_and_rename(logs_dir + '/SAC')

    new_logger = configure(f'{logs_dir_SAC}',
                           ['stdout', 'csv', 'tensorboard'])  # imprime log en consola y guarda en csv y tensorboard
    model.set_logger(new_logger)

    print(model.policy)

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    models_dir_SAC = check_and_rename(models_dir + '/SAC')

    for i in range(1, CYCLES + 1):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="SAC")
        model.save(f"{models_dir_SAC}/{TIMESTEPS * i}")

    # return model


if __name__ == '__main__':
    # INITIAL DATA
    policy = 'MlpPolicy'  # Define el modelo (red neuronal)
    entorno = 'PandaReach-v2'  # Define el robot, la tarea y el tipo de reward
    algoritmo = 'PPO'  # 'PPO', 'SAC', 'both'
    PRECISION = 0.05  # precisión al goal, por defecto 0.05 m
    GOAL_RANGE = 0.3  # define el volumen en el que aparece el goal, por defecto 0.3 --> 0.3x0.3x0.3 m3
    SEEDS = [17, 27]

    # MAIN
    dir = os.getcwd()
    dir = dir + '/train'  # Carpeta en la que se guardarán logs y modelos

    for seed in SEEDS:
        env = gym.make(entorno)

        env.unwrapped.task.distance_threshold = PRECISION
        env.unwrapped.task.goal_range = GOAL_RANGE
        env.unwrapped.task.goal_range_low = np.array([-GOAL_RANGE / 2, -GOAL_RANGE / 2, 0])  # el origen de coordenadas es el centro de la mesa
        env.unwrapped.task.goal_range_high = np.array([GOAL_RANGE / 2, GOAL_RANGE / 2, GOAL_RANGE])

        env = FilterObservation(
            env, filter_keys=['desired_goal', 'observation'])
        env = FlattenObservation(env)

        env.seed(seed)

        models_dir = f"{dir}/_models"  # Subcarpeta para guardar los modelos
        logs_dir = f"{dir}/_logs"  # Subcarpeta para guardar los logs

        if algoritmo == 'both':
            learnPPO(policy, env, models_dir, logs_dir)
            learnSAC(policy, env, models_dir, logs_dir)
        elif algoritmo == 'PPO':
            learnPPO(policy, env, models_dir, logs_dir)
        elif algoritmo == 'SAC':
            learnSAC(policy, env, models_dir, logs_dir)
        else:
            print('Especifica un algoritmo correcto')
