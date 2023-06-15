from stable_baselines3 import PPO, SAC
import gymnasium
from gymnasium.wrappers import FilterObservation, FlattenObservation
import panda_gym
import numpy as np
import calendar
import time
import os
import csv
from matplotlib import animation
import matplotlib.pyplot as plt


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=5)


def runModel(model, vec_env, GIF):
    # Run the model

    episodes = 15

    if GIF:
        frames = []

    dir = os.getcwd()
    dir = dir + '/test'  # Carpeta en la que se guarda el log

    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)

    log_file = dir + '/log_{}.csv'.format(time_stamp)

    with open(log_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['episode', 'step', 'action1', 'action2', 'action3', 'obs1', 'obs2', 'obs3', 'obs4',
                         'obs5', 'obs6', 'obs7', 'obs8', 'obs9', 'reward', 'terminated', 'truncated'])

    for ep in range(1, episodes+1):
        obs, _ = vec_env.reset()
        terminated = False
        i = 0

        # log initial observation
        fields = [ep, i, "", "", "", obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7], obs[8], "",
                  terminated, ""]
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

        ret = 0

        if GIF:
            frames.append(vec_env.render())
            time.sleep(.2)
        else:
            vec_env.render()
            time.sleep(.2)

        while not terminated:
            i += 1
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = vec_env.step(action)

            if GIF:
                frames.append(vec_env.render())
                time.sleep(.2)
            else:
                vec_env.render()
                time.sleep(.2)

            print('Step: {}, rew: {}'.format(i, reward))
            print('Actions: {}\nObs: {}\nState: {}'.format(action, obs, _state))

            # log
            fields = [ep, i, action[0], action[1], action[2], obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6],
                      obs[7], obs[8], reward, terminated, truncated]
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(fields)

            ret += reward

            if i > 50:
                break
        print('Episode {} finished with return: {} \n'.format(ep, ret))

    vec_env.close()

    if GIF:
        save_frames_as_gif(frames, dir, '/animation_{}.gif'.format(time_stamp))


if __name__ == '__main__':
    # INITIAL DATA
    entorno = 'PandaReach-v3'  # Define el robot, la tarea y el tipo de reward
    algoritmo = 'PPO_1'  # Subcarpeta en la que está el modelo
    modelo = 'modelo1.zip'
    PRECISION = 0.05  # precisión al goal, por defecto 0.05 m
    GOAL_RANGE = 0.3  # define el volumen en el que aparece el goal, por defecto 0.3 --> 0.3x0.3x0.3 m3

    # RENDER DATA
    GIF = True  # si queremos guardar un .gif del render (True) o no (False)
    render = 'rgb_array'
    renderer = 'OpenGL'
    render_width = 720
    render_height = 480
    render_target_position = [0., 0., 0.]
    render_distance = 1.0
    render_yaw = 45
    render_pitch = -30
    render_roll = 0

    # MAIN
    env = gymnasium.make(
        entorno,
        render_mode=render,  # render_mode="human"
        renderer=renderer,
        render_width=render_width,  # 720
        render_height=render_height,  # 480
        render_target_position=render_target_position,  # [0., 0., 0.]
        render_distance=render_distance,  # 1.4
        render_yaw=render_yaw,  # 45
        render_pitch=render_pitch,  # -30
        render_roll=render_roll  # 0
    )

    env.unwrapped.task.distance_threshold = PRECISION
    env.unwrapped.task.goal_range = GOAL_RANGE
    env.unwrapped.task.goal_range_low = \
        np.array([-GOAL_RANGE / 2, -GOAL_RANGE / 2, 0])  # el origen de coordenadas es el centro de la mesa
    env.unwrapped.task.goal_range_high = \
        np.array([GOAL_RANGE / 2, GOAL_RANGE / 2, GOAL_RANGE])

    env.reset()

    env = FilterObservation(
        env, filter_keys=['desired_goal', 'observation'])  # filter_keys=['desired_goal', 'observation']
    env = FlattenObservation(env)

    dir = os.getcwd()
    dir = dir + '/train/_models'  # Carpeta en la que están los modelos

    model_path = f"{dir}/{algoritmo}/{modelo}"

    if 'PPO' in algoritmo:
        model = PPO.load(model_path, env=env)
        runModel(model, env, GIF)
    elif 'SAC' in algoritmo:
        model = SAC.load(model_path, env=env)
        runModel(model, env, GIF)
    else:
        print('Se debe elegir un modelo PPO o SAC')
