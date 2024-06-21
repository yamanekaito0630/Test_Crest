# ライブラリのインポート
import os
import subprocess
import time
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse
import sys
import pickle
import utils as u
import gc

from pyvirtualdisplay import Display
from permutation_invariant.solutions_mpi_evojax import PIAttentionAgent
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--os', help='Mac or Linux', type=str, default="Linux")
    parser.add_argument('--n-at', help='Num of attempt', type=int, default=4)
    parser.add_argument('--ns-trial', required=True, nargs="*", help='Num list of trial', type=int)
    parser.add_argument('--ns-robo', required=True, nargs="*", help='Num list of robo', type=int)
    parser.add_argument('--es-robo', required=True, nargs="*", help='Num list of eval robo', type=int)
    parser.add_argument('--versions', required=True, nargs="*", help='Num list of versions', type=int)
    parser.add_argument('--eval-version', required=True, help='eval version', type=int)
    parser.add_argument('--load-model', help='Select model file', default='Iter_500.npz')
    parser.add_argument('--n-fitness', help='Num of Fitness', type=int, default=4)
    parser.add_argument('--steps', help='Steps for eval', type=int, default=1000)
    parser.add_argument('--reps', help='repeats of recode', type=int, default=1)
    parser.add_argument('--env-name', help='name of environment', type=str, default="default")
    parser.add_argument('--exclude-other-envs', help='True or False', type=int, default=0)
    parser.add_argument('--save-movies', help='True or False', type=int, default=1)
    parser.add_argument('--headless', help='True or False', type=int, default=1)
    config, _ = parser.parse_known_args()
    return config


def main(config, log_dir, n_at, e_robo, e_version):
    save_path = log_dir + 'eval_{num_robo}robo/{env_name}/'.format(num_robo=e_robo, env_name=config.env_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device('cpu')
    if e_robo == 1:
        if config.env_name == "default":
            # file_name = 'render_apps/UnderWaterDrones_IM_Round_OneRobot_At{at}'.format(at=n_at)
            file_name = 'Test_Crest_App/{}/render_app/UnderWaterDrones_IM_Round_OneRobot_At{}_V{}'.format(config.os, n_at, e_version)
        else:
            file_name = 'Test_Crest_App/{}/render_app/UnderWaterDrones_IM_Round_OneRobot_At{}_V{}_{}'.format(config.os, n_at, e_version, config.env_name)
    else:
        if config.env_name == "default":
            # file_name = 'render_apps/UnderWaterDrones_IM_Round_{num_robo}Robots_At{at}'.format(num_robo=e_robo, at=n_at)
            file_name = 'Test_Crest_App/{}/render_app/UnderWaterDrones_IM_Round_{}Robots_At{}_V{}'.format(config.os, e_robo, n_at, e_version)
        else:
            file_name = 'Test_Crest_App/{}/render_app/UnderWaterDrones_IM_Round_{}Robots_At{}_V{}_{}'.format(config.os, e_robo, n_at, e_version, config.env_name)

    agent = PIAttentionAgent(
        device=device,
        file_name=file_name,
        act_dim=5,
        msg_dim=16,
        pos_em_dim=8,
        patch_size=6,
        stack_k=3,
        aa_image_size=64,
        aa_query_dim=4,
        aa_hidden_dim=16,
        aa_top_k=50
    )
    agent.load(log_dir + config.load_model)

    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=1, width=1500, height=600, capture_frame_rate=50)
    env = UnityEnvironment(file_name=file_name, side_channels=[channel])
    env.reset()

    behavior_names = list(env.behavior_specs.keys())
    decision_steps, terminal_steps = env.get_steps(behavior_names[0])

    counter = 0
    n_recode = 0

    xs, ys, zs = [], [], []
    rss, gss, bss = [], [], []

    obs_actions, obs_leds = [], []
    z_actions = []
    
    # episode_nodes, episode_links = [], []
    # timestep_nodes, timestep_links = {}, []

    while True:
        start = time.time()
        for i in decision_steps.agent_id:
            input_image = agent.img_scale(decision_steps.obs[0][i] * 255)
            obs = np.transpose(input_image, (2, 1, 0))
            action = agent.get_action(obs)
            action_tuple = ActionTuple(continuous=np.expand_dims(action, axis=0))
            env.set_action_for_agent(behavior_names[0], i, action_tuple)
            if counter > 0:
                rs, gs, bs = agent.show_gui(obs=input_image, counter=counter, path="log/")
                top_color = u.get_top_color(rs, gs, bs)
                obs_actions.append([action[1], action[2], top_color])
                obs_leds.append([action[3], action[4], top_color])
                z_actions.append(np.abs(action[2]))

                rss += rs
                gss += gs
                bss += bs

                del rs, gs, bs
                gc.collect()
            

            # print('action:', action)

            xs.append(decision_steps.obs[1][i][config.n_fitness + 1:][0])
            ys.append(decision_steps.obs[1][i][config.n_fitness + 1:][1])
            zs.append(decision_steps.obs[1][i][config.n_fitness + 1:][2])
            
            # node_index = decision_steps.obs[1][i][config.n_fitness + 4 : config.n_fitness + 6][0]
            # coordinate = decision_steps.obs[1][i][config.n_fitness + 1 : config.n_fitness + 4]
            # timestep_nodes[node_index] = coordinate
            # timestep_links.append(tuple(decision_steps.obs[1][i][config.n_fitness + 4 : config.n_fitness + 6]))
            # timestep_links.append(tuple(decision_steps.obs[1][i][config.n_fitness + 6 : config.n_fitness + 8]))
        # episode_nodes.append(timestep_nodes)
        # episode_links.append(timestep_links)
        # timestep_nodes, timestep_links = {}, []
        env.step()
        # img = agent.img_scale(decision_steps.obs[0][0] * 255)

        # 入力値（画像）
        # simple_img = cv2.resize(img, (400, 400))[:, :, ::-1]
        # cv2.imwrite('simple_obs/img_' + str(counter) + '.png', simple_img)

        # timestep_color_counts = np.zeros(8)
        # rs, gs, bs= agent.show_gui(obs=img, counter=counter, path="log/")
        
        # rss += rs
        # gss += gs
        # bss += bs

        counter += 1
        decision_steps, terminal_steps = env.get_steps(behavior_names[0])
        print('steps:', counter)
        end = time.time()
        print(end - start, "[s]")
        if counter % config.steps == 0:
            robots_coordinates = np.array([xs, ys, zs])
            pickle.dump(robots_coordinates, open(save_path + 'robots_coordinates_{}.pkl'.format(n_recode), 'wb'))
            # pickle.dump(episode_nodes, open(save_path + 'episode_nodes_{}.pkl'.format(n_recode), 'wb'))
            # pickle.dump(episode_links, open(save_path + 'episode_links_{}.pkl'.format(n_recode), 'wb'))
            pickle.dump(obs_actions, open(save_path + 'obs_actions_{}.pkl'.format(n_recode), 'wb'))
            pickle.dump(obs_leds, open(save_path + 'obs_leds_{}.pkl'.format(n_recode), 'wb'))
            pickle.dump(z_actions, open(save_path + 'z_actions_{}.pkl'.format(n_recode), 'wb'))
            u.get_scatter_of_actions(path=save_path, obs_actions=obs_actions)
            u.get_scatter_of_leds(path=save_path, obs_leds=obs_leds)
            u.coordinates_3d(path=save_path, n_recode=n_recode, coordinates=robots_coordinates, n_robo=e_robo)
            u.z_t_creator(path=save_path, n_recode=n_recode, coordinates=robots_coordinates, n_robo=e_robo, timesteps=config.steps)

            apm = (rss, gss, bss)
            pickle.dump(apm, open(save_path + 'ap_material_{}.pkl'.format(n_recode), 'wb'))
            u.at_scatter_creator(path=save_path, n_recode=n_recode, apm=apm)

            n_recode += 1
            env.reset()
            
            del robots_coordinates, apm, xs, ys, zs, rss, gss, bss
            gc.collect()

            xs, ys, zs = [], [], []
            rss, gss, bss = [], [], []
            obs_actions, obs_leds = [], []
            z_actions = []
            # episode_nodes, episode_links = [], []

            if n_recode >= config.reps:
                env.close()
                return


if __name__ == '__main__':
    args = parse_args()
    print("test")
    d = Display()
    if args.headless:
        d.start()

    for n_robo in args.ns_robo:
        print("n_robo=", n_robo)
        for n_version in args.versions:
            for n_trial in args.ns_trial:
                log_dir = "log/at{at}/{n_robo}robo/v{version}/trial_{n_trial}/".format(at=args.n_at, n_robo=n_robo, version=n_version, n_trial=n_trial)

                while not os.path.isfile(log_dir + args.load_model):
                    print(log_dir + args.load_model + " is not exist.")
                    time.sleep(5)

                for e_robo in args.es_robo:
                    print("e_robo=", e_robo)
                    if args.exclude_other_envs:
                        if e_robo != n_robo:
                            continue

                    main(args, log_dir, args.n_at, e_robo, args.eval_version)

                    if args.save_movies:
                        res_1 = subprocess.run("sh recode_pi_behave.sh {} {} {} {} {} {}".format(args.n_at, n_robo, n_version, e_robo, n_trial, args.env_name), shell=True)
                        print("res_1:", res_1.returncode)
                        res_2 = subprocess.run("sh recode_pi_attention.sh {} {} {} {} {} {}".format(args.n_at, n_robo, n_version, e_robo, n_trial, args.env_name), shell=True)
                        print("res_2:", res_2.returncode)

    if args.headless:
        d.stop()
