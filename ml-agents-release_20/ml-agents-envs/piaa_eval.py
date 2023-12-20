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

from pyvirtualdisplay import Display
from permutation_invariant.solutions_mpi_evojax import PIAttentionAgent
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-at', help='Num of attempt', type=int, default=5)
    parser.add_argument('--ns-robo', required=True, nargs="*", help='Num list of robo', type=int)
    parser.add_argument('--versions', required=True, nargs="*", help='Num list of versions', type=int)
    parser.add_argument('--load-model', help='Select model file', default='Iter_1000.npz')
    parser.add_argument('--n-fitness', help='Num of Fitness', type=int, default=4)
    parser.add_argument('--steps', help='Steps for eval', type=int, default=2000)
    parser.add_argument('--reps', help='repeats of recode', type=int, default=3)
    parser.add_argument('--headless', help='True or False', type=bool, default=True)
    config, _ = parser.parse_known_args()
    return config


def main(config, log_dir, n_at, n_robo, n_version):
    device = torch.device('cpu')
    file_name = 'render_apps/UnderWaterDrones_IM_Round_{}Robots_At{}'.format(n_robo, n_at)

    if n_robo == 1:
        file_name = 'render_apps/UnderWaterDrones_IM_Round_OneRobot_At{}'.format(n_at)

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

    xs = []
    ys = []
    zs = []

    rss = []
    gss = []
    bss = []
    impss = []

    while True:
        for i in decision_steps.agent_id:
            camera_obs = np.transpose(agent.img_scale(decision_steps.obs[0][i] * 255), (2, 1, 0))
            action = agent.get_action(camera_obs)

            # print('action:', action)
            action_tuple = ActionTuple(continuous=np.expand_dims(action, axis=0))
            env.set_action_for_agent(behavior_names[0], i, action_tuple)

            xs.append(decision_steps.obs[1][i][config.n_fitness + 1:][0])
            ys.append(decision_steps.obs[1][i][config.n_fitness + 1:][1])
            zs.append(decision_steps.obs[1][i][config.n_fitness + 1:][2])
        env.step()
        img = agent.img_scale(decision_steps.obs[0][0] * 255)

        # 入力値（画像）
        # simple_img = cv2.resize(img, (400, 400))[:, :, ::-1]
        # cv2.imwrite('simple_obs/img_' + str(counter) + '.png', simple_img)

        rs, gs, bs, imps = agent.show_gui(obs=img, counter=counter, path=log_dir)
        rss += rs
        gss += gs
        bss += bs
        impss += imps

        counter += 1
        decision_steps, terminal_steps = env.get_steps(behavior_names[0])
        print('steps:', counter)
        if counter % config.steps == 0:
            robots_coordinates = np.array([xs, ys, zs])
            pickle.dump(robots_coordinates, open(log_dir + 'robots_coordinates_{}.pkl'.format(n_recode), 'wb'))
            u.coordinates_heatmap_creator(path=log_dir, coordinates=robots_coordinates, n_recode=n_recode)

            apm = (rss, gss, bss, impss)
            pickle.dump(apm, open(log_dir + 'ap_material_{}.pkl'.format(n_recode), 'wb'))
            u.at_scatter_creator(path=log_dir, n_recode=n_recode, apm=apm)

            n_recode += 1
            env.reset()

            xs = []
            ys = []
            zs = []

            rss = []
            gss = []
            bss = []
            impss = []

            if n_recode >= config.reps:
                env.close()
                return


if __name__ == '__main__':
    args = parse_args()
    d = Display()
    if args.headless:
        d.start()

    for n_version in args.versions:
        for n_robo in args.ns_robo:
            log_dir = "log/at{}/round_im_{}_robo_slurm_at{}/".format(args.n_at, n_robo, args.n_at)

            while not os.path.isfile(log_dir + args.load_model):
                print(log_dir + args.load_model + " is not exist.")
                time.sleep(5)

            main(args, log_dir, args.n_at, n_robo, n_version)

            res_1 = subprocess.run("sh recode_pi_behave.sh {} {} {}".format(args.n_at, n_robo, n_version), shell=True)
            print("res_1:", res_1.returncode)
            res_2 = subprocess.run("sh recode_pi_attention.sh {} {} {}".format(args.n_at, n_robo, n_version), shell=True)
            print("res_2:", res_2.returncode)

    if args.headless:
        d.stop()
