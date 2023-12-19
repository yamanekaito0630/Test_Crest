# ライブラリのインポート
import os
import numpy as np
import torch
import argparse

from pyvirtualdisplay import Display
from permutation_invariant.solutions_mpi_evojax import PIAttentionAgent
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', help='Path to log folder.', default='log/round_im_2_robots_slurm_at2_disvisible/')
    parser.add_argument('--load-model', help='Path to model file.', default='Iter_1000.npz')
    parser.add_argument('--n-fitness', help='Number of fitness', type=int, default=4)
    parser.add_argument('--num-eval', help='Number of eval', type=int, default=100)
    parser.add_argument('--worker-id', help='Worker Id', type=int, default=0)
    parser.add_argument('--headless', help='True or False', type=int, default=0)
    config, _ = parser.parse_known_args()
    return config


def main(config):
    device = torch.device('cpu')
    file_name = 'apps/UnderWaterDrones_IM_Round_16Robots_At4'
    agent = PIAttentionAgent(
        device=device,
        file_name=file_name,
        act_dim=3,
        msg_dim=16,
        pos_em_dim=8,
        patch_size=6,
        stack_k=3,
        aa_image_size=64,
        aa_query_dim=4,
        aa_hidden_dim=16,
        aa_top_k=50
    )
    agent.load(config.log_dir + config.load_model)

    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=20, width=1500, height=600, capture_frame_rate=50)
    env = UnityEnvironment(file_name=file_name, side_channels=[channel], worker_id=config.worker_id)

    total_scores = []
    each_scores = []

    for i in range(config.num_eval):
        env.reset()
        behavior_names = list(env.behavior_specs.keys())
        decision_steps, terminal_steps = env.get_steps(behavior_names[0])

        velocity = 0
        done = False
        reward = 0
        each_reward = np.array([0 for _ in range(config.n_fitness)])

        while not done:
            for j in decision_steps.agent_id:
                camera_obs = np.transpose(agent.img_scale(decision_steps.obs[0][j] * 255), (2, 1, 0))
                action = agent.get_action(camera_obs)
                # print('action:', action)
                action_tuple = ActionTuple(continuous=np.expand_dims(action, axis=0))
                env.set_action_for_agent(behavior_names[0], j, action_tuple)
                velocity += decision_steps.obs[1][j][config.n_fitness:][0]
            env.step()
            decision_steps, terminal_steps = env.get_steps(behavior_names[0])
            reward += sum(decision_steps.reward)
            done = len(terminal_steps.interrupted) > 0
            if done:
                for j in terminal_steps.agent_id:
                    each_reward = each_reward + terminal_steps.obs[1][j][:config.n_fitness]
                n_arrival = each_reward[2] / len(terminal_steps.agent_id)

                if not os.path.isfile(config.log_dir + 'n_one_robot_arrivals.txt'):
                    os.path.join(config.log_dir, 'n_one_robot_arrivals.txt')

                if not os.path.isfile(config.log_dir + 'one_robot_velocities.txt'):
                    os.path.join(config.log_dir, 'one_robot_velocities.txt')

                with open(file=config.log_dir + 'n_one_robot_arrivals.txt', mode='a') as f:
                    f.write('{:.2f}\n'.format(n_arrival))

                with open(file=config.log_dir + 'one_robot_velocities.txt', mode='a') as f:
                    f.write('{:.2f}\n'.format(velocity / (len(terminal_steps.agent_id) * 100)))

        total_scores.append(reward)
        each_scores.append(each_reward)
        print('num={0}, score={1:.2f}'.format(i, reward))

    print('total_scores_mean:', np.mean(total_scores), 'std:', np.std(total_scores))
    print('each_scores_mean:', np.mean(each_scores, axis=0), 'std:', np.std(each_scores, axis=0))


if __name__ == '__main__':
    args = parse_args()
    d = Display()
    if args.headless:
        d.start()

    main(args)

    if args.headless:
        d.stop()
