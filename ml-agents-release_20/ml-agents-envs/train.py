# ライブラリのインポート
import numpy as np
import torch
import argparse
import sys
import torch.multiprocessing as mp
import time
from pyvirtualdisplay import Display

from permutation_invariant.solutions import PIFCSolution, PIAttentionAgent, MultiDrones
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='Task (piaa or pifc or drone)', default='piaa')
    parser.add_argument('--mode', help='Mode (train or test)', default='train')
    parser.add_argument('--log-dir', help='Path to log directory.', default='log/marine_drones_12robo')
    parser.add_argument('--load-model', help='Path to model file.', default='log/marine_drones_12robo/best.npz')
    parser.add_argument('--population-size', help='Population size', type=int, default=32)
    parser.add_argument('--num-workers', help='Number of workers', type=int, default=-1)
    parser.add_argument('--max-iter', help='Max training iterations.', type=int, default=1000)
    parser.add_argument('--save-interval', help='Model saving period.', type=int, default=10)
    parser.add_argument('--seed', help='Random seed for evaluation.', type=int, default=42)
    parser.add_argument('--reps', help='Number of rollouts for fitness.', type=int, default=1)
    parser.add_argument('--init-sigma', help='Initial std.', type=float, default=0.1)
    config, _ = parser.parse_known_args()
    return config


def main(config):
    device = torch.device('cpu')
    if config.task == 'piaa':
        file_name = 'apps/UnderWaterDrones_IM_Round_OneRobot_At2'
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
    elif config.task == 'pifc':
        file_name = 'apps/DroneRoundTrip'
        agent = PIFCSolution(
            device=device,
            file_name=file_name,
            act_dim=5,
            hidden_dim=16,
            msg_dim=16,
            pos_em_dim=8,
            num_hidden_layers=2
        )
    elif config.task == 'drone':
        file_name = 'apps/DronesToGoal'
        agent = MultiDrones(
            device=device,
            file_name=file_name,
            act_dim=5,
            msg_dim=16,
            pos_em_dim=8,
            patch_size=2,
            stack_k=3,
            aa_image_size=64,
            aa_query_dim=4,
            aa_hidden_dim=16,
            aa_top_k=50,
            ray_vector_dim=29
        )
    else:
        print(f'Task: "{config.task}" is not defined. ("piaa" or "pifc")')
        sys.exit()

    if config.mode == 'train':
        print('config.num_workers:', config.num_workers)
        agent.train(
            population_size=config.population_size,
            max_iter=config.max_iter,
            reps=config.reps,
            save_interval=config.save_interval,
            num_workers=config.num_workers,
            log_dir=config.log_dir,
            init_sigma=config.init_sigma,
            seed=config.seed
        )
    elif config.mode == 'test':
        file_name = file_name + 'Render'
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=1, width=600, height=600, capture_frame_rate=60)
        env = UnityEnvironment(file_name=file_name, side_channels=[channel], worker_id=config.num_workers)
        env.reset()

        agent.load(config.load_model)
        behavior_names = list(env.behavior_specs.keys())
        decision_steps, terminal_steps = env.get_steps(behavior_names[0])
        counter = 0
        while True:
            for i in decision_steps.agent_id:
                camera_obs = np.transpose(decision_steps.obs[0][i] * 255, (2, 1, 0))
                # print('obs:', obs)
                action = agent.get_action(camera_obs)
                # print('action:',action)
                # print('obs,',obs)
                # print('action,',action)
                action_tuple = ActionTuple(continuous=np.expand_dims(action, axis=0))
                env.set_action_for_agent(behavior_names[0], i, action_tuple)
                # with open('test.txt', mode='a') as f:
                #     f.write(f'{i}\n')

            env.step()

            agent.show_gui(decision_steps.obs[0][1] * 255, counter)
            counter += 1
            decision_steps, terminal_steps = env.get_steps(behavior_names[0])
            # print('rewards:', decision_steps.reward)
            done = len(terminal_steps.interrupted) > 0

            if done:
                env.reset()
                decision_steps, terminal_steps = env.get_steps(behavior_names[0])
                time.sleep(1 / 30)

    else:
        print(f'Mode: "{config.mode}" is not defined. ("train" or "test")')
        sys.exit()


if __name__ == '__main__':
    args = parse_args()
    if args.num_workers < 0:
        args.num_workers = mp.cpu_count()
    print('num_cpu: ', args.num_workers)

    d = Display()
    d.start()
    main(args)
    d.stop()
