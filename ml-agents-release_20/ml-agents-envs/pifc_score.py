# ライブラリのインポート
import numpy as np
import torch
import argparse
import gc
import sys

from pyvirtualdisplay import Display
from permutation_invariant.solutions_mpi_evojax import PIFCSolution
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-model', help='Path to model file.', default='log/round_ir_one_robot_slurm/Iter_1000.npz')
    parser.add_argument('--steps', help='Steps for eval', type=int, default=3000)
    parser.add_argument('--n-fitness', help='num of fitness', type=int, default=4)
    parser.add_argument('--headless', help='True or False', type=bool, default=False)
    config, _ = parser.parse_known_args()
    return config


def main(config):
    device = torch.device('cpu')
    file_name = 'render_apps/UnderWaterDrones_IR_Round_OneRobot'
    agent = PIFCSolution(
        device=device,
        file_name=file_name,
        act_dim=5,
        hidden_dim=64,
        msg_dim=32,
        pos_em_dim=16,
        num_hidden_layers=2
    )
    agent.load(config.load_model)

    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=1, width=1500, height=600, capture_frame_rate=50)
    env = UnityEnvironment(file_name=file_name, side_channels=[channel])
    env.reset()

    behavior_names = list(env.behavior_specs.keys())
    decision_steps, terminal_steps = env.get_steps(behavior_names[0])
    counter = 0
    while True:
        for i in decision_steps.agent_id:
            obs = np.concatenate([decision_steps.obs[0][i][3::4],
                                  decision_steps.obs[1][i][3::4],
                                  decision_steps.obs[2][i][3::4],
                                  decision_steps.obs[3][i][config.n_fitness:]])
            action = agent.get_action(obs)
            # print('action:', action)
            action_tuple = ActionTuple(continuous=np.expand_dims(action, axis=0))
            env.set_action_for_agent(behavior_names[0], i, action_tuple)
        env.step()
        counter += 1
        decision_steps, terminal_steps = env.get_steps(behavior_names[0])
        print('steps:', counter * 10)
        if counter > (config.steps - 1) / 10:
            env.close()
            sys.exit()


if __name__ == '__main__':
    args = parse_args()
    d = Display()
    if args.headless:
        d.start()

    main(args)

    if args.headless:
        d.stop()
