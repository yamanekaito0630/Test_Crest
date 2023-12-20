# ライブラリのインポート
import argparse
import torch

from pyvirtualdisplay import Display
from permutation_invariant.solutions_mpi_evojax import PIAttentionAgent, PIFCSolution, AttentionAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--os', help='Mac or Linux', type=str, default="Linux")
    parser.add_argument('--t', help='Num of loop', type=int, default=1)
    parser.add_argument('--base', help='piaa, pifc, aa', default='piaa')
    parser.add_argument('--version', help='Number of attenpt', type=int, default=1)
    parser.add_argument('--n-at', help='Number of attenpt', type=int, default=5)
    parser.add_argument('--n-robo', help='Number of robots', type=int, default=2)
    parser.add_argument('--n-fitness', help='Number of fitness', type=int, default=6)
    parser.add_argument('--log-dir', help='Path to log directory.', default='log/marine_drones_12robo')
    parser.add_argument('--load-model', help='Path to model file.', default=None)
    parser.add_argument('--algo', help='Select Algorithms (cma, open_es, ga, pepg)', type=str, default="cma")
    parser.add_argument('--max-iter', help='Max training iterations.', type=int, default=1000)
    parser.add_argument('--is-resume', help='Restart training', type=int, default=0)
    parser.add_argument('--from-iter', help='From training iterations.', type=int, default=1)
    parser.add_argument('--save-interval', help='Model saving period.', type=int, default=50)
    parser.add_argument('--seed', help='Random seed for evaluation.', type=int, default=42)
    parser.add_argument('--reps', help='Number of rollouts for fitness.', type=int, default=1)
    parser.add_argument('--init-sigma', help='Initial std.', type=float, default=0.1)
    parser.add_argument('--init-best', help='Initial best.', type=float, default=-float('Inf'))
    config, _ = parser.parse_known_args()
    return config


def main(config):
    device = torch.device('cpu')
    if config.base == 'piaa':
        if config.n_robo == 1:
            file_name = 'Test_Crest_App/{}/app/UnderWaterDrones_IM_Round_OneRobot_At{}_V{}'.format(config.os,
                                                                                                   config.n_at,
                                                                                                   config.version)
        else:
            file_name = 'Test_Crest_App/{}/app/UnderWaterDrones_IM_Round_{}Robots_At{}_V{}'.format(config.os,
                                                                                                   config.n_robo,
                                                                                                   config.n_at,
                                                                                                   config.version)
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
    elif config.base == 'aa':
        if config.n_robo == 1:
            file_name = 'Test_Crest_App/{}/apps/UnderWaterDrones_IM_Round_OneRobot_At{}_V{}'.format(config.os,
                                                                                                    config.n_at,
                                                                                                    config.version)
        else:
            file_name = 'Test_Crest_App/{}/apps/UnderWaterDrones_IM_Round_{}Robots_At{}_V{}'.format(config.os,
                                                                                                    config.n_robo,
                                                                                                    config.n_at,
                                                                                                    config.version)
        agent = AttentionAgent(
            device=device,
            file_name=file_name,
            image_size=96,
            patch_size=7,
            patch_stride=4,
            query_dim=4,
            hidden_dim=16,
            top_k=50
        )
    elif config.base == 'pifc':
        file_name = 'apps/UnderWaterDrones_IR_Round_OneRobot'
        agent = PIFCSolution(
            device=device,
            file_name=file_name,
            act_dim=3,
            hidden_dim=64,
            msg_dim=16,
            pos_em_dim=8,
            num_hidden_layers=2
        )

    agent.train(
        t=config.t,
        base=config.base,
        algo=config.algo,
        max_iter=config.max_iter,
        reps=config.reps,
        is_resume=config.is_resume,
        from_iter=config.from_iter,
        save_interval=config.save_interval,
        log_dir=config.log_dir,
        seed=config.seed,
        init_best=config.init_best,
        n_fitness=config.n_fitness
    )


if __name__ == '__main__':
    args = parse_args()
    d = Display()

    if args.os == "Linux":
        # 仮想ディスプレイの使用
        d.start()

    main(args)

    if args.os == "Linux":
        d.stop()
