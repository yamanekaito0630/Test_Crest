import utils as u
import argparse
import pickle
import gc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-at', help='Num of attempt', type=int, default=4)
    parser.add_argument('--ns-trial', required=True, nargs="*", help='Num list of trial', type=int)
    parser.add_argument('--ns-robo', required=True, nargs="*", help='Num list of robo', type=int)
    parser.add_argument('--es-robo', required=True, nargs="*", help='Num list of eval robo', type=int)
    parser.add_argument('--version', help='version', type=int, default=3)
    parser.add_argument('--reps', help='Num of attempt', type=int, default=1)
    parser.add_argument('--env-name', help='name of environment', type=str, default="default")
    config, _ = parser.parse_known_args()
    return config


def main(config):
    for i in config.ns_robo:
        for t in config.ns_trial:
            for j in config.es_robo:
                base_dir = 'log/at{at}/{n_robo}robo/v{v}/trial_{t}/eval_{e_robo}robo/{env_name}/'.format(at=config.n_at, n_robo=i, v=config.version, t=t, e_robo=j, env_name=config.env_name)
                for k in range(config.reps):
                # for k in [2]:
                    # c = pickle.load(open(base_dir + 'robots_coordinates_{}.pkl'.format(k), 'rb'))
                    # u.coordinates_3d(path=base_dir, n_recode=k, coordinates=c, n_robo=j)
                    # nodes = pickle.load(open(base_dir + 'episode_nodes_{}.pkl'.format(k), 'rb'))
                    # links = pickle.load(open(base_dir + 'episode_links_{}.pkl'.format(k), 'rb'))
                    # u.network_analyzer(path=base_dir, nodes=nodes, links=links, n_robo=j, n_recode=k)
                    # u.z_t_creator(path=base_dir, n_recode=k, coordinates=c, n_robo=j)
                    # u.save_movement_amount_of_z(path=base_dir, coordinates=c, n_robo=j)
                    # if j > 1:
                    #     u.save_variance_of_z(path=base_dir, coordinates=c, n_robo=j)
                    if k == 0:
                        # apm = pickle.load(open(base_dir + 'ap_material_{}.pkl'.format(k), 'rb'))
                        # u.rgb_analyzer(path=base_dir, apm=apm)
                        obs_actions = pickle.load(open(base_dir + 'obs_actions_{}.pkl'.format(k), 'rb'))
                        u.get_scatter_of_actions(path=base_dir, obs_actions=obs_actions)
                        obs_leds = pickle.load(open(base_dir + 'obs_leds_{}.pkl'.format(k), 'rb'))
                        u.get_scatter_of_leds(path=base_dir, obs_leds=obs_leds)
                        # u.at_scatter_creator(path=base_dir, n_recode=k, apm=apm)
                        # del apm
                        # gc.collect()


if __name__ == '__main__':
    args = parse_args()
    main(args)
