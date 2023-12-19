# ライブラリのインポート
import logging
import math
import os
import time
import pickle
import cv2
import numpy as np
import torch
import torch.nn as nn
import gc

from evolution_strategies.es import CMAES, SimpleGA, OpenES, PEPG

from mpi4py import MPI
from torchvision.transforms import transforms
from PIL import Image
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from permutation_invariant.base_solution import BaseSolution
from permutation_invariant.modules import SelfAttentionMatrix, VisionAttentionNeuronLayer, AttentionNeuronLayer

torch.set_num_threads(1)


class BaseTorchSolution(BaseSolution):
    def __init__(self, device):
        self.is_resume = None
        self.seed = None
        self.reps = None
        self.max_iter = None
        self.popsize = None
        self.algo = None
        self.env = None
        self.file_name = None
        self.modules_to_learn = []
        self.device = torch.device(device)

    def get_action(self, obs):
        with torch.no_grad():
            return self._get_action(obs)

    def get_params(self):
        params = []
        with torch.no_grad():
            for layer in self.modules_to_learn:
                for p in layer.parameters():
                    params.append(p.cpu().numpy().ravel())
        return np.concatenate(params)

    def set_params(self, params):
        params = np.array(params)
        assert isinstance(params, np.ndarray)
        ss = 0
        for layer in self.modules_to_learn:
            for p in layer.parameters():
                ee = ss + np.prod(p.shape)
                p.data = torch.from_numpy(
                    params[ss:ee].reshape(p.shape)
                ).float().to(self.device)
                ss = ee
        assert ss == params.size

    def save(self, filename):
        params = self.get_params()
        np.savez(filename, params=params)

    def load(self, filename):
        with np.load(filename) as data:
            params = data['params']
            self.set_params(params)

    def get_num_params(self):
        return self.get_params().size

    def _get_action(self, obs):
        raise NotImplementedError()

    def reset(self):
        pass

    def piaa_get_fitness(self, worker_id, params, seed, num_rollouts, n_fitness):
        self.set_params(params)
        # print('worker_id:', worker_id)
        # np.random.seed(seed)

        total_scores = []
        each_scores = []
        for _ in range(num_rollouts):
            self.env.reset()
            behavior_names = list(self.env.behavior_specs.keys())
            decision_steps, terminal_steps = self.env.get_steps(behavior_names[0])

            done = False
            reward = 0
            each_reward = np.array([0 for _ in range(n_fitness)])
            while not done:
                start = time.time()
                for i in decision_steps.agent_id:
                    camera_obs = np.transpose((self.img_scale(decision_steps.obs[0][i] * 255)), (2, 1, 0))
                    action = self.get_action(camera_obs)
                    action_tuple = ActionTuple(continuous=np.expand_dims(action, axis=0))
                    self.env.set_action_for_agent(behavior_names[0], i, action_tuple)

                self.env.step()
                decision_steps, terminal_steps = self.env.get_steps(behavior_names[0])
                reward += sum(decision_steps.reward)
                done = len(terminal_steps.interrupted) > 0
                if done:
                    reward += sum(terminal_steps.reward)
                    for i in terminal_steps.agent_id:
                        each_reward = each_reward + terminal_steps.obs[1][i][:n_fitness]
                end = time.time()

                # print('worker_id:', worker_id, 'times:', end - start)
            total_scores.append(reward)
            each_scores.append(each_reward)

        # self.env.close()
        return np.mean(total_scores), np.mean(each_scores, axis=0)

    def aa_get_fitness(self, worker_id, params, seed, num_rollouts, n_fitness):
        self.set_params(params)
        # print('worker_id:', worker_id)
        # np.random.seed(seed)

        total_scores = []
        each_scores = []
        for _ in range(num_rollouts):
            self.env.reset()
            behavior_names = list(self.env.behavior_specs.keys())
            decision_steps, terminal_steps = self.env.get_steps(behavior_names[0])

            done = False
            reward = 0
            each_reward = np.array([0 for _ in range(n_fitness)])
            while not done:
                start = time.time()
                for i in decision_steps.agent_id:
                    camera_obs = np.transpose((self.img_scale(decision_steps.obs[0][i] * 255)), (2, 1, 0))
                    action = self.get_action(camera_obs)
                    action_tuple = ActionTuple(continuous=np.expand_dims(action, axis=0))
                    self.env.set_action_for_agent(behavior_names[0], i, action_tuple)

                self.env.step()
                decision_steps, terminal_steps = self.env.get_steps(behavior_names[0])
                reward += sum(decision_steps.reward)
                done = len(terminal_steps.interrupted) > 0
                if done:
                    reward += sum(terminal_steps.reward)
                    for i in terminal_steps.agent_id:
                        each_reward = each_reward + terminal_steps.obs[1][i][:n_fitness]
                end = time.time()

                # print('worker_id:', worker_id, 'times:', end - start)
            total_scores.append(reward)
            each_scores.append(each_reward)

        # self.env.close()
        return np.mean(total_scores), np.mean(each_scores, axis=0)

    def pifc_get_fitness(self, worker_id, params, seed, num_rollouts, n_fitness):
        self.set_params(params)
        # print('worker_id:', worker_id)
        np.random.seed(seed)

        total_scores = []
        each_scores = []
        for _ in range(num_rollouts):
            self.env.reset()
            behavior_names = list(self.env.behavior_specs.keys())
            decision_steps, terminal_steps = self.env.get_steps(behavior_names[0])

            done = False
            reward = 0
            each_reward = np.array([0, 0, 0, 0])
            while not done:
                start = time.time()
                for i in decision_steps.agent_id:
                    obs = np.concatenate([decision_steps.obs[0][i][3::4],
                                          decision_steps.obs[1][i][3::4],
                                          decision_steps.obs[2][i][3::4],
                                          decision_steps.obs[3][i][n_fitness:]])
                    action = self.get_action(obs)
                    action_tuple = ActionTuple(continuous=np.expand_dims(action, axis=0))
                    self.env.set_action_for_agent(behavior_names[0], i, action_tuple)

                self.env.step()
                decision_steps, terminal_steps = self.env.get_steps(behavior_names[0])
                reward += sum(decision_steps.reward)
                done = len(terminal_steps.interrupted) > 0
                if done:
                    for i in terminal_steps.agent_id:
                        each_reward = each_reward + terminal_steps.obs[3][i][:n_fitness]
                end = time.time()

                # print('worker_id:', worker_id, 'times:', end - start)
            total_scores.append(reward)
            each_scores.append(each_reward)

        # self.env.close()
        return np.mean(total_scores), np.mean(each_scores, axis=0)

    @staticmethod
    def img_scale(img, margin=5):
        img = cv2.resize(img, (img.shape[0] + 2 * margin, img.shape[1] + 2 * margin))
        height, width, _ = img.shape
        return img[margin:height - margin, margin:width - margin]

    @staticmethod
    def action_scale(actions):
        scaled_actions = []
        for action in actions:
            scaled_action = np.tanh(action)
            scaled_actions.append(scaled_action)

        return np.array(scaled_actions)

    @staticmethod
    def save_params(solver, solution, model_path):
        solution.set_params(solver.best_param())
        solution.save(model_path)

    def create_logger(self, name, log_dir=None, debug=False, base='piaa'):
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            os.makedirs(log_dir + '/gen_params')
            os.makedirs(log_dir + '/gen_fitnesses')
            os.makedirs(log_dir + '/gen_es')
            os.makedirs(log_dir + '/each_fitneses')
        log_format = '%(asctime)s %(process)d [%(levelname)s] %(message)s'
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO, format=log_format)
        logger = logging.getLogger(name)
        if log_dir:
            if not os.path.isfile(log_dir + '/n_one_robot_arrivals.txt'):
                os.path.join(log_dir, 'n_one_robot_arrivals.txt')

            if not os.path.isfile(log_dir + '/one_robot_velocities.txt'):
                os.path.join(log_dir, 'one_robot_velocities.txt')

            if not os.path.isfile(log_dir + '/hyper_parameters.txt'):
                os.path.join(log_dir, 'hyper_parameters.txt')
                with open(file=log_dir + '/hyper_parameters.txt', mode='a') as f:
                    if base == 'piaa':
                        f.write(
                            'file_name={}\nnum_param={}\nsalgolithm={}\npopulation_size={}\nmax_iter={}\nroll_out={}\n'
                            '\nact_dim={}\nmsg_dim={}\npos_em_dim={}\npatch_size={}\nstack_k={}\naa_image_size={}\n'
                            'aa_query_dim={}\naa_hidden_dim={}\naa_top_k={}'.format(self.file_name,
                                                                                    self.get_num_params(),
                                                                                    self.algo,
                                                                                    self.popsize, self.max_iter,
                                                                                    self.reps,
                                                                                    self.act_dim, self.msg_dim,
                                                                                    self.pos_em_dim,
                                                                                    self.patch_size,
                                                                                    self.stack_k,
                                                                                    self.aa_image_size,
                                                                                    self.aa_query_dim,
                                                                                    self.lstm_hidden_dim,
                                                                                    self.top_k))
                    elif base == 'aa':
                        f.write(
                            'file_name={}\nnum_patches={}\nnum_param={}\nsalgolithm={}\npopulation_size={}\nmax_iter={}\nroll_out={}\n'
                            '\nact_dim={}\npatch_size={}\npatch_stride={}\nquery_dim={}\nhidden_dim={}\nimage_size={}\n'
                            'top_k={}'.format(self.file_name,
                                              self.num_patches,
                                              self.get_num_params(),
                                              self.algo,
                                              self.popsize, self.max_iter,
                                              self.reps,
                                              self.act_dim, self.patch_size,
                                              self.patch_stride,
                                              self.query_dim,
                                              self.hidden_dim,
                                              self.image_size,
                                              self.top_k))
            log_file = os.path.join(log_dir, '{}.txt'.format(name))
            file_hdl = logging.FileHandler(log_file)
            formatter = logging.Formatter(fmt=log_format)
            file_hdl.setFormatter(formatter)
            logger.addHandler(file_hdl)
        return logger

    @staticmethod
    def each_fitness_logger(log_dir, n_iter, each_fitnesses):
        sum_fitness = np.sum(each_fitnesses, axis=1)
        max_index = np.argmax(sum_fitness)
        max_fitness = each_fitnesses[max_index]
        max_div_path = log_dir + '/each_fitneses/fitness_max_div.txt'

        if not os.path.exists(max_div_path):
            os.path.join(max_div_path)
        with open(file=max_div_path, mode='a') as f:
            for i in range(len(max_fitness)):
                if i == 0:
                    f.write('Iter={0},{1:.2f},'.format(n_iter + 1, max_fitness[i]))
                elif i == len(max_fitness) - 1:
                    f.write('{:.2f}\n'.format(max_fitness[i]))
                else:
                    f.write('{:.2f},'.format(max_fitness[i]))

        n_fitness = each_fitnesses.shape[1]
        for i in range(n_fitness):
            path = log_dir + '/each_fitneses/fitness_{}.txt'.format(i + 1)
            if not os.path.exists(path):
                os.path.join(path)
            with open(file=path, mode='a') as f:
                f.write('Iter={0}, '
                        'max={1:.2f}, avg={2:.2f}, min={3:.2f}, std={4:.2f}\n'.format(n_iter + 1,
                                                                                      np.max(each_fitnesses[:, i]),
                                                                                      np.mean(each_fitnesses[:, i]),
                                                                                      np.min(each_fitnesses[:, i]),
                                                                                      np.std(each_fitnesses[:, i])))

    def init_run(self):
        self.env.reset()
        behavior_names = list(self.env.behavior_specs.keys())
        decision_steps, terminal_steps = self.env.get_steps(behavior_names[0])

        done = False
        while not done:
            start = time.time()
            for i in decision_steps.agent_id:
                action = (np.random.rand(self.act_dim) * 2.0 - 1.0).reshape(1, self.act_dim).astype(np.float32)
                action_tuple = ActionTuple(continuous=action)
                self.env.set_action_for_agent(behavior_names[0], i, action_tuple)
            self.env.step()
            decision_steps, terminal_steps = self.env.get_steps(behavior_names[0])
            done = len(terminal_steps.interrupted) > 0

    def train(self,
              t: int = 1,
              base: str = 'piaa',
              algo: str = "cma",
              max_iter: int = 1000,
              reps: int = 1,
              is_resume: bool = False,
              from_iter: int = 1,
              log_dir: str = None,
              save_interval: int = 10,
              seed: int = 42,
              init_best: float = -float('Inf'),
              n_fitness: int = 4,
              ):
        solver, best_so_far, logger, params_sets, rnd = None, None, None, None, None
        ii32 = np.iinfo(np.int32)
        rnd = np.random.RandomState(seed=seed)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            self.algo = algo
            self.popsize = size * t
            self.max_iter = max_iter
            self.reps = reps
            self.seed = seed
            logger = self.create_logger(name='train_log', log_dir=log_dir, base=base)
            best_so_far = init_best

            num_params = self.get_num_params()
            if base == 'aa':
                print('#num_patches={}'.format(self.num_patches))
            print('#params={}'.format(num_params))
            if is_resume:
                print("is_resume: True")
                solver = pickle.load(open(log_dir + '/gen_es/es_' + algo + '_{}.pkl'.format(from_iter - 1), 'rb'))
            else:
                print("is_resume: False")
                from_iter = 1
                if algo == 'open_es':
                    solver = OpenES(
                        num_params=num_params,
                        sigma_init=0.1,
                        sigma_decay=0.999,
                        sigma_limit=0.01,
                        learning_rate=0.01,
                        learning_rate_decay=0.9999,
                        learning_rate_limit=0.001,
                        popsize=size * t,
                        antithetic=False,
                        weight_decay=0.01,
                        rank_fitness=True,
                        forget_best=True
                    )
                elif algo == 'ga':
                    solver = SimpleGA(
                        num_params=num_params,
                        sigma_init=0.1,
                        sigma_decay=0.999,
                        sigma_limit=0.01,
                        popsize=size * t,
                        elite_ratio=0.1,
                        forget_best=False,
                        weight_decay=0.01,
                    )
                elif algo == 'pepg':
                    solver = PEPG(
                        num_params=num_params,
                        sigma_init=0.10,
                        sigma_alpha=0.20,
                        sigma_decay=0.999,
                        sigma_limit=0.01,
                        sigma_max_change=0.2,
                        learning_rate=0.01,
                        learning_rate_decay=0.9999,
                        learning_rate_limit=0.01,
                        elite_ratio=0,
                        popsize=size * t,
                        average_baseline=True,
                        weight_decay=0.01,
                        rank_fitness=True,
                        forget_best=True
                    )
                elif algo == 'cma':
                    solver = CMAES(
                        num_params=num_params,
                        sigma_init=0.10,
                        popsize=size * t,
                        weight_decay=0.01
                    )

        comm.barrier()
        # Unity環境の生成
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=20, width=10, height=10)
        self.env = UnityEnvironment(file_name=self.file_name, no_graphics=False, side_channels=[channel],
                                    worker_id=rank, seed=seed)
        self.init_run()

        for n_iter in range(from_iter - 1, max_iter):
            task_seed = rnd.randint(0, ii32.max)
            comm.barrier()
            if rank == 0:
                params_sets = solver.ask()

                # パラメータの保存
                if (n_iter + 1) % save_interval == 0:
                    params_path = os.path.join(log_dir + '/gen_params', 'params_{}.npz'.format(n_iter + 1))
                    np.savez(params_path, params=params_sets)

                c_rank = 0
                for i in range(0, len(params_sets), t):
                    if c_rank == 0:
                        params_set = params_sets[i:i + t]
                    else:
                        data = params_sets[i:i + t]
                        comm.send(data, dest=c_rank, tag=c_rank)
                    c_rank += 1
            else:
                data = comm.recv(source=0, tag=rank)
                params_set = data

            fitness = []
            each_fitness = []
            for i in range(t):
                if base == 'piaa':
                    f, e_f = self.piaa_get_fitness(rank, params_set[i], task_seed, reps, n_fitness)
                elif base == 'aa':
                    f, e_f = self.aa_get_fitness(rank, params_set[i], task_seed, reps, n_fitness)
                else:
                    f, e_f = self.pifc_get_fitness(rank, params_set, task_seed, reps, n_fitness)
                fitness.append(f)
                each_fitness.append(e_f)

            fitnesses = comm.gather(fitness, root=0)
            each_fitnesses = comm.gather(each_fitness, root=0)
            if rank == 0:
                fitnesses = np.concatenate(np.array(fitnesses))
                each_fitnesses = np.concatenate(np.array(each_fitnesses))

                self.each_fitness_logger(log_dir=log_dir, n_iter=n_iter, each_fitnesses=each_fitnesses)

                # 適応度の保存
                if (n_iter + 1) % save_interval == 0:
                    fitnesses_path = os.path.join(log_dir + '/gen_fitnesses', 'fitnesses_{}.npz'.format(n_iter + 1))
                    np.savez(fitnesses_path, fitnesses=fitnesses)

                solver.tell(fitnesses)

                # ESの保存
                if (n_iter + 1) % save_interval == 0:
                    pickle.dump(solver, open(log_dir + '/gen_es/es_' + algo + '_{}.pkl'.format(n_iter + 1), 'wb'))

                logger.info(
                    'Iter={0}, '
                    'max={1:.2f}, avg={2:.2f}, min={3:.2f}, std={4:.2f}'.format(
                        n_iter + 1, np.max(fitnesses), np.mean(fitnesses), np.min(fitnesses), np.std(fitnesses)))

                best_fitness = max(fitnesses)
                if best_fitness > best_so_far:
                    best_so_far = best_fitness
                    model_path = os.path.join(log_dir, 'best.npz')
                    self.save_params(solver=solver, solution=self, model_path=model_path)
                    logger.info('Best model updated, score={}'.format(best_fitness))
                if (n_iter + 1) % save_interval == 0:
                    model_path = os.path.join(log_dir, 'Iter_{}.npz'.format(n_iter + 1))
                    self.save_params(solver=solver, solution=self, model_path=model_path)
            gc.collect()

        self.env.close()


class PIAttentionAgent(BaseTorchSolution):
    def __init__(self,
                 device,
                 file_name,
                 act_dim,
                 msg_dim,
                 pos_em_dim,
                 patch_size=6,
                 stack_k=4,
                 aa_image_size=100,
                 aa_query_dim=4,
                 aa_hidden_dim=16,
                 aa_top_k=10):
        super(PIAttentionAgent, self).__init__(device)
        self.aa_query_dim = aa_query_dim
        self.stack_k = stack_k
        self.pos_em_dim = pos_em_dim
        self.file_name = file_name

        self.patch_size = patch_size
        self.aa_image_size = aa_image_size

        self.alpha = 0.
        self.attended_patch_ix = None
        self.patches_importance_vector = None
        self.act_dim = act_dim
        self.prev_act = torch.zeros(1, self.act_dim)
        self.hidden_dim = aa_image_size ** 2
        self.msg_dim = msg_dim
        self.prev_hidden = torch.zeros(self.hidden_dim, self.msg_dim)

        self.vision_pi_layer = VisionAttentionNeuronLayer(
            act_dim=act_dim,
            hidden_dim=aa_image_size ** 2,
            msg_dim=msg_dim,
            pos_em_dim=pos_em_dim,
            patch_size=patch_size,
            stack_k=stack_k
        )
        self.modules_to_learn.append(self.vision_pi_layer)

        self.top_k = aa_top_k
        self.patch_centers = torch.div(torch.tensor(
            [[i, j] for i in range(aa_image_size) for j in range(aa_image_size)]
        ).float(), aa_image_size)

        self.attention = SelfAttentionMatrix(
            dim_in=self.msg_dim,
            msg_dim=aa_query_dim,
            scale=True
        )
        self.modules_to_learn.append(self.attention)

        self.hx = None
        self.lstm_hidden_dim = aa_hidden_dim
        self.lstm = nn.LSTMCell(
            input_size=aa_top_k * 2,
            hidden_size=aa_hidden_dim
        )
        self.modules_to_learn.append(self.lstm)

        self.output_fc = nn.Sequential(
            nn.Linear(in_features=aa_hidden_dim,
                      out_features=act_dim)
        )
        self.modules_to_learn.append(self.output_fc)

        self.mixing_fc = nn.Sequential(
            nn.Linear(in_features=aa_hidden_dim + act_dim,
                      out_features=aa_hidden_dim),
            nn.Tanh(),
            nn.Linear(in_features=aa_hidden_dim,
                      out_features=1),
            nn.Sigmoid()
        )
        self.modules_to_learn.append(self.mixing_fc)

    def _get_action(self, obs):
        x = self.vision_pi_layer(obs=obs, prev_act=self.prev_act)

        self.attended_patch_ix = (
            self.vision_pi_layer.attention.mostly_attended_entries
        )
        self.patches_importance_vector = (
            self.vision_pi_layer.attention.importance_vector
        )

        x = (1 - self.alpha) * x + self.alpha * self.prev_hidden
        self.prev_hidden = x

        attention_matrix = self.attention(data_q=x, data_k=x)
        patch_importance_matrix = torch.softmax(attention_matrix, dim=-1)
        patch_importance = patch_importance_matrix.sum(dim=0)

        ix = torch.argsort(patch_importance, descending=True)
        top_k_ix = ix[:self.top_k]
        centers = self.patch_centers[top_k_ix]
        centers = centers.flatten(0, -1)

        if self.hx is None:
            self.hx = (
                torch.zeros(1, self.lstm_hidden_dim),
                torch.zeros(1, self.lstm_hidden_dim),
            )
        self.hx = self.lstm(centers.unsqueeze(0), self.hx)
        output = self.output_fc(self.hx[0])
        self.prev_act = output

        self.alpha = self.mixing_fc(
            torch.cat([self.hx[0], self.prev_act], dim=-1).squeeze(0)
        )
        action = output.squeeze(0).cpu().numpy()
        action = self.action_scale(actions=action)
        
        gc.collect()
        return action

    def reset(self):
        self.alpha = 0.
        self.prev_act = torch.zeros(1, self.act_dim)
        self.prev_hidden = torch.zeros(self.hidden_dim, self.msg_dim)
        self.hx = None

    def plot_attention_patches(self, img, path, counter, attention_patch_ix, patches_importance_vector):
        # print('importances:', patches_importance_vector.sort())

        attention_patch = np.ones([self.patch_size, self.patch_size, 3])
        num_patches = img.shape[0] // self.patch_size
        black_img = img * np.array([0, 0, 0])

        counter_0 = 0
        counter_1 = 0
        counter_2 = 0
        counter_3 = 0
        counter_4 = 0
        counter_5 = 0
        counter_6 = 0
        counter_7 = 0
        counter_8 = 0
        counter_9 = 0

        rs = []
        gs = []
        bs = []
        imps = []

        for ix in attention_patch_ix:
            row_ix = ix // num_patches
            col_ix = ix % num_patches
            row_ss = row_ix * self.patch_size
            col_ss = col_ix * self.patch_size
            row_ee = row_ss + self.patch_size
            col_ee = col_ss + self.patch_size

            # パッチ保存
            # patch = cv2.resize(img[row_ss:row_ee, col_ss:col_ee], (400, 400))[:, :, ::-1]

            rs += (img[row_ss:row_ee, col_ss:col_ee][:, :, 0]).flatten().tolist()
            gs += (img[row_ss:row_ee, col_ss:col_ee][:, :, 1]).flatten().tolist()
            bs += (img[row_ss:row_ee, col_ss:col_ee][:, :, 2]).flatten().tolist()
            imp = [patches_importance_vector[ix] for _ in range(self.patch_size ** 2)]
            imps += imp

            if patches_importance_vector[ix] <= 85:
                save_path = path + 'importance_0-25'
                counter_0 += 1
            elif patches_importance_vector[ix] <= 170:
                save_path = path + 'importance_25-50'
                counter_1 += 1
            elif patches_importance_vector[ix] > 170:
                save_path = path + 'importance_50-75'
                counter_2 += 1
            elif patches_importance_vector[ix] <= 100:
                save_path = path + 'importance_75-100'
                counter_3 += 1
            elif patches_importance_vector[ix] <= 125:
                save_path = path + 'importance_100-125'
                counter_4 += 1
            elif patches_importance_vector[ix] <= 150:
                save_path = path + 'importance_125-150'
                counter_5 += 1
            elif patches_importance_vector[ix] <= 175:
                save_path = path + 'importance_150-175'
                counter_6 += 1
            elif patches_importance_vector[ix] <= 200:
                save_path = path + 'importance_175-200'
                counter_7 += 1
            elif patches_importance_vector[ix] <= 225:
                save_path = path + 'importance_200-225'
                counter_8 += 1
            else:
                save_path = path + 'importance_225_250'
                counter_9 += 1

            # if path and not os.path.exists(save_path):
            #     os.makedirs(save_path)

            # cv2.imwrite(save_path + '/img_' + str(counter) + '_' + str(ix) + '.png', patch)
            # cv2.waitKey(1)

            if patches_importance_vector[ix] <= 0:
                patches_importance_vector[ix] = 0
            elif patches_importance_vector[ix] >= 250:
                patches_importance_vector[ix] = 250
            importance = math.floor(255 - (255 / 100) * patches_importance_vector[ix])

            black_img[row_ss:row_ee, col_ss:col_ee] = img[row_ss:row_ee, col_ss:col_ee]
            img[row_ss:row_ee, col_ss:col_ee] = (
                0.5 * img[row_ss:row_ee, col_ss:col_ee] + 0.5 * attention_patch * [255, importance, 0])

        ap = (rs, gs, bs, imps)
        return img.astype(np.uint8), black_img.astype(np.uint8), ap

    def show_gui(self, obs, counter, path):
        # print('obs.shape:', obs.shape)
        if hasattr(self, 'attended_patch_ix') and hasattr(self, 'patches_importance_vector'):
            attended_patch_ix = self.attended_patch_ix
            patches_importance_vector = self.patches_importance_vector
        else:
            attended_patch_ix = None
            patches_importance_vector = None

        if (attended_patch_ix is not None) and (patches_importance_vector is not None):
            obs, black_img, ap = self.plot_attention_patches(
                img=obs, path=path, counter=counter, attention_patch_ix=attended_patch_ix,
                patches_importance_vector=patches_importance_vector)

        img = cv2.resize(obs, (400, 400))[:, :, ::-1]
        black_img = cv2.resize(black_img, (400, 400))[:, :, ::-1]
        # obs保存
        save_path = path + 'attention_gui'
        if path and not os.path.exists(save_path):
            os.makedirs(save_path)
        if path and not os.path.exists(path + 'attention_movie'):
            os.makedirs(path + 'attention_movie')
        # cv2.imwrite(save_path + '/img_' + str(counter) + '.png', img)
        cv2.imshow('render', img)
        cv2.waitKey(1)

        save_path = path + 'attention_black_gui'
        if path and not os.path.exists(save_path):
            os.makedirs(save_path)
        if path and not os.path.exists(path + 'attention_black_movie'):
            os.makedirs(path + 'attention_black_movie')
        # cv2.imwrite(save_path + '/img_' + str(counter) + '.png', black_img)
        cv2.imshow('render', black_img)
        cv2.waitKey(1)
        gc.collect()

        return ap


class PIFCSolution(BaseTorchSolution):
    def __init__(self,
                 device,
                 file_name,
                 act_dim,
                 hidden_dim,
                 msg_dim,
                 pos_em_dim,
                 num_hidden_layers=2,
                 pi_layer_bias=True,
                 pi_layer_scale=True):
        super(PIFCSolution, self).__init__(device=device)
        self.file_name = file_name

        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.pos_em_dim = pos_em_dim
        self.prev_act = torch.zeros(1, self.act_dim)

        self.pi_layer = AttentionNeuronLayer(
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            msg_dim=msg_dim,
            pos_em_dim=pos_em_dim,
            bias=pi_layer_bias,
            scale=pi_layer_scale,
        )
        self.modules_to_learn.append(self.pi_layer)

        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_layers.extend([
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                nn.Tanh(),
            ])
        self.net = nn.Sequential(
            *hidden_layers,
            nn.Linear(in_features=hidden_dim, out_features=act_dim),
        )
        self.modules_to_learn.append(self.net)

    def _get_action(self, obs):
        x = self.pi_layer(obs=obs, prev_act=self.prev_act)
        self.prev_act = self.net(x.T)
        action = self.prev_act.squeeze(0).cpu().numpy()
        action = self.action_scale(actions=action)
        return action

    def reset(self):
        self.prev_act = torch.zeros(1, self.act_dim)
        self.pi_layer.reset()


class AttentionAgent(BaseTorchSolution):
    """Attention Agent solution."""

    def __init__(self,
                 device,
                 file_name,
                 image_size=96,
                 act_dim=3,
                 patch_size=7,
                 patch_stride=4,
                 query_dim=4,
                 hidden_dim=16,
                 top_k=10):
        super(AttentionAgent, self).__init__(device=device)
        self.file_name = file_name
        self.image_size = image_size
        self.act_dim = act_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        n = int((image_size - patch_size) / patch_stride + 1)
        offset = self.patch_size // 2
        patch_centers = []
        for i in range(n):
            patch_center_row = offset + i * patch_stride
            for j in range(n):
                patch_center_col = offset + j * patch_stride
                patch_centers.append([patch_center_row, patch_center_col])
        self.patch_centers = torch.tensor(patch_centers).float()

        self.num_patches = n ** 2
        self.attention = SelfAttentionMatrix(
            dim_in=3 * self.patch_size ** 2,
            msg_dim=query_dim,
        )
        self.modules_to_learn.append(self.attention)

        self.hx = None
        self.lstm = nn.LSTMCell(
            input_size=self.top_k * 2,
            hidden_size=hidden_dim,
        )
        self.modules_to_learn.append(self.lstm)

        self.output_fc = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=act_dim),
            nn.Tanh(),
        )
        self.modules_to_learn.append(self.output_fc)

    def _get_action(self, obs):
        # ob.shape = (h, w, c)
        ob = self.transform(torch.from_numpy(obs)).permute(1, 2, 0)
        h, w, c = ob.size()
        patches = ob.unfold(
            0, self.patch_size, self.patch_stride).permute(0, 3, 1, 2)
        patches = patches.unfold(
            2, self.patch_size, self.patch_stride).permute(0, 2, 1, 4, 3)
        patches = patches.reshape((-1, self.patch_size, self.patch_size, c))

        # flattened_patches.shape = (1, n, p * p * c)
        flattened_patches = patches.reshape(
            (1, -1, c * self.patch_size ** 2))
        # attention_matrix.shape = (1, n, n)
        attention_matrix = self.attention(flattened_patches, flattened_patches)
        # patch_importance_matrix.shape = (n, n)
        patch_importance_matrix = torch.softmax(
            attention_matrix.squeeze(), dim=-1)
        # patch_importance.shape = (n,)
        patch_importance = patch_importance_matrix.sum(dim=0)
        self.patches_importance_vector = patch_importance
        # extract top k important patches
        ix = torch.argsort(patch_importance, descending=True)
        top_k_ix = ix[:self.top_k]
        self.top_k_ix = top_k_ix

        centers = self.patch_centers[top_k_ix]
        self.centers = centers
        centers = centers.flatten(0, -1)
        centers = centers / self.image_size

        if self.hx is None:
            self.hx = (
                torch.zeros(1, self.hidden_dim),
                torch.zeros(1, self.hidden_dim),
            )
        self.hx = self.lstm(centers.unsqueeze(0), self.hx)
        output = self.output_fc(self.hx[0]).squeeze(0)
        
        gc.collect()
        return output.cpu().numpy()

    def reset(self):
        self.hx = None

    def plot_attention_patches(self, img, path, counter, attention_patch_ix, patches_importance_vector):
        # print('importances:', patches_importance_vector.sort())

        attention_patch = np.ones([self.patch_size, self.patch_size, 3])
        sub_img = img
        half_patch_size = self.patch_size // 2
        black_img = img * np.array([0, 0, 0])

        counter_0 = 0
        counter_1 = 0
        counter_2 = 0
        counter_3 = 0
        counter_4 = 0
        counter_5 = 0
        counter_6 = 0
        counter_7 = 0
        counter_8 = 0
        counter_9 = 0

        rs = []
        gs = []
        bs = []
        imps = []

        for i, center in enumerate(self.centers):
            row_ss = int(center[0]) - half_patch_size
            row_ee = int(center[0]) + half_patch_size + 1
            col_ss = int(center[1]) - half_patch_size
            col_ee = int(center[1]) + half_patch_size + 1
            ratio = 1.0 * i / self.top_k

            # パッチ保存
            # patch = cv2.resize(img[row_ss:row_ee, col_ss:col_ee], (400, 400))[:, :, ::-1]

            rs += (img[row_ss:row_ee, col_ss:col_ee][:, :, 0]).flatten().tolist()
            gs += (img[row_ss:row_ee, col_ss:col_ee][:, :, 1]).flatten().tolist()
            bs += (img[row_ss:row_ee, col_ss:col_ee][:, :, 2]).flatten().tolist()
            imp = [0 for _ in range(self.patch_size ** 2)]
            imps += imp

            # if patches_importance_vector[ix] <= 85:
            #     save_path = path + 'importance_0-25'
            #     counter_0 += 1
            # elif patches_importance_vector[ix] <= 170:
            #     save_path = path + 'importance_25-50'
            #     counter_1 += 1
            # elif patches_importance_vector[ix] > 170:
            #     save_path = path + 'importance_50-75'
            #     counter_2 += 1
            # elif patches_importance_vector[ix] <= 100:
            #     save_path = path + 'importance_75-100'
            #     counter_3 += 1
            # elif patches_importance_vector[ix] <= 125:
            #     save_path = path + 'importance_100-125'
            #     counter_4 += 1
            # elif patches_importance_vector[ix] <= 150:
            #     save_path = path + 'importance_125-150'
            #     counter_5 += 1
            # elif patches_importance_vector[ix] <= 175:
            #     save_path = path + 'importance_150-175'
            #     counter_6 += 1
            # elif patches_importance_vector[ix] <= 200:
            #     save_path = path + 'importance_175-200'
            #     counter_7 += 1
            # elif patches_importance_vector[ix] <= 225:
            #     save_path = path + 'importance_200-225'
            #     counter_8 += 1
            # else:
            #     save_path = path + 'importance_225_250'
            #     counter_9 += 1

            # if path and not os.path.exists(save_path):
            #     os.makedirs(save_path)

            # cv2.imwrite(save_path + '/img_' + str(counter) + '_' + str(ix) + '.png', patch)
            # cv2.waitKey(1)

            # if patches_importance_vector[ix] <= 0:
            #     patches_importance_vector[ix] = 0
            # elif patches_importance_vector[ix] >= 250:
            #     patches_importance_vector[ix] = 250
            # importance = math.floor(255 - (255 / 100) * patches_importance_vector[ix])
            importance = 255

            
            black_img[row_ss:row_ee, col_ss:col_ee] = sub_img[row_ss:row_ee, col_ss:col_ee]
            img[row_ss:row_ee, col_ss:col_ee] = (
                ratio * img[row_ss:row_ee, col_ss:col_ee] + (1 - ratio) * attention_patch * [255, importance, 0])

        ap = (rs, gs, bs, imps)
        return img.astype(np.uint8), black_img.astype(np.uint8), ap

    def show_gui(self, obs, counter, path):
        # print('obs.shape:', obs.shape)
        attended_patch_ix = self.top_k_ix
        patches_importance_vector = self.patches_importance_vector

        if (attended_patch_ix is not None) and (patches_importance_vector is not None):
            obs, black_img, ap = self.plot_attention_patches(
                img=obs, path=path, counter=counter, attention_patch_ix=attended_patch_ix,
                patches_importance_vector=patches_importance_vector)

        img = cv2.resize(obs, (400, 400))[:, :, ::-1]
        black_img = cv2.resize(black_img, (400, 400))[:, :, ::-1]
        # obs保存
        save_path = path + 'attention_gui'
        if path and not os.path.exists(save_path):
            os.makedirs(save_path)
        if path and not os.path.exists(path + 'attention_movie'):
            os.makedirs(path + 'attention_movie')
        cv2.imwrite(save_path + '/img_' + str(counter) + '.png', img)
        cv2.imshow('render', img)
        cv2.waitKey(1)

        save_path = path + 'attention_black_gui'
        if path and not os.path.exists(save_path):
            os.makedirs(save_path)
        if path and not os.path.exists(path + 'attention_black_movie'):
            os.makedirs(path + 'attention_black_movie')
        cv2.imwrite(save_path + '/img_' + str(counter) + '.png', black_img)
        cv2.imshow('render', black_img)
        cv2.waitKey(1)
        gc.collect()

        return ap
