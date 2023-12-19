import logging
import os
import gc
import pickle
import torch.multiprocessing as mp
import cma
import cv2
import numpy as np
import torch
import torch.nn as nn

from evolution_strategies.es import CMAES, SimpleGA, OpenES, PEPG
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from permutation_invariant.base_solution import BaseSolution
from permutation_invariant.modules import SelfAttentionMatrix, VisionAttentionNeuronLayer, AttentionNeuronLayer

torch.set_num_threads(1)
mp.set_sharing_strategy('file_system')


class BaseTorchSolution(BaseSolution):
    def __init__(self, device):
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

    def get_fitness(self, params):
        params, seed, num_rollouts = params
        self.set_params(params)
        worker_id = mp.current_process().ident % self.num_workers
        print('worker_id:', worker_id)

        # Unity環境の生成
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=20)
        file_name = self.file_name
        env = UnityEnvironment(file_name=file_name, side_channels=[channel], no_graphics=False,
                               worker_id=worker_id, seed=42)

        scores = []
        for _ in range(num_rollouts):
            env.reset()
            behavior_names = list(env.behavior_specs.keys())
            decision_steps, terminal_steps = env.get_steps(behavior_names[0])

            done = False
            reward = 0
            while not done:
                for i in decision_steps.agent_id:
                    camera_obs = np.transpose(self.img_scale(decision_steps.obs[0][i] * 255), (2, 1, 0))
                    action = self.get_action(camera_obs)
                    # print(decision_steps.obs[0][i])
                    action_tuple = ActionTuple(continuous=np.expand_dims(action, axis=0))
                    env.set_action_for_agent(behavior_names[0], i, action_tuple)

                decision_steps, terminal_steps = env.get_steps(behavior_names[0])
                reward += sum(decision_steps.reward)
                done = len(terminal_steps.interrupted) > 0
                env.step()
                # if worker_id == 2:
                #     print('step!', worker_id)
            scores.append(reward)

        env.close()
        gc.collect()
        return np.mean(scores)

    @staticmethod
    def save_params(solver, solution, model_path):
        solution.set_params(solver.best_param())
        solution.save(model_path)

    @staticmethod
    def create_logger(name, log_dir=None, debug=False):
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            os.makedirs(log_dir + '/gen_params')
        log_format = '%(asctime)s %(process)d [%(levelname)s] %(message)s'
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO, format=log_format)
        logger = logging.getLogger(name)
        if log_dir:
            log_file = os.path.join(log_dir, '{}.txt'.format(name))
            file_hdl = logging.FileHandler(log_file)
            formatter = logging.Formatter(fmt=log_format)
            file_hdl.setFormatter(formatter)
            logger.addHandler(file_hdl)
        return logger

    @staticmethod
    def img_scale(img, margin=5):
        img = cv2.resize(img, (img.shape[0] + 2 * margin, img.shape[1] + 2 * margin))
        height, width, _ = img.shape
        return img[margin:height - margin, margin:width - margin]

    @staticmethod
    def action_scale(action):
        action = np.tanh(action)
        return action

    def train(self,
              population_size: int = 256,
              max_iter: int = 20000,
              reps: int = 16,
              init_sigma: float = 0.1,
              load_model: str = None,
              log_dir: str = None,
              save_interval: int = 100,
              num_workers: int = 1,
              seed: int = 42
              ):
        self.num_workers = num_workers
        # self.env = gym.make(self.env_name)
        rnd = np.random.RandomState(seed=seed)
        num_params = self.get_num_params()
        if load_model is not None:
            init_params = self.get_params()
        else:
            init_params = None
        print('here')
        solver = CMAES(
            num_params=num_params,
            sigma_init=0.10,
            popsize=population_size,
            weight_decay=0.01
        )

        best_so_far = -float('Inf')
        ii32 = np.iinfo(np.int32)
        repeats = [reps] * population_size

        logger = self.create_logger(name='train_log', log_dir=log_dir)
        with mp.Pool(num_workers) as p:
            for n_iter in range(max_iter):
                params_set = solver.ask()
                task_seeds = [rnd.randint(0, ii32.max)] * population_size
                fitnesses = []
                ss = 0
                while ss < population_size:
                    ee = ss + min(num_workers, population_size - ss)
                    fitnesses.append(
                        p.map(func=self.get_fitness,
                              iterable=zip(
                                  params_set[ss:ee],
                                  task_seeds[ss:ee],
                                  repeats[ss:ee])))
                    ss = ee
                    # print('fitnesses:', fitnesses)
                # print('done pop')
                fitnesses = np.concatenate(fitnesses)
                solver.tell(fitnesses)

                # ESの保存
                if (n_iter + 1) % save_interval == 0:
                    pickle.dump(solver, open(log_dir + '/gen_es/es_' + '_{}.pkl'.format(n_iter + 1), 'wb'))

                logger.info(
                    'Iter={0}, '
                    'max={1:.2f}, avg={2:.2f}, min={3:.2f}, std={4:.2f}'.format(
                        n_iter, np.max(fitnesses), np.mean(fitnesses), np.min(fitnesses), np.std(fitnesses)))

                best_fitness = max(fitnesses)
                if best_fitness > best_so_far:
                    best_so_far = best_fitness
                    model_path = os.path.join(log_dir, 'best.npz')
                    self.save_params(solver=solver, solution=self, model_path=model_path)
                    logger.info('Best model updated, score={}'.format(best_fitness))
                if (n_iter + 1) % save_interval == 0:
                    model_path = os.path.join(log_dir, 'Iter_{}.npz'.format(n_iter + 1))
                    self.save_params(solver=solver, solution=self, model_path=model_path)


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
        self.file_name = file_name

        self.patch_size = patch_size
        self.image_size = aa_image_size

        self.alpha = 0.
        self.attended_patch_ix = None
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

        print('#params={}'.format(self.get_params().size))

    def _get_action(self, obs):
        x = self.vision_pi_layer(obs=obs, prev_act=self.prev_act)
        self.attended_patch_ix = (
            self.vision_pi_layer.attention.mostly_attended_entries
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
        # print('centers:', centers)

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
        action = list(map(self.action_scale, action))
        return action

    def reset(self):
        self.alpha = 0.
        self.prev_act = torch.zeros(1, self.act_dim)
        self.prev_hidden = torch.zeros(self.hidden_dim, self.msg_dim)
        self.hx = None

    def plot_attention_patches(self, img, white_patch_ix):
        attention_patch = np.ones([self.patch_size, self.patch_size, 3]) * [255, 255, 0]
        num_patches = self.image_size // self.patch_size
        for ix in white_patch_ix:
            row_ix = ix // num_patches
            col_ix = ix % num_patches
            row_ss = row_ix * self.patch_size
            col_ss = col_ix * self.patch_size
            row_ee = row_ss + self.patch_size
            col_ee = col_ss + self.patch_size
            img[row_ss:row_ee, col_ss:col_ee] = (
                0.5 * img[row_ss:row_ee, col_ss:col_ee] + 0.5 * attention_patch)
        return img.astype(np.uint8)

    def show_gui(self, obs, counter):
        if hasattr(self, 'attended_patch_ix'):
            attended_patch_ix = self.attended_patch_ix
        else:
            attended_patch_ix = None

        if attended_patch_ix is not None:
            obs = self.plot_attention_patches(
                img=obs, white_patch_ix=attended_patch_ix)

        img = cv2.resize(obs, (400, 400))[:, :, ::-1]
        cv2.imwrite('attention_gui/img_' + str(counter) + '.png', img)
        cv2.imshow('render', img)
        cv2.waitKey(1)


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

        print('#params={}'.format(self.get_num_params()))

    def _get_action(self, obs):
        x = self.pi_layer(obs=obs, prev_act=self.prev_act)
        self.prev_act = self.net(x.T)
        action = self.prev_act.squeeze(0).cpu().numpy()
        action = list(map(self.action_scale, action))
        return action

    def reset(self):
        self.prev_act = torch.zeros(1, self.act_dim)
        self.pi_layer.reset()


class MultiDrones(BaseTorchSolution):
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
                 aa_top_k=10,
                 ray_vector_dim=0):
        super(MultiDrones, self).__init__(device)
        self.file_name = file_name

        self.patch_size = patch_size
        self.image_size = aa_image_size

        self.alpha = 0.
        self.attended_patch_ix = None
        self.ray_vector_dim = ray_vector_dim
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
            input_size=aa_top_k * 2 + self.ray_vector_dim,
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

        print('#params={}'.format(self.get_params().size))

    def _get_action(self, obs):
        x = self.vision_pi_layer(obs=obs[0], prev_act=self.prev_act)
        self.attended_patch_ix = (
            self.vision_pi_layer.attention.mostly_attended_entries
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
        feature = torch.cat([centers, torch.Tensor(obs[1])])
        self.ray_vector_dim = len(obs[1])
        # print('feature:', feature)

        if self.hx is None:
            self.hx = (
                torch.zeros(1, self.lstm_hidden_dim),
                torch.zeros(1, self.lstm_hidden_dim),
            )
        self.hx = self.lstm(feature.unsqueeze(0), self.hx)
        output = self.output_fc(self.hx[0])
        self.prev_act = output

        self.alpha = self.mixing_fc(
            torch.cat([self.hx[0], self.prev_act], dim=-1).squeeze(0)
        )
        action = output.squeeze(0).cpu().numpy()
        action = list(map(self.action_scale, action))
        return action

    def reset(self):
        self.alpha = 0.
        self.prev_act = torch.zeros(1, self.act_dim)
        self.prev_hidden = torch.zeros(self.hidden_dim, self.msg_dim)
        self.hx = None

    def plot_attention_patches(self, img, attention_patch_ix):
        white_patch = np.ones([self.patch_size, self.patch_size, 3]) * [255, 255, 0]
        num_patches = self.image_size // self.patch_size
        for ix in attention_patch_ix:
            row_ix = ix // num_patches
            col_ix = ix % num_patches
            row_ss = row_ix * self.patch_size
            col_ss = col_ix * self.patch_size
            row_ee = row_ss + self.patch_size
            col_ee = col_ss + self.patch_size
            img[row_ss:row_ee, col_ss:col_ee] = (
                0.5 * img[row_ss:row_ee, col_ss:col_ee] + 0.5 * white_patch)
        return img.astype(np.uint8)

    def show_gui(self, obs, counter):
        if hasattr(self, 'attended_patch_ix'):
            attended_patch_ix = self.attended_patch_ix
        else:
            attended_patch_ix = None

        # print('len:',len(attended_patch_ix))

        if attended_patch_ix is not None:
            obs = self.plot_attention_patches(
                img=obs, attention_patch_ix=self.attended_patch_ix)

        img = cv2.resize(obs, (400, 400))[:, :, ::-1]
        cv2.imwrite('attention_gui/img_' + str(counter) + '.png', img)
        cv2.imshow('render', img)
        cv2.waitKey(1)
