from itertools import chain

import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader

from agents.ppo import PPOLoss
from rewards import Reward, Advantage
from reporters import NoReporter, Reporter, MolecularWriter
from models import ModelFactory, GCPN
from curiosity import CuriosityFactory
from agents.agent import Agent
from envs import MultiEnv

class MolPPO(Agent):
    def __init__(self, env: MultiEnv, writer: MolecularWriter, model_factory: ModelFactory, curiosity_factory: CuriosityFactory,
                 reward: Reward, advantage: Advantage, learning_rate: float, clip_range: float, v_clip_range: float,
                 c_entropy: float, c_value: float, n_mini_batches: int, n_optimization_epochs: int,
                 clip_grad_norm: float, normalize_state: bool, normalize_reward: bool, normalize_advantage: bool,
                 reporter: Reporter = NoReporter()) -> None:
        """
        :param env: environment to train on
        :param model_factory: factory to construct the model used as the brain of the agent
        :param curiosity_factory: factory to construct curiosity object
        :param reward: reward function to use for discounted reward calculation
        :param advantage: advantage function to use for advantage calculation
        :param learning_rate: learning rate
        :param clip_range: clip range for surrogate function and value clipping
        :param v_clip_range: clip range for value function clipping
        :param c_entropy: entropy coefficient constant
        :param c_value: value coefficient constant
        :param n_mini_batches: number of mini batches to devide experience into for optimization
        :param n_optimization_epochs: number of optimization epochs on same experience. This value is called ``K``
               in paper
        :param clip_grad_norm: value used to clip gradient by norm
        :param normalize_state: whether to normalize the observations or not
        :param normalize_reward: whether to normalize rewards or not
        :param reporter: reporter to be used for reporting learning statistics, defaults to NoReporter()
        """
        super().__init__(env, model_factory, curiosity_factory, normalize_state, normalize_reward, writer=writer)
        self.reward = reward
        self.advantage = advantage
        self.n_mini_batched = n_mini_batches
        self.n_optimization_epochs = n_optimization_epochs
        self.clip_grad_norm = clip_grad_norm
        self.normalize_advantage = normalize_advantage
        self.optimizer = Adam(chain(self.model.parameters(), self.curiosity.parameters()), learning_rate, eps=1e-5)
        self.loss = PPOLoss(clip_range, v_clip_range, c_entropy, c_value, reporter)
        self.scheduler = None

    def _train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        _, policy_old, values_old = self.model(self._to_tensor(self.state_converter.reshape_as_input(states,
                                                                                                  self.model.recurrent)))
        if not self.scheduler:
            # linearly decrease learning rate factor from 1 to 0 over total epochs
            self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1, end_factor=0, total_iters=self.epochs)

        policy_old = policy_old.detach().view(*states.shape[:2], -1)
        values_old = values_old.detach().view(*states.shape[:2])
        values_old_numpy = values_old.cpu().detach().numpy()
        discounted_rewards = self.reward.discounted(rewards, values_old_numpy, dones)
        advantages = self.advantage.discounted(rewards, values_old_numpy, dones)
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # # debug
        # print("policy_old: ",policy_old[:, :-1].shape)
        # print("actions: ", actions.shape)
        # # ----------------------
        # for i in range(policy_old.shape[1]-1):
        #     assert policy_old[0][i][int(actions[0][i][0].item())] > -1e5, f'{i} not ok'
        # #-----------------------
        dataset = self.model.dataset(policy_old[:, :-1], values_old[:, :-1], states[:, :-1], states[:, 1:],actions,
                                        discounted_rewards, advantages)
        loader = DataLoader(dataset, batch_size=len(dataset) // self.n_mini_batched, shuffle=True)
        # with torch.autograd.detect_anomaly():
        for _ in range(self.n_optimization_epochs):
            for tuple_of_batches in loader:
                (batch_policy_old, batch_values_old, batch_states, batch_next_states,
                    batch_actions, batch_rewards, batch_advantages) = self._tensors_to_device(*tuple_of_batches)
                _, batch_policy, batch_values = self.model(batch_states)
                batch_values = batch_values.squeeze()
                # debug
                # print("batch_policy_old.shape: ", batch_policy_old.shape)
                distribution_old = self.action_converter.distribution(batch_policy_old)
                distribution = self.action_converter.distribution(batch_policy)
                #debug
                # print("batch_policy_old: ", batch_policy_old)
                # print("batch_actions: ", batch_actions)

                loss: torch.Tensor = self.loss(distribution_old, batch_values_old, distribution, batch_values,
                                            batch_actions, batch_rewards, batch_advantages)
                loss = self.curiosity.loss(loss, batch_states, batch_next_states, batch_actions)
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()
        # Decrease learning rate factor every epoch
        self.scheduler.step()

    def act(self, state: np.ndarray) -> np.ndarray:
        # debug
        # print("state.shape: ", state.shape)   
        state = self.state_normalizer.transform(state[:, None, :])
        # debug
        # print("state.shape: ", state.shape)
        reshaped_states = self.state_converter.reshape_as_input(state, self.model.recurrent)
        # debug
        # print("reshaped_states.shape: ", reshaped_states.shape)
        action  = self.model.policy_logits(torch.tensor(reshaped_states, device=self.device))
        return action.cpu().detach().numpy()