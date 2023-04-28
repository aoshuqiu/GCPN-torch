from itertools import chain


import numpy as np
import torch
from torch.nn.modules.loss import _Loss
from torch.distributions import Distribution
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from reporters import Reporter, NoReporter
from agents import Agent
from envs import MultiEnv
from models import ModelFactory
from curiosity import CuriosityFactory
from rewards import Reward, Advantage

class PPOLoss(_Loss):
    r"""
    Calculates the PPo loss given by equation:
    .. math:: L_t^{CLIP+VF+S}(\theta) = \mathbb{E} \left [L_t^{CLIP}(\theta) - c_v * L_t^{VF}(\theta)+ c_e * S[\pi_\theta](s_t) \right ]
    
    where:

    .. math:: L_t^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left [\text{min}(r_t(\theta)\hat{A}_t,\text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)\hat{A}_t )\right ]

    .. math:: r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}

    .. math:: L_t^{VF}(\theta) = (V_\theta(s_t) - V_t^{targ})^2

    and :math:`S[\pi_\theta](s_t)` is an entropy

    """


    def __init__(self, clip_range: float, v_clip_range: float, c_entropy: float, c_value: float, reporter: Reporter):
        """
        
        :param clip_range: clip range for surrogate function clippting 
        :param v_clip_range: clip range for value function clipping
        :param c_entropy: entropy coefficient constant
        :param c_value: value coefficient constant
        :param reporter: reporter to be used to report loss scalars
        """

        super().__init__()
        self.clip_range = clip_range
        self.v_clip_range = v_clip_range
        self.c_entropy = c_entropy
        self.c_value = c_value
        self.reporter = reporter

    def forward(self, distribution_old: Distribution, value_old: Tensor,  distribution: Distribution,
                value: Tensor, action: Tensor, reward: Tensor, advantage: Tensor):
        # Value loss
        value_old_clipped = value_old + (value - value_old).clamp(-self.v_clip_range, self.v_clip_range)
        v_old_loss_clipped = (reward - value_old_clipped).pow(2)
        v_loss = (reward - value).pow(2)
        value_loss = torch.min(v_old_loss_clipped, v_loss).mean()

        self.reporter.scalar('ppo_loss/adavantageb4_max', advantage.max().item())
        # Policy loss
        advantage = (advantage - advantage.mean()) /  (advantage.std(unbiased=False) + 1e-8)
        advantage.detach_()
        log_prob = distribution.log_prob(action)

        # dubug
        # print("distribution_old.logits: ", distribution_old.logits)
        # print("action: ", action)
        log_prob_old = distribution_old.log_prob(action)
        log_prob_diff = torch.clamp(log_prob - log_prob_old, max=10)
        ratio = torch.clamp(log_prob_diff.exp().view(-1), min=-5, max=5)

        surrogate = advantage * ratio
        surrogate_clipped = advantage * ratio.clamp(1 - self.clip_range, 1 + self.clip_range)

        policy_loss = torch.min(surrogate, surrogate_clipped).mean()
        if policy_loss < -100:
            #debug:
            print("policy_loss:", policy_loss)
            print("surrogate.shape: ", surrogate.shape)
            print("surrogate_clipped.shape: ", surrogate_clipped.shape)
            print("surrogate:", surrogate)
            print("surrogate_clipped.shape:", surrogate_clipped)
            print("advantage:", advantage)
            print("ratio:", ratio)
            print("torch.min(surrogate, surrogate_clipped):", torch.min(surrogate, surrogate_clipped))
            input()
        # Entropy
        entropy = distribution.entropy().mean()

        # Total loss
        losses = policy_loss + self.c_entropy * entropy - self.c_value * value_loss
        total_loss = -losses
        self.reporter.scalar('ppo_loss/surrogate', surrogate.mean().item())
        self.reporter.scalar('ppo_loss/surrogate_clipped', surrogate_clipped.mean().item())
        self.reporter.scalar('ppo_loss/adavantage', advantage.mean().item())
        self.reporter.scalar('ppo_loss/ratio', ratio.mean().item())
        self.reporter.scalar('ppo_loss/policy', -policy_loss.item())
        self.reporter.scalar('ppo_loss/entropy', -entropy.item())
        self.reporter.scalar('ppo_loss/value_loss', value_loss.item())   
        self.reporter.scalar('ppo_loss/total', total_loss)
        # return total_loss
        return total_loss, policy_loss                 

class PPO(Agent):
    """
    Implementation of PPO algorithm described in paper: https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(self, env: MultiEnv, model_factory: ModelFactory, curiosity_factory: CuriosityFactory,
                 reward: Reward, advantage: Advantage, learning_rate: float, clip_range: float, v_clip_range: float,
                 c_entropy: float, c_value: float, n_mini_batches: int, n_optimization_epochs: int,
                 clip_grad_norm: float, normalize_state: bool, normalize_reward: bool,
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
        super().__init__(env, model_factory, curiosity_factory, normalize_state, normalize_reward)
        self.reward = reward
        self.advantage = advantage
        self.n_mini_batched = n_mini_batches
        self.n_optimization_epochs = n_optimization_epochs
        self.clip_grad_norm = clip_grad_norm
        self.optimizer = Adam(chain(self.model.parameters(), self.curiosity.parameters()), learning_rate)
        self.loss = PPOLoss(clip_range, v_clip_range, c_entropy, c_value, reporter)

    def _train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        policy_old, values_old = self.model(self._to_tensor(self.state_converter.reshape_as_input(states,
                                                                                                  self.model.recurrent)))
        policy_old = policy_old.detach().view(*states.shape[:2], -1)
        values_old = values_old.detach().view(*states.shape[:2])
        values_old_numpy = values_old.cpu().detach().numpy()
        discounted_rewards = self.reward.discounted(rewards, values_old_numpy, dones)
        advantages = self.advantage.discounted(rewards, values_old_numpy, dones)

        
        dataset = self.model.dataset(policy_old[:, :-1], values_old[:, :-1], states[:, :-1], states[:, 1:],actions,
                                        discounted_rewards, advantages)
        loader = DataLoader(dataset, batch_size=len(dataset) // self.n_mini_batched, shuffle=True)
        with torch.autograd.detect_anomaly():
            for _ in range(self.n_optimization_epochs):
                for tuple_of_batches in loader:
                    (batch_policy_old, batch_values_old, batch_states, batch_next_states,
                     batch_actions, batch_rewards, batch_advantages) = self._tensors_to_device(*tuple_of_batches)
                    batch_policy, batch_values = self.model(batch_states)
                    batch_values = batch_values.squeeze()
                    distribution_old = self.action_converter.distribution(batch_policy_old)
                    distribution = self.action_converter.distribution(batch_policy)
                    loss: Tensor = self.loss(distribution_old, batch_values_old, distribution, batch_values,
                                             batch_actions, batch_rewards, batch_advantages)
                    loss = self.curiosity.loss(loss, batch_states, batch_next_states, batch_actions)
                    # print('loss:', loss)
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()


