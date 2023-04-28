from envs import MultiEnv
from typing import Union, List

import numpy as np
import torch

from models import ModelFactory
from curiosity import CuriosityFactory
from reporters import NoReporter, Reporter
from envs import Converter, RandomRunner, Runner
from normalizers import StandardNormalizer, NoNormalizer

class Agent:
    """
    Base interface for agents
    """

    def __init__(self, env: MultiEnv, model_factory: ModelFactory, curiosity_factory: CuriosityFactory,
                 normalize_state: bool, normalize_reward:bool, reporter: Reporter = NoReporter(), writer = None) -> None:
        self.env = env
        self.reporter = reporter
        self.state_converter = Converter.for_space(self.env.observation_space)
        self.action_converter = Converter.for_space(self.env.action_space)
        self.model = model_factory.create(self.state_converter, self.action_converter)
        self.curiosity = curiosity_factory.create(self.state_converter, self.action_converter)
        self.reward_normalizer = StandardNormalizer() if normalize_reward else NoNormalizer()
        self.state_normalizer = self.state_converter.state_normalizer() if normalize_state else NoNormalizer()
        self.normalize_state = normalize_state
        self.device: torch.device = None
        self.dtype: torch.dtype = None
        self.numpy_dtype: object = None
        self.writer = writer

    def act(self, state: np.ndarray) -> np.ndarray:
        """
        Acts in the envrionment. Returns the action for the given state

        Note: ``N`` in the dimensions stands for number of parallel envrionments being explored

        :param state: state of shape N * (state space shape) that we want to know the action for 
        :return: the action which is array of shape N * (action space shape)
        """     
        # add a new axis for normalizer require two dimensions for N and T.
        state = self.state_normalizer.transform(state[:, None, :]) 
        reshaped_states = self.state_converter.reshape_as_input(state, self.model.recurrent)
        logits = self.model.policy_logits(torch.tensor(reshaped_states, device=self.device))
        return self.action_converter.action(logits).cpu().detach().numpy()

    def _train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        """
        Trains the agent using previous experience.
        Legend for the dimensions of input arrays:
         * ``N`` - number of parallel envrionments being explored(see ``MultiEnv``)
         * ``T`` - number of time steps that were run on the environment

        :param states: of shape N * T * (state space shape)
        :param actions: of shape N * T * (action space shape)
        :param rewards: of shape N * T * 1
        :param dones:  of shape N * T * 1
        """
        raise NotImplementedError('Implement me')

    def learn(self, epochs: int, n_steps: int, initialization_steps: int = 1000, render: bool = False):
        """
        Trains the agent for ``epochs`` number of times by running simulation on the environment for ``n_steps``

        :param epochs: number of epochs of training
        :param n_steps: number of steps made in the environment each epoch
        :param initialization_steps: number of steps made on the environment to gather the states then used for
               initialization of the state normalizer, defaults to 1000
        :param render: whether to render the environment during learning, defaults to False
        """
        self.epochs = epochs
        if initialization_steps and self.normalize_state:
            s, _, _, _ = RandomRunner(self.env).run(initialization_steps)
            self.state_normalizer.partial_fit(s)
        for epoch in range(epochs):
            states, actions, rewards, dones = Runner(self.env, self, writer=self.writer).run(n_steps, render)
            # debug:
            # print("action.dtype: ", actions.dtype)
            # input()
            # print("actions.shape: ", actions.shape)
            # print("states.shape: ", states.shape)
            states = self.state_normalizer.partial_fit_transform(states)
            rewards = self.curiosity.reward(rewards, states, actions)
            rewards = self.reward_normalizer.partial_fit_transform(rewards)
            self._train(states, actions, rewards, dones)
            print(f'Epoch: {epoch} done')

    def eval(self, n_steps: int, render: bool = False):
        Runner(self.env, self).run(n_steps, render)

    def to(self, device: torch.device, dtype: torch.dtype, numpy_dtype: Union[object, str]) -> None:
        """
        Transfers the agent's model to device 
        :param device: device to transfer agent to
        :param dtype: dtype to which cast the model parameters
        :param numpy_dtype:  dtype to use for the environment. *Must* be the same as ``dtype`` parameter
        """      
        self.device = device
        self.dtype = dtype
        self.numpy_dtype = numpy_dtype
        self.model.to(device, dtype)
        # self.model = self.model.cuda()
        self.curiosity.to(device, dtype)
        self.env.astype(numpy_dtype)

        

    def _tensors_to_device(self, *tensors: torch.Tensor) -> List[torch.Tensor]:
        return [tensor.to(self.device, self.dtype) for tensor in tensors]

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, device=self.device, dtype=self.dtype)
