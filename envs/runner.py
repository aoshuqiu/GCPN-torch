from typing import Tuple, TYPE_CHECKING

import numpy as np

from envs.multi_env import MultiEnv
from reporters import NoWriter

if TYPE_CHECKING:
    from agents.agent import Agent


class Runner:
    """
    Runs the simulation on the environments using specified agent for choosing the actions
    """

    def __init__(self, env: MultiEnv, agent: 'Agent', writer = NoWriter()) -> None:
        """
        :param env: environment to be used for the simulation
        :param agent: agent to be used to act on the envrionment
        """
        self.env = env
        self.agent = agent
        self.writer = writer
    def run(self, n_steps: int, render: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs the simulation for specified number of steps and aggregates all the observations made on the environment

        :param n_steps: number of steps to run the simulation for
        :param render: whether to render the environment during simulation or not
        :return: returns tuple of(``T`` stands for time steps and usually is equal to ``n_steps``,
                 ``N`` number of environments in ``MultiEnv``):

                 \* ``states`` - of shape N*T*(state space shape) Note ``T`` here is equal to ``n_steps + 1``.

                   This shape allows us to construct ``prev_states`` and ``next_states`` easily later by doing:

                   .. code-block:: python

                   prev_states = states[:-1]
                   next_states = states[1:]

                 \* ``actions`` - of shape N*T*(action space shape)

                 \* ``rewards`` - of shape N*T
                 
                 \* ``dones`` - of shape N*T
        """
        state = self.env.reset()
        # debug
        # print("origin state.shape: ", state.shape)
        states = np.empty(self.get_mini_batch_shape(state.shape, n_steps + 1),
                          dtype=self.env.dtype) # +1 for initial state
        rewards = np.empty(self.get_mini_batch_shape((self.env.n_envs, ), n_steps), dtype=self.env.dtype)
        dones = np.empty(self.get_mini_batch_shape((self.env.n_envs, ), n_steps), dtype=self.env.dtype)
        actions = None
        states[:, 0] = state # inital state
        infos = []
        for step in range(n_steps):
            if render:
                self.env.render()
            action = self.agent.act(state)
            if step == 0: # lazy init when we know the action space shape
                actions = np.empty(self.get_mini_batch_shape(action.shape, n_steps), dtype=self.env.dtype)
            state, reward, done, info = self.env.step(action)
            states[:, step+1] = state
            actions[:, step] = action
            rewards[:, step] = reward
            dones[:, step] = done
            infos.append(info)
        reshape_infos = [] # reshape infos 
        for i in range(len(infos[0])):
            reshape_infos.extend(list(list(zip(*infos))[i]))
        self.writer.write(dones, reshape_infos, rewards)
        return states, actions, rewards, dones

    def get_mini_batch_shape(self, observation_shape, n_steps):
        return (self.env.n_envs, n_steps, *observation_shape[1:])

class RandomRunner(Runner):
    def __init__(self, env: MultiEnv):
        from agents import RandomAgent
        super().__init__(env, RandomAgent(env.action_space))