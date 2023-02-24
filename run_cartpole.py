import numpy as np
import torch
import gym

from rewards import GeneralizedRewardEstimation, GeneralizedAdvantageEstimation
from reporters import TensorBoardReporter
from agents import PPO
from envs import MultiEnv
from models import MLP
from curiosity import NoCuriosity 

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reporter = TensorBoardReporter()
    env = MultiEnv('CartPole-v1', 4, reporter)
    agent = PPO(env,
                reporter=reporter,
                normalize_state=False,
                normalize_reward=False,
                model_factory=MLP.factory(),
                curiosity_factory=NoCuriosity.factory(),
                reward=GeneralizedRewardEstimation(gamma=0.99, lam=0.95),
                advantage = GeneralizedAdvantageEstimation(gamma=0.99, lam=0.95),
                learning_rate=5e-3,
                clip_range=0.2,
                v_clip_range=0.3,
                c_entropy=1e-2,
                c_value=0.5,
                n_mini_batches=4,
                n_optimization_epochs=5,
                clip_grad_norm=0.5)
    agent.to(device, torch.float32, np.float32)

    agent.learn(epochs=1000, n_steps=500)

    env = gym.wrappers.RecordVideo(env, 'video', episode_trigger=lambda x:x%100==0)
    agent.env = env
    agent.eval(n_steps=500, render=True)
    env.close()