import os

import torch
import numpy as np
from rdkit import RDLogger

from reporters import TensorBoardReporter, MolecularWriter
from agents import MolPPO
from envs import MultiEnv
from models import GCPN
from curiosity import NoCuriosity, ICM, MlpICMModel
from rewards import GeneralizedAdvantageEstimation, GeneralizedRewardEstimation


if __name__ == '__main__':
    torch.set_num_threads(3)
    torch.multiprocessing.set_start_method('spawn')
    device = torch.device('cuda:2')
    reporter = TensorBoardReporter() 
    if not os.path.exists("molecule_gen"):
        os.makedirs("./molecule_gen")
    writer = MolecularWriter('molecule_gen/molcule.csv')
    writer.reporter = reporter
    RDLogger.DisableLog('rdApp.*')
    molenv_context = {
        "data_type":'zinc',
        "logp_ratio": 1, 
        "qed_ratio":1,
        "sa_ratio":1,
        "reward_step_total":1,
        "is_normalize":0,
        "reward_type":'qed',
        "reward_target":0.5,
        "has_scaffold":False,
        "has_feature":False,
        "is_conditional":False,
        "conditional":'low',
        "max_action":128,
        "min_action":20,
        "force_final":False,
        "device": device
    }
    agent = MolPPO(MultiEnv('molecule-v0', 3, reporter, molenv_context),
                   writer= writer,
                   reporter=reporter,
                   normalize_state=False,
                   normalize_reward=True,
                   model_factory=GCPN.factory(),
                #    curiosity_factory=NoCuriosity.factory(),
                   curiosity_factory=ICM.factory(MlpICMModel.factory(), policy_weight=1, reward_scale=0.01, weight=0.2,
                            intrinsic_reward_integration=0.01, reporter=reporter),
                   reward=GeneralizedRewardEstimation(gamma=1,lam=0.95),
                   advantage=GeneralizedAdvantageEstimation(gamma=1, lam=0.95),
                   learning_rate=5e-4,
                   clip_range=0.2,
                   v_clip_range=0.2,
                #    c_entropy=1e-2,
                   c_entropy=0.05,
                   c_value=0.5,
                   n_mini_batches=32,
                   n_optimization_epochs=8,
                   clip_grad_norm=0.5,
                   normalize_advantage=True,
                  )
    agent.to(device, torch.float32, np.float32)
    agent.learn(epochs=3000, n_steps=256)