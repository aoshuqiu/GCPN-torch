from typing import List

import torch
from rdkit.Chem.Descriptors import qed, MolLogP
from rdkit.Chem import rdMolDescriptors

from molgym.envs.predictor import Predictor
from molgym.envs.homo_predictor import HomoPredictor
from molgym.envs.utils import reward_penalized_log_p, func_combine
from molgym.envs.utils import sa_reward, reward_target_new
from molgym.envs.utils import reward_target_qed, reward_valid

class CriticMap:
    def __init__(self, device,reward_target=0):
        """
        :param reward_target: target reward for target optimization, defaults to 0
        """

        self.map = {}
        homo_critic = HomoPredictor.factory().create("./save_mean2.pt",device)
        logppen_critic = func_combine([reward_penalized_log_p],[1/3])
        qed_critic = func_combine([qed],[2])
        qedsa_critic = func_combine([qed, sa_reward],[1.5, 0.5])
        sa_critic = sa_reward
        logp_target_critic = lambda mol:reward_target_new(mol, MolLogP, x_start=reward_target,
                                                                 x_mid = reward_target + 0.25)
        qed_target_critic = lambda mol: reward_target_qed(mol,target=reward_target)
        mw_target_critic = lambda mol: reward_target_new(mol, rdMolDescriptors.CalcExactMolWt,
                                                         x_start=self.reward_target, x_mid=self.reward_target+25)
        valid_critic = reward_valid

        self.map['homo'] = homo_critic
        self.map['logppen'] = logppen_critic
        self.map['qed'] = qed_critic
        self.map['qedsa'] = qedsa_critic
        self.map['sa'] = sa_critic
        self.map['logp_target'] = logp_target_critic
        self.map['qed_target'] = qed_target_critic
        self.map['mw_target'] = mw_target_critic
        self.map['valid'] = valid_critic

    def set_combine_func(self, funcs:List[str], weights:List[float]):
        """
        Use function strs and weights to get a main critic func

        :param funcs: List of function strs, like qed, sa, valid etc.
        :param weights: List of weights.
        :return: a combination of all the selected functions.
        """
        funclist = [self.map.get(func) for func in funcs]
        return func_combine(funclist, weights)

