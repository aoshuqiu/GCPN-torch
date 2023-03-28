import pytest
import sys

import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem.Descriptors import qed
import gym
import numpy as np

sys.path.append('/home/bachelor/zhangjinhang/molRL/molppo')
from models.gcpn import GCPN
from envs.converters import Converter,DiscreteConverter
from molgym import dict_to_np, np_to_dict

EPISILON = 1e-3
@pytest.fixture
def mol_env():
    mol_env = gym.make("molecule-v0")
    mol_env.set_hyperparams()
    return mol_env

@pytest.fixture
def obadj():
    obadj = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
    return obadj

@pytest.fixture
def obnode():
    obnode = np.array([[[3,4,5],[6,7,8]]])
    return obnode

@pytest.fixture
def obnp():
    obnp = np.array([[[1,2,3,4,5],
                        [3,4,6,7,8]],
                        [[5,6,3,4,5],
                        [7,8,6,7,8]]])
    return obnp

def test_dict_to_np(mol_env, obadj, obnode):
    ob = {}
    ob["adj"] = obadj
    ob["node"] = obnode
    npob = dict_to_np(ob)
    print(npob)
    np.testing.assert_allclose(npob,torch.Tensor([[[1,2,3,4,5],
                                                   [3,4,6,7,8]],
                                                  [[5,6,3,4,5],
                                                   [7,8,6,7,8]]]))
def test_np_to_dict(mol_env,obadj, obnode, obnp):
    ob = np_to_dict(obnp)
    print(ob)
    np.testing.assert_allclose(ob["adj"],obadj)
    np.testing.assert_allclose(ob["node"],obnode)

@pytest.mark.parametrize("smiles",
                         ["C1=CC=CC=C1","C"])
def test_ob_len(mol_env, smiles):
    mol_env.reset(smile="C1=CC=CC=C1")
    state = mol_env.get_observation()
    a = dict_to_np(state)
    dict = np_to_dict(a)
    np.testing.assert_allclose(dict["adj"],state["adj"])
    np.testing.assert_allclose(dict["node"],state["node"])