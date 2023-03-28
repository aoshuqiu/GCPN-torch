import pytest
import sys

import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem.Descriptors import qed
import gym
import molgym

sys.path.append('/home/bachelor/zhangjinhang/molRL/molppo')
from models.gcpn import GCPN
from envs.converters import Converter,DiscreteConverter

EPISILON = 1e-3
@pytest.fixture
def mol_env():
    mol_env = gym.make("molecule-v0")
    mol_env.set_hyperparams()
    return mol_env

@pytest.fixture
def gcpn():
    a =DiscreteConverter(gym.spaces.Discrete(2))
    b =DiscreteConverter(gym.spaces.Discrete(2))
    gcpn = GCPN(a, b)
    return gcpn

@pytest.fixture
def emb_node():
    a =DiscreteConverter(gym.spaces.Discrete(2))
    b =DiscreteConverter(gym.spaces.Discrete(2))
    gcpn = GCPN(a, b)
    mol_env = gym.make("molecule-v0")
    mol_env.set_hyperparams()
    tanh = nn.Tanh()
    mol_env.reset(smile='C1=CC=CC=C1')
    state = mol_env.get_observation()
    node = torch.Tensor(state['node'])
    adj = torch.Tensor(state['adj'])
    if adj.dim()==3:
        adj = adj.unsqueeze(0)
    if node.dim()==3:
        node = node.unsqueeze(0)
    ob_node = gcpn.emb(node)
    emb_node = tanh(gcpn.gcn1(torch.matmul(adj,ob_node)))
    emb_node = tanh(gcpn.gcn2(torch.matmul(adj,emb_node)))
    emb_node = gcpn.gcn3(torch.matmul(adj,emb_node))
    emb_node = emb_node.squeeze(1)
    return emb_node

    
def test_ob_len(mol_env):
    mol_env.reset(smile="C1=CC=CC=C1")
    state = mol_env.get_observation()
    state = torch.Tensor(state['node'])
    assert GCPN.get_ob_len(state)-9 == 6

@pytest.mark.parametrize("atom_num,real_num",
                         [(1,1),(2,2),(3,3),(4,4),(5,5),(6,6)])
def test_mask_emb_len(emb_node, atom_num, real_num):
    GCPN.mask_emb_len_(emb_node, torch.Tensor([atom_num]), 0)
    assert GCPN.get_ob_len(emb_node)==real_num

def test_mask_logits_len():
    logits = torch.Tensor([1,3,2,1])
    mask_len = 3
    GCPN.mask_logits_len_(logits,mask_len)
    torch.testing.assert_allclose(logits,torch.Tensor([1,3,2,-1e10]))

