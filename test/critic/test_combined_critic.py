import pytest
import sys

import torch
from rdkit import Chem
from rdkit.Chem.Descriptors import qed

sys.path.append('/home/bachelor/zhangjinhang/molRL/molppo')
from molgym.envs.critic import CriticMap

EPISILON = 1e-3
@pytest.fixture
def combined_func():
    combined_func = CriticMap().set_combine_func(["qed", "valid"],[1, 1])
    return combined_func

def test_combined_func(combined_func):
    mol = Chem.MolFromSmiles("C1=CC=CC=C1")
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)

    assert abs(combined_func(mol)-qed(mol)*2-2) < EPISILON

