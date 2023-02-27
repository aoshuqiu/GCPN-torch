import pytest
import sys

import torch

sys.path.append('/home/bachelor/zhangjinhang/molRL/molppo')
from molgym.homo_predictor import HomoPredictor

EPISILON = 1e-3
@pytest.fixture
def homo_predictor():
    homo_predictor = HomoPredictor.factory().create("./save_mean2.pt",torch.device('cuda:0'))
    return homo_predictor

def test_homo_predictor(homo_predictor):
    assert abs(homo_predictor.predict("./PC1=CC=C2C3=C(C=C(Br)N=C3)C3=C(P)C=CC4=C5C=NC(Br)=CC5=C1C2=C43.xyz")+4.974857330322266) < EPISILON

