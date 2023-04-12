import torch
from torch import nn

from models.model import Model
from models.gcn import GCN
from models.gcpn import GCPN
from envs import Converter

class FragmentGCPN(GCPN):

    def __init__(self, state_space: Converter, action_space: Converter, out_channels=128,
                 in_channels=9, edge_type=3, atom_type_num=9, stop_shift=-3, max_atom=65, vocab=None) -> None:
        super().__init__(state_space, action_space)
        self.stop_shift = stop_shift
        self.atom_type_num = atom_type_num
        self.in_channels = in_channels
        self.max_atomn = max_atom
        self.edge_type = edge_type
        self.emb = nn.Linear(in_channels, 8)
        self.gcn1 = GCN(8, out_channels, 3)
        self.gcn2 = GCN(out_channels, out_channels, 3)
        self.gcn3 = GCN(out_channels, out_channels, 3, need_normalize=True)
        self.value_input = nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=False),
            nn.Tanh()
        )
        self.value_out = nn.Linear(out_channels, 1)
        self.policy_stop_input= nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=False),
            nn.Tanh()
        )
        self.vocab = vocab
        self.policy_stop_out = nn.Linear(out_channels, 2)
        vocab_size = self.vocab.size()

        self.policy_motif_input = nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=False),
            nn.Tanh()
        )
        self.policy_motif_out = nn.Linear(out_channels, vocab_size)

        self.policy_first = nn.Sequential(
            nn.Linear(2*out_channels, out_channels),
            nn.Tanh(),
            nn.Linear(out_channels, 1)
        )
        self.policy_second = nn.Sequential(
            nn.Linear(3*out_channels, out_channels),
            nn.Tanh(),
            nn.Linear(out_channels, 1)
        )
        self.policy_edge = nn.Sequential(
            nn.Linear(3*out_channels, out_channels),
            nn.Tanh(),
            nn.Linear(out_channels, edge_type+1)
        )

    def forward(self, state):
        tanh = nn.Tanh()
        state = state.reshape((-1, self.edge_type, self.max_atomn, self.max_atomn+self.in_channels))
        device = state.device
        state = self.tensor_to_dict(state)






