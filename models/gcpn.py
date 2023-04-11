from abc import abstractmethod

import torch
from torch import Tensor
from torch import nn
import numpy as np
from torch.utils.data import Dataset
import torch.distributions as D

from envs import Converter
from models.model import Model, ModelFactory
from models.gcn import GCN
from models.datasets import NonSequentialDataset
class GCPN(Model):

    def __init__(self, state_space: Converter, action_space: Converter, out_channels=128,
                 in_channels=9, edge_type=3, atom_type_num=9, stop_shift=-3, max_atomn=47):
        super().__init__(state_space, action_space)
        # debug
        # print('GCPN')
        # TODO no bn current whether to add batch normalize.
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
        self.policy_stop_out = nn.Linear(out_channels, 2)
        self.policy_first_out = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Tanh(),
            nn.Linear(out_channels, 1)
        )
        self.policy_second_out = nn.Sequential(
            nn.Linear(2*out_channels, out_channels),
            nn.Tanh(),
            nn.Linear(out_channels, 1)
        )
        self.policy_edge = nn.Sequential(
            nn.Linear(2*out_channels, out_channels),
            nn.Tanh(),
            nn.Linear(out_channels, edge_type)
        )
        self.in_channels = in_channels
        self.atom_type_num = atom_type_num
        self.stop_shift = stop_shift
        self.max_atomn = max_atomn
        self.edge_type = edge_type

    def forward(self, state):
        # print("state_shape: ",state.shape)
        tanh = nn.Tanh()
        state = state.reshape((-1,self.edge_type,self.max_atomn,self.max_atomn+self.in_channels))
        device = state.device
        state = self.tensor_to_dict(state)
        # print("state['node'].shape", state['node'].shape)
        adj = state['adj']
        node = state['node']
        # Turn single observation to batch
        if adj.dim()==3:
            adj = adj.unsqueeze(0)
        if node.dim()==3:
            node = node.unsqueeze(0)
        # One dense Layer
        # print("node.shape1", node.shape)
        ob_node = self.emb(node)
        # Three gcn Layers
        # print("ob_node.shape1", ob_node.shape)
        # print("adj.shape1", adj.shape)
        emb_node = tanh(self.gcn1(torch.matmul(adj,ob_node)))
        emb_node = tanh(self.gcn2(torch.matmul(adj,emb_node)))
        emb_node = self.gcn3(torch.matmul(adj,emb_node))
        emb_node = emb_node.squeeze(1) # emb_node: (B,n,f)

        ob_len = GCPN.get_ob_len(node)
        # print("ob_len.shape", ob_len.shape)
        ob_len_first = ob_len - self.atom_type_num
        self.mask_emb_len_(emb_node, ob_len)

        # Autoregressive Action Space
        # Fisrt get logits then construct distribution eventually sample action.
        emb_stop = self.policy_stop_input(emb_node)
        policy_stop = torch.sum(emb_stop,1) #(B,1,f)
        policy_stop = self.policy_stop_out(policy_stop) #(B,1,2)

        stop_shift = torch.tensor([0, self.stop_shift], device=device)
        pd_stop = D.Categorical(logits=policy_stop + stop_shift)
        ac_stop = pd_stop.sample() #(B,1,1)
        ac_stop = ac_stop.unsqueeze(-1) #(B, 1)

        policy_first = self.policy_first_out(emb_node).squeeze(-1) # policy_first:(B,n)
        GCPN.mask_logits_len_(policy_first, ob_len_first)
        pd_first = D.Categorical(logits=policy_first)
        ac_first = pd_first.sample()
        ac_first = ac_first.unsqueeze(-1) #(B,1)
        emb_first = GCPN.filter_first_node(emb_node, ac_first)

        emb_first_cat = torch.cat((emb_first.expand(emb_node.shape),emb_node),-1) #(B,n,2f)
        policy_second = self.policy_second_out(emb_first_cat).squeeze(-1)
        GCPN.mask_logits_len_(policy_second,ob_len)
        policy_second_maskfirst = policy_second.clone()
        GCPN.mask_logits_first_(policy_second_maskfirst, ac_first)
        pd_second = D.Categorical(logits=policy_second_maskfirst)
        ac_second = pd_second.sample()
        ac_second = ac_second.unsqueeze(-1)
        emb_second = GCPN.filter_first_node(emb_node, ac_second)

        emb_first = emb_first.squeeze(1)
        emb_second = emb_second.squeeze(1)
        # debug:
        # print('emb_second',emb_second.shape)
        # print('emb_first',emb_first.shape)
        emb_second_cat = torch.cat((emb_first, emb_second), -1)
        policy_edge = self.policy_edge(emb_second_cat)
        pd_edge = D.Categorical(logits=policy_edge)
        ac_edge = pd_edge.sample()
        ac_edge = ac_edge.unsqueeze(-1)

        value = self.value_input(emb_node)
        value = torch.max(value,1).values
        value = self.value_out(value)

        ac = torch.cat((ac_first,ac_second,ac_edge,ac_stop),-1)
        # debug:
        # print('ac.dtype: ',ac.dtype)
        # input()

        policy = {"first":policy_first,
                  "second":policy_second,
                  "edge":policy_edge,
                  "stop":policy_stop}
        policy = self._arrange_policy(policy)
        assert policy.shape[-1] == 2+3+2*self.max_atomn-self.atom_type_num
        # # debug:
        # # print("ac: ", ac.shape)
        # # print("policy: ", policy.shape)
        # # input()
        # # ----------------------
        # for i in range(policy.shape[0]):
        #     assert policy[i][int(ac[i][0].item())] > -1e5, f'policy: {policy[i]}, action: {int(ac[i][0].item())}'
        # for i in range(policy.shape[0]):
        #     ob_len = GCPN.get_ob_len(torch.tensor(state["node"][i], device=policy.device))
        #     for j in range(0, ob_len-9):
        #         assert policy[i][j]>-1e5 , f'GCPN error in state{i} not ok state: {state["node"][i] } policy: {policy[i]} j: {j}'
        # #-----------------------
        return ac, policy, value
        
    @staticmethod
    def filter_first_node(emb_node, ac_first):
        seq_range = torch.arange(0, emb_node.shape[-2], device=emb_node.device)
        mask = seq_range.unsqueeze(-1).expand(emb_node.shape) == ac_first.squeeze(0).unsqueeze(-1).expand(emb_node.shape)
        emb_first = torch.masked_select(emb_node, mask)
        emb_first = emb_first.reshape((emb_node.shape[0],1,emb_node.shape[-1])) 
        return emb_first
    
    @staticmethod
    def get_ob_len(node):
        """
        Get Atom Num of node features in observation. 

        :param node: Observation['node'].
        :return: Atom num of molecule observation.
        """
        # debug
        ob_len = torch.sum((torch.sum(node,-1)!=0), -1)
        # print("ob_len: ", ob_len)
        return ob_len
    
    @staticmethod
    def mask_emb_len_(emb_node, mask_len, fill=0):
        node_len = emb_node.shape[-2]
        v_size = mask_len.tile((1, node_len))
        seq_range = torch.arange(0, node_len, device=emb_node.device).tile(v_size.shape[0],1)
        mask = seq_range>=v_size
        mask = mask.unsqueeze(-1).expand(emb_node.shape)
        emb_node.masked_fill_(mask, fill)

    @staticmethod
    def mask_logits_len_(logits, mask_len, fill=-1e5):
        seq_range = torch.arange(0, logits.shape[-1], device=logits.device)
        # TODO: test .clone .detach
        if not isinstance(mask_len, Tensor):
            mask_len = torch.tensor(mask_len, device=logits.device)
        else:
            mask_len = mask_len.clone().detach()
        mask = seq_range.expand(logits.shape)>=mask_len.expand(logits.shape)
        # debug:
        # print("mask: ", mask)
        # print("logits: ", logits)
        # print("mask_len: ", mask_len)
        # print("seq_range.expand(logits.shape)", seq_range.expand(logits.shape))
        # print("torch.tensor(mask_len,device=logits.device).expand(logits.shape)",torch.tensor(mask_len,device=logits.device).expand(logits.shape))
        logits.masked_fill_(mask, fill)

    @staticmethod
    def mask_logits_first_(logits, ac_first, fill=-1e5):
        seq_range = torch.arange(0, logits.shape[-1],device=logits.device)
        mask = ac_first.expand(logits.shape) == seq_range.unsqueeze(0).expand(logits.shape)
        # debug:
        # print(logits.shape)
        # print("mask: ", mask)
        # print("logits: ", logits)
        # print("ac_first.expand(logits.shape): ", ac_first.expand(logits.shape))
        # print("seq_range.unsqueeze(0).expand(logits.shape)", seq_range.unsqueeze(0).expand(logits.shape))
        logits.masked_fill_(mask, fill)
        # print("logits after: ", logits)

    @property
    def recurrent(self) -> bool:
        """
        :return: whether the model is recurrent or not
        """
        return False
    
    def _arrange_policy(self, policy, fill=-1e5):
        first_pad = nn.ConstantPad1d((0,self.max_atomn-self.atom_type_num-policy["first"].shape[-1]), fill)
        second_pad = nn.ConstantPad1d((0, self.max_atomn-policy["first"].shape[-1]), fill)
        policy["first"] = first_pad(policy["first"])
        policy["second"] = second_pad(policy["second"])
        # debug
        # print("policy['first']:",policy["first"].shape)
        # print("policy['second']:",policy["second"].shape)
        # print("policy['edge']:",policy["edge"].shape)
        # print("policy['stop']:",policy["stop"].shape)
        return torch.cat((policy["first"],policy["second"],policy["edge"],policy["stop"]), dim=-1)
        
    def value(self, states: Tensor) -> Tensor:
        _,_,value = self(states)
        return value

    def policy_logits(self, states: Tensor):
        action,_,_ = self(states)
        # debug
        # print("action:",action)
        # print(action)
        return action
    
    def dataset(self, *array: np.ndarray) -> Dataset:
        return NonSequentialDataset(*array)

    @staticmethod
    def factory():
        return GCPNFactory()
    

    def np_to_dict(self, obarray):
        """
        Turn numpy observation to origin dict.

        :param obarray: numpy: d_e * N * (N + F)
        :return: Molecule observation. Contains:
                'adj': d_e * N * N --- d_e for edge type num. 
                                        N for max atom num.
                'node': 1 * N * F --- F for atom features num.
        """
        nodenum = obarray.shape[1]
        oblist = np.split(obarray,[nodenum],-1)
        # debug
        # print(oblist[0].shape)
        adj = oblist[0][:,:,:]
        node = oblist[1][:,0,:]
        ob = {}
        ob["adj"] = adj
        ob["node"] = node
        return ob
    
    def tensor_to_dict(self, obarray):
        """
        Turn numpy observation to origin dict.

        :param obarray: numpy: T * d_e * N * (N + F)
        :return: Molecule observation. Contains:
                'adj': d_e * N * N --- d_e for edge type num. 
                                        N for max atom num.
                'node': 1 * N * F --- F for atom features num.
        """
        nodenum = obarray.shape[2]
        oblist = torch.split(obarray,[nodenum,obarray.shape[-1]-nodenum],-1)
        # print("oblist[1].shape",oblist[1].shape)
        # print("oblist[0].shape",oblist[0].shape)
        adj = oblist[0][:,:,:]
        node = oblist[1][:,0,:]
        node = node.unsqueeze(1)
        # print("adj.shape",adj.shape)
        # print("node.shape",node.shape)
        ob = {}
        ob["adj"] = adj
        ob["node"] = node
        return ob
    
class GCPNFactory(ModelFactory):
    def create(self, state_space: Converter, action_space: Converter, out_channels=128,
               in_channels=9, edge_type=3, atom_type_num=9, stop_shift=-3, max_atomn=47) -> Model:
        return GCPN(state_space, action_space, out_channels, in_channels, 
                    edge_type, atom_type_num, stop_shift, max_atomn)
    

    
