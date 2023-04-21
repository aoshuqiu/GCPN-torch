import torch
from torch import nn
from torch import distributions as D
from torch import functional as F
from rdkit import Chem
import numpy as np

from models.model import Model, ModelFactory
from models.gcn import GCN
from models.gcpn import GCPN
from envs import Converter
from molgym.envs import Vocab
from molgym.envs import MoleculeFragmentEnv



class FragmentGCPN(GCPN):

    #TODO need to pass vocab and context from factory create() method
    def __init__(self, state_space: Converter, action_space: Converter, out_channels=128,
                 in_channels=9, edge_type=3, atom_type_num=9, stop_shift=-3, max_atom=65, 
                 context = None) -> None:
        super().__init__(state_space, action_space, out_channels, in_channels, edge_type, atom_type_num, stop_shift, max_atom)
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
        self.context = context
        if context["symmetric_action"]:
            self.vocab,_ = Vocab.get_cof_vocab(context["vocab_file_strs"])
        else:
            self.vocab = Vocab.get_vocab_by_counter(context["vocab_file_strs"], context["thresholds"])
        self.policy_stop_out = nn.Linear(out_channels, 2)
        vocab_size = self.vocab.size()

        self.policy_motif_input = nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=False),
            nn.Tanh()
        )
        self.policy_motif_out = nn.Linear(out_channels, vocab_size)

        self.policy_first_out = nn.Sequential(
            nn.Linear(2*out_channels, out_channels),
            nn.Tanh(),
            nn.Linear(out_channels, 1)
        )
        self.policy_second_out = nn.Sequential(
            nn.Linear(3*out_channels, out_channels),
            nn.Tanh(),
            nn.Linear(out_channels, 1)
        )
        self.policy_edge_out = nn.Sequential(
            nn.Linear(4*out_channels, out_channels),
            nn.Tanh(),
            nn.Linear(out_channels, edge_type+1)
        )

    def forward(self, state):
        tanh = nn.Tanh()
        state = state.reshape((-1, self.edge_type, self.max_atomn, self.max_atomn+self.in_channels))
        device = state.device
        state = self.tensor_to_dict(state)
        adj = state['adj']
        node = state['node']

        if adj.dim()==3:
            adj = adj.unsqueeze(0)
        if node.dim()==3:
            node = node.unsqueeze(0)

        ob_node = self.emb(node)
        emb_node = tanh(self.gcn1(torch.matmul(adj,ob_node)))
        emb_node = tanh(self.gcn2(torch.matmul(adj,emb_node)))
        emb_node = self.gcn3(torch.matmul(adj,emb_node))
        emb_node = emb_node.squeeze(1) # emb_node: (B,n,f)
        ob_len = GCPN.get_ob_len(node)
        # print("ob_len.shape", ob_len.shape)
        ob_len_first = ob_len
        self.mask_emb_len_(emb_node, ob_len)
        emb_graph = torch.sum(emb_node,1).unsqueeze(1) # emb_graph: (B,1,f)
        
        # Autoregressive Action Space
        # Fisrt get logits then construct distribution eventually sample action.
        emb_stop = self.policy_stop_input(emb_node)
        policy_stop = torch.sum(emb_stop,1) #(B,1,f)
        policy_stop = self.policy_stop_out(policy_stop) #(B,1,2)

        stop_shift = torch.tensor([0, self.stop_shift], device=device)
        pd_stop = D.Categorical(logits=policy_stop + stop_shift)
        ac_stop = pd_stop.sample() #(B,1,1)
        ac_stop = ac_stop.unsqueeze(-1) #(B, 1)

        emb_motif = self.policy_motif_input(emb_node)
        policy_motif = torch.sum(emb_motif,1) #(B,1,f)
        policy_motif = self.policy_motif_out(policy_motif) #(B,1,vocab_size)
        pd_motif = D.Categorical(logits=policy_motif)
        ac_motif = pd_motif.sample() #(B,1)
        ac_motif = ac_motif.unsqueeze(-1) #(B)

        motif_embs, motif_sembs, motif_nodes = self.get_motif_embs(ac_motif)

        emb_first_cat = torch.cat((motif_embs.expand(emb_node.shape), emb_node), -1)
        policy_first = self.policy_first_out(emb_first_cat).squeeze(-1)
        GCPN.mask_logits_len_(policy_first, ob_len_first)
        pd_first = D.Categorical(logits=policy_first)
        ac_first = pd_first.sample() #(B)
        ac_first = ac_first.unsqueeze(-1) #(B,1)
        emb_first = GCPN.filter_first_node(emb_node, ac_first)

        emb_second_cat = torch.cat((emb_graph.expand(motif_sembs.shape),
                                    emb_first.expand(motif_sembs.shape),
                                    motif_sembs), -1)
        motif_ob_len = GCPN.get_ob_len(motif_nodes)
        policy_second = self.policy_second_out(emb_second_cat).squeeze(-1)
        GCPN.mask_logits_len_(policy_second, motif_ob_len)
        pd_second = D.Categorical(logits=policy_second)
        ac_second = pd_second.sample() #(B)
        ac_second = ac_second.unsqueeze(-1) #(B,1)
        emb_second = GCPN.filter_first_node(motif_sembs, ac_second)

        # # debug
        # print("emb_graph.shape", emb_graph.shape)
        # print("motif_embs.shape", motif_embs.shape)
        # print("emb_first.shape", emb_first.shape)
        # print("emb_second.shape", emb_second.shape)
        # input()

        emb_edge_cat = torch.cat((emb_graph.squeeze(1), 
                                  motif_embs.squeeze(1),
                                  emb_first.squeeze(1),
                                  emb_second.squeeze(1)), -1) #(B, 4f)
        policy_edge = self.policy_edge_out(emb_edge_cat).squeeze(-1) #(B, edge_type+1)
        pd_edge = D.Categorical(logits=policy_edge)
        ac_edge = pd_edge.sample() #(B)
        ac_edge = ac_edge.unsqueeze(-1) #(B,1)

        value = self.value_input(emb_node)
        value = torch.sum(value,1) 
        value = self.value_out(value)

        ac = torch.cat((ac_motif, ac_first, ac_second, ac_edge, ac_stop),-1)

        policy = {
            'motif': policy_motif,
            'first': policy_first,
            'second': policy_second,
            'edge': policy_edge,
            'stop': policy_stop,
        }
        # TODO no arrange for node is curtain
        policy = self._arrange_policy(policy)
        return ac, policy, value

    def _arrange_policy(self, policy):
        return torch.cat((policy['motif'], policy['first'], 
                          policy['second'], policy['edge'], 
                          policy['stop']), -1)
    
    def get_motif_embs(self, ac_motif:torch.Tensor):
        motif_wholeembs=[]
        motif_sembs=[]
        motif_nodes=[]
        device = ac_motif.device
        tanh = nn.Tanh()

        for motif_idx in ac_motif:
            motif_smiles = self.vocab.vocab_list[int(motif_idx.item())]
            motif_ob = MoleculeFragmentEnv.get_observation_mol(Chem.MolFromSmiles(motif_smiles),self.context)
            motif_adj = torch.tensor(motif_ob["adj"], device=device, dtype=torch.float)
            motif_node = torch.tensor(motif_ob["node"], device=device, dtype=torch.float)
            motif_nodes.append(motif_node.detach().cpu().numpy())
            if motif_adj.dim() == 3:
                motif_adj = motif_adj.unsqueeze(0)
            if motif_node.dim() == 3:
                motif_node = motif_node.unsqueeze(0)
            motif_obnode = self.emb(motif_node)
            motif_emb = tanh(self.gcn1(torch.matmul(motif_adj, motif_obnode)))
            motif_emb = tanh(self.gcn2(torch.matmul(motif_adj, motif_emb)))
            motif_emb = self.gcn3(torch.matmul(motif_adj, motif_emb))
            motif_emb = motif_emb.squeeze(0)

            motif_sinemb = motif_emb.squeeze(0) #(n,f)
            motif_sembs.append(motif_sinemb.detach().cpu().numpy())

            motif_emb = torch.sum(motif_emb,1)
            motif_wholeembs.append(motif_emb.detach().cpu().numpy())
        motif_wholeembs = torch.tensor(np.array(motif_wholeembs), device=device, dtype=torch.float)
        motif_sembs = torch.tensor(np.array(motif_sembs), device=device, dtype=torch.float)
        motif_nodes = torch.tensor(np.array(motif_nodes), device=device, dtype=torch.float)
        return motif_wholeembs, motif_sembs, motif_nodes
    
    @staticmethod
    def factory(context):
        return FragmentGCPNFactory(context)
        
    
class FragmentGCPNFactory(ModelFactory):
    def __init__(self, context):
        self.context = context

    def create(self, state_space: Converter, action_space: Converter):
        return FragmentGCPN(state_space,action_space, context=self.context)
