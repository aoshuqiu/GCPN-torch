import copy
import os
import random

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import gym
from gym.spaces import MultiDiscrete,Discrete
import networkx as nx
import torch

from molgym.envs.utils import load_conditional, load_scaffold, convert_radical_electrons_to_hydrogens
from molgym.envs.dataset_utils import gdb_dataset, mol_to_nx
from molgym.envs.critic import CriticMap
from molgym.envs.molecule_spaces import MolecularActionTupleSpace, MolecularObDictSpace

class MoleculeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        RDLogger.DisableLog('rdApp.*')
        pass

    def set_hyperparams(self, device=torch.device('cpu'), data_type='zinc',logp_ratio=1, qed_ratio=1,sa_ratio=1,
                        reward_step_total=1,is_normalize=0,reward_type='qed',reward_target=0.5,
                        has_scaffold=False,has_feature=False,is_conditional=False,conditional='low',
                        max_action=128,min_action=20,force_final=False):
        """
        Set huperparameters of the gym envs.

        :param data_type: ``zinc`` or ``gdb`` Which data type for expert training,
         it also decide possible atom types, defaults to 'zinc'
        :param logp_ratio: Weight for molecule's logp reward, defaults to 1
        :param qed_ratio: Weight for molecule's qed reward, defaults to 1
        :param sa_ratio: Weight for molecule's sa reward, defaults to 1
        :param reward_step_total: Decide maximam reward step total 
         during the generation, defaults to 1
        :param is_normalize: Whether to normalize the observation, defaults to 0
        :param reward_type: Performance type which guides the policy, defaults to 'gan'
        :param reward_target: Target reward value for target generation, defaults to 0.5
        :param has_scaffold: Whether to use fragments instead of atoms, defaults to False
        :param has_feature: Whether to add more information in atom feature
         like ringinformation degrees and so on, defaults to False
        :param is_conditional: Whether to generate from an exist molecule, defaults to False
        :param conditional: Data to load base conditoinal molecule, defaults to 'low'
        :param max_action: Max action count for one molecule (include invalid ones), defaults to 128
        :param min_action: Mini action count for one molecule (include invalid ones), defaults to 20
        :param force_final: Just feedback final reward (final molecule performance), defaults to False
        :param generate_unit: atom or fragment for generate unit.
        """
        self.device = device
        self.is_normalize = bool(is_normalize)
        self.is_conditional = is_conditional
        self.has_feature = has_feature
        self.reward_type = reward_type
        self.reward_target = reward_target
        self.force_final = force_final
        self.criticmap = CriticMap(self.device).map
        self.conditional_list = load_conditional(conditional)
        self.smile_list = []
        if data_type=='gdb':
            possible_atoms = ['C', 'N', 'O', 'S', 'Cl'] # gdb 13
        elif data_type=='zinc':
            possible_atoms = ['C', 'N', 'O', 'S', 'P', 'F', 'I', 'Cl',
                              'Br']  # ZINC
        if self.has_feature:
            self.possible_formal_charge = np.array([-1, 0, 1])
            self.possible_implicit_valence = np.array([-1,0, 1, 2, 3, 4])
            self.possible_ring_atom = np.array([True, False])
            self.possible_degree = np.array([0, 1, 2, 3, 4, 5, 6, 7])
            self.possible_hybridization = np.array([                      # 杂化轨道类型 电子层的亚层
                Chem.rdchem.HybridizationType.SP,
                                      Chem.rdchem.HybridizationType.SP2,
                                      Chem.rdchem.HybridizationType.SP3,
                                      Chem.rdchem.HybridizationType.SP3D,
                                      Chem.rdchem.HybridizationType.SP3D2],
                dtype=object)
        possible_bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE] #, Chem.rdchem.BondType.AROMATIC (芳香键) 单键，双键，三键 smiles中是没有芳香键的
        self.atom_type_num = len(possible_atoms)    # 可能加的原子的类型数
        self.possible_atom_types = np.array(possible_atoms)
        self.possible_bond_types = np.array(possible_bonds, dtype=object)
        if self.has_feature:
            # self.d_n = len(self.possible_atom_types) + len(
            #     self.possible_formal_charge) + len(
            #     self.possible_implicit_valence) + len(self.possible_ring_atom) + \
            #       len(self.possible_degree) + len(self.possible_hybridization)
            self.d_n = len(self.possible_atom_types)+6 # 6 is the ring feature
        else:
            self.d_n = len(self.possible_atom_types)

        self.max_action = max_action
        self.min_action = min_action
        if data_type=='gdb':
            self.max_atom = 13 + len(possible_atoms) # gdb 13
        elif data_type=='zinc':
            if self.is_conditional:
                self.max_atom = 38 + len(possible_atoms) + self.min_action # ZINC
            else:
                self.max_atom = 38 + len(possible_atoms) # ZINC  + self.min_action

        self.logp_ratio = logp_ratio
        self.qed_ratio = qed_ratio
        self.sa_ratio = sa_ratio
        self.reward_step_total = reward_step_total
        self.action_space = MultiDiscrete([self.max_atom, self.max_atom, 3, 2])
        self.observation_space = gym.Space(shape=[len(possible_bonds), self.max_atom, self.max_atom+self.d_n])

        self.counter = 0
        self.init_molecule(None)

        ## load expert data
        cwd = os.path.dirname(__file__)
        if data_type=='gdb':
            path = os.path.join(os.path.dirname(cwd), 'dataset',
                                'gdb13.rand1M.smi.gz')  # gdb 13
        elif data_type=='zinc':
            path = os.path.join(os.path.dirname(cwd), 'dataset',
                                '250k_rndm_zinc_drugs_clean_sorted.smi')  # ZINC
        self.dataset = gdb_dataset(path)

        ## load scaffold data if necessary
        self.has_scaffold = has_scaffold
        if has_scaffold:
            self.scaffold = load_scaffold()
            self.max_scaffold = 6
        self.level = 0 # for curriculum learning, level starts with 0, and increase afterwards

        self.action_space = MolecularActionTupleSpace(self.max_atom, len(self.possible_atom_types), len(self.possible_bond_types))
        self.observation_space = MolecularObDictSpace(self.max_atom, len(self.possible_atom_types), len(self.possible_bond_types), self.d_n)

    def level_up(self):
        """
        Raise the level of Curriculum Study in expert learning, which decides how difficult
        expert molecule data is.
        """
        self.level += 1

    def seed(self, seed):
        np.random.seed(seed=seed)
        random.seed(seed)

    @staticmethod
    def normalize_adj(adj):
        """
        Change adjacency matrix to Normalized Laplace matrix for embeding.

        :param adj: E * A * A, E for possible edge type number; 
                               A for current molecule atom number. 
        :return: E * A * A,  Normalized Laplace matrix
        """
        degrees = np.sum(adj,axis=2)
        # print('degrees',degrees)
        D = np.zeros((adj.shape[0],adj.shape[1],adj.shape[2]))
        for i in range(D.shape[0]):
            D[i,:,:] = np.diag(np.power(degrees[i,:],-0.5))
        adj_normal = D@adj@D
        adj_normal[np.isnan(adj_normal)]=0
        return adj_normal
    
    def init_molecule(self, smiles):
        """
        Init env's molecule 

        :param smiles: Generate base on certain molecule for conditional generation.
        """
        if self.is_conditional:
            self.conditional = random.sample(self.conditional_list, 1)[0]
            self.mol = Chem.RWMol(Chem.MolFromSmiles(self.conditional[0]))
            Chem.SanitizeMol(self.mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        elif smiles is not None:
            self.mol = Chem.RWMol(Chem.MolFromSmiles(smiles))
            Chem.SanitizeMol(self.mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        else:
            self.mol = Chem.RWMol()
            # self._add_atom(np.random.randint(len(self.possible_atom_types)))  # random add one atom
            self._add_atom(0) # always add carbon firs

    def _add_atom(self, atom_type_id):
        """
        Adds an atom
        :param atom_type_id: atom_type id
        :return:
        """
        # assert action.shape == (len(self.possible_atom_types),)
        # atom_type_idx = np.argmax(action)
        atom_symbol = self.possible_atom_types[atom_type_id]
        self.mol.AddAtom(Chem.Atom(atom_symbol))

    def step(self, action):
        """
        Perform a given action
        :param action: (first, second, edge, stop), action that need to be done.
        :return: reward of 1 if resulting molecule graph does not exceed valency,
        -1 if otherwise.
        """
        ### init
        info = {}
        self.mol_old = copy.deepcopy(self.mol) # keep old mol
        total_atoms = self.mol.GetNumAtoms()
        # debug
        # print("action: ", action)
        ### take action
        if action[3]==0 or self.counter < self.min_action: # not stop
            stop = False
            if action[1] >= total_atoms:
                self._add_atom(action[1] - total_atoms)  # add new node
                action[1] = total_atoms  # new node id
                self._add_bond(action)  # add new edge
            else:
                self._add_bond(action)  # add new edge
        else: # stop
            stop = True

        ### calculate intermediate rewards 
        if self.check_valency(): 
            if self.mol.GetNumAtoms()+self.mol.GetNumBonds()-self.mol_old.GetNumAtoms()-self.mol_old.GetNumBonds()>0:
                reward_step = self.reward_step_total/self.max_atom # successfully add node/edge
                self.smile_list.append(self.get_final_smiles())
            else:
                reward_step = -self.reward_step_total/self.max_atom # edge exist
        else:
            reward_step = -self.reward_step_total/self.max_atom  # invalid action
            self.mol = self.mol_old

        ### calculate terminal rewards
        if self.fit_terminate_condition(stop) or self.force_final:
            reward_valid_func = self.criticmap['valid']
            reward_final_func = self.criticmap[self.reward_type]

            final_mol = convert_radical_electrons_to_hydrogens(self.mol)
            final_mol = Chem.MolFromSmiles(Chem.MolToSmiles(final_mol, isomericSmiles=True))
            reward_valid = reward_valid_func(final_mol)
            try:
                reward_final = reward_final_func(final_mol)
            except Exception as ex:
                print(f"reward error : {ex}, {type(ex)}")
                reward_final = 0
            
            new = True # end of episode
            if self.force_final:
                reward = reward_final
            else:
                reward = reward_step + reward_final + reward_valid
            
            info['smiles'] = self.get_final_smiles()
            if self.is_conditional:
                info['reward_valid'] = self.conditional[-1] ### temp change
            else:
                info['reward_valid'] = reward_valid
            info['final_stat'] = reward_final
            info['reward'] = reward
            info['stop'] = stop

        ### use stepwise reward
        else:
            new = False
            reward = reward_step

        # get observation
        ob = self.get_observation()

        self.counter += 1
        if new:
            self.counter = 0
        
        return ob, reward, new, info

    def _add_bond(self, action):
        '''
        :param action: [first_node, second_node, bong_type_id]
        :return:
        '''
        # GetBondBetweenAtoms fails for np.int64
        bond_type = self.possible_bond_types[action[2]]

        # if bond exists between current atom and other atom, modify the bond
        # type to new bond type. Otherwise create bond between current atom and
        # other atom with the new bond type
        bond = self.mol.GetBondBetweenAtoms(int(action[0]), int(action[1]))
        if bond:
            # print('bond exist!')
            return False
        else:
            self.mol.AddBond(int(action[0]), int(action[1]), order=bond_type)
            # bond = self.mol.GetBondBetweenAtoms(int(action[0, 0]), int(action[0, 1]))
            # bond.SetIntProp('ordering',self.mol.GetNumBonds())
            return True

    def get_expert(self, batch_size,is_final=False,curriculum=0,level_total=6,level=0):
        ob = {}
        atom_type_num = len(self.possible_atom_types)
        bond_type_num = len(self.possible_bond_types)
        ob['node'] = np.zeros((batch_size, 1, self.max_atom, self.d_n))
        ob['adj'] = np.zeros((batch_size, bond_type_num, self.max_atom, self.max_atom))

        ac = np.zeros((batch_size, 4))
        ### select molecule
        dataset_len = len(self.dataset)
        for i in range(batch_size):
            is_final_temp = is_final
            # print('--------------------------------------------------')
            ### get a subgraph
            if curriculum==1:
                ratio_start = level/float(level_total)
                ratio_end = (level+1)/float(level_total)
                idx = np.random.randint(int(ratio_start*dataset_len), int(ratio_end*dataset_len))
            else:
                idx = np.random.randint(0, dataset_len)
            mol = self.dataset[idx]
            # print('ob_before',Chem.MolToSmiles(mol, isomericSmiles=True))
            # from rdkit.Chem import Draw
            # Draw.MolToFile(mol, 'ob_before'+str(i)+'.png')
            # mol = self.dataset[i] # sanitity check
            Chem.SanitizeMol(mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            graph = mol_to_nx(mol)
            edges = graph.edges()
            # # always involve is_final probability
            # if is_final==False and np.random.rand()<1.0/batch_size:
            #     is_final = True

            # select the edge num for the subgraph
            if is_final_temp:
                edges_sub_len = len(edges)
            else:
                # edges_sub_len = random.randint(1,len(edges))
                edges_sub_len = random.randint(1,len(edges)+1)
                if edges_sub_len==len(edges)+1:
                    edges_sub_len = len(edges)
                    is_final_temp=True
            edges_sub = random.sample(edges,k=edges_sub_len)
            graph_sub = nx.Graph(edges_sub)
            graph_sub = max(nx.connected_component_subgraphs(graph_sub), key=len)
            if is_final_temp: # when the subgraph the whole molecule, the expert show stop sign
                node1 = random.randint(0,mol.GetNumAtoms()-1)
                while True:
                    node2 = random.randint(0,mol.GetNumAtoms()+atom_type_num-1)
                    if node2!=node1:
                        break
                edge_type = random.randint(0,bond_type_num-1)
                ac[i,:] = [node1,node2,edge_type,1] # stop
            else:
                ### random pick an edge from the subgraph, then remove it 要一个最大的连通图
                edge_sample = random.sample(graph_sub.edges(),k=1)
                graph_sub.remove_edges_from(edge_sample)
                graph_sub = max(nx.connected_component_subgraphs(graph_sub), key=len)
                edge_sample = edge_sample[0] # get value
                ### get action
                if edge_sample[0] in graph_sub.nodes() and edge_sample[1] in graph_sub.nodes():
                    node1 = list(graph_sub.nodes()).index(edge_sample[0])
                    node2 = list(graph_sub.nodes()).index(edge_sample[1])
                elif edge_sample[0] in graph_sub.nodes():
                    node1 = list(graph_sub.nodes()).index(edge_sample[0])
                    node2 = np.argmax(
                        graph.node[edge_sample[1]]['symbol'] == self.possible_atom_types) + graph_sub.number_of_nodes()
                elif edge_sample[1] in graph_sub.nodes():
                    node1 = list(graph_sub.nodes()).index(edge_sample[1])
                    node2 = np.argmax(
                        graph.node[edge_sample[0]]['symbol'] == self.possible_atom_types) + graph_sub.number_of_nodes()
                else:
                    print('Expert policy error!')
                edge_type = np.argmax(graph[edge_sample[0]][edge_sample[1]]['bond_type'] == self.possible_bond_types)
                ac[i,:] = [node1,node2,edge_type,0] # don't stop
                # print('action',[node1,node2,edge_type,0])
            # print('action',ac)
            # plt.axis("off")
            # nx.draw_networkx(graph_sub)
            # plt.show()
            ### get observation
            # rw_mol = Chem.RWMol()
            n = graph_sub.number_of_nodes()
            for node_id, node in enumerate(graph_sub.nodes()):
                if self.has_feature:
                    # float_array = np.concatenate([(graph.node[node]['symbol'] ==
                    #                                self.possible_atom_types),
                    #                               (graph.node[node]['formal_charge'] ==
                    #                                self.possible_formal_charge),
                    #                               (graph.node[node]['implicit_valence'] ==
                    #                                self.possible_implicit_valence),
                    #                               (graph.node[node]['ring_atom'] ==
                    #                                self.possible_ring_atom),
                    #                               (graph.node[node]['degree'] == self.possible_degree),
                    #                               (graph.node[node]['hybridization'] ==
                    #                                self.possible_hybridization)]).astype(float)
                    cycle_info = nx.cycle_basis(graph_sub, node)
                    cycle_len_info = [len(cycle) for cycle in cycle_info]
                    # print(cycle_len_info)
                    float_array = np.concatenate([(graph.node[node]['symbol'] ==
                                                   self.possible_atom_types),
                                                  ([len(cycle_info)==0]),
                                                  ([3 in cycle_len_info]),
                                                  ([4 in cycle_len_info]),
                                                  ([5 in cycle_len_info]),
                                                  ([6 in cycle_len_info]),
                                                  ([len(cycle_info)!=0 and (not 3 in cycle_len_info)
                                                   and (not 4 in cycle_len_info)
                                                   and (not 5 in cycle_len_info)
                                                   and (not 6 in cycle_len_info)]
                                                   )]).astype(float)
                else:
                    float_array = (graph.node[node]['symbol'] == self.possible_atom_types).astype(float)

                # assert float_array.sum() == 6
                ob['node'][i, 0, node_id, :] = float_array
                # print('node',node_id,graph.node[node]['symbol'])
                # atom = Chem.Atom(graph.node[node]['symbol'])
                # rw_mol.AddAtom(atom)
            auxiliary_atom_features = np.zeros((atom_type_num, self.d_n))  # for padding
            temp = np.eye(atom_type_num)
            auxiliary_atom_features[:temp.shape[0], :temp.shape[1]] = temp
            ob['node'][i ,0, n:n + atom_type_num, :] = auxiliary_atom_features

            for j in range(bond_type_num):
                ob['adj'][i, j, :n + atom_type_num, :n + atom_type_num] = np.eye(n + atom_type_num)
            for edge in graph_sub.edges():
                begin_idx = list(graph_sub.nodes()).index(edge[0])
                end_idx = list(graph_sub.nodes()).index(edge[1])
                bond_type = graph[edge[0]][edge[1]]['bond_type']
                float_array = (bond_type == self.possible_bond_types).astype(float)
                assert float_array.sum() != 0
                ob['adj'][i, :, begin_idx, end_idx] = float_array
                ob['adj'][i, :, end_idx, begin_idx] = float_array
                # print('edge',begin_idx,end_idx,bond_type)
                # rw_mol.AddBond(begin_idx, end_idx, order=bond_type)
            if self.is_normalize:
                ob['adj'][i] = MoleculeEnv.normalize_adj(ob['adj'][i])
            # print('ob',Chem.MolToSmiles(rw_mol, isomericSmiles=True))
            # from rdkit.Chem import Draw
            # Draw.MolToFile(rw_mol, 'ob' + str(i) + '.png')

        return ob,ac

    def get_observation(self):
        """
        ob['adj']:d_e*max_atom*max_atom --- 'E' 代表边类型的邻接矩阵
        ob['node']:1*max_atom*d_n --- 'F' 1*原子类型*原子嵌入
        n = atom_num + atom_type_num 原子数目+原子种类数
        """
        mol = copy.deepcopy(self.mol)
        try:
            Chem.SanitizeMol(mol)
        except:
            pass
        n = mol.GetNumAtoms()
        n_shift = len(self.possible_atom_types) # assume isolated nodes new nodes exist


        F = np.zeros((1, self.max_atom, self.d_n))
        for a in mol.GetAtoms():
            atom_idx = a.GetIdx()
            atom_symbol = a.GetSymbol()
            # if self.has_feature:
                # formal_charge = a.GetFormalCharge()
                # implicit_valence = a.GetImplicitValence()
                # ring_atom = a.IsInRing()
                # degree = a.GetDegree()
                # hybridization = a.GetHybridization()
            # print(atom_symbol,formal_charge,implicit_valence,ring_atom,degree,hybridization)
            if self.has_feature:
                # float_array = np.concatenate([(atom_symbol ==
                #                                self.possible_atom_types),
                #                               (formal_charge ==
                #                                self.possible_formal_charge),
                #                               (implicit_valence ==
                #                                self.possible_implicit_valence),
                #                               (ring_atom ==
                #                                self.possible_ring_atom),
                #                               (degree == self.possible_degree),
                #                               (hybridization ==
                #                                self.possible_hybridization)]).astype(float)
                float_array = np.concatenate([(atom_symbol ==
                                               self.possible_atom_types),
                                              ([not a.IsInRing()]),
                                              ([a.IsInRingSize(3)]),
                                              ([a.IsInRingSize(4)]),
                                              ([a.IsInRingSize(5)]),
                                              ([a.IsInRingSize(6)]),
                                              ([a.IsInRing() and (not a.IsInRingSize(3))
                                               and (not a.IsInRingSize(4))
                                               and (not a.IsInRingSize(5))
                                               and (not a.IsInRingSize(6))]
                                               )]).astype(float)
            else:
                float_array = (atom_symbol == self.possible_atom_types).astype(float)
            # assert float_array.sum() == 6   # because there are 6 types of one
            # print(float_array,float_array.sum())
            # hot atom features
            F[0, atom_idx, :] = float_array
        # add the atom features for the auxiliary atoms. We only include the
        # atom symbol features
        auxiliary_atom_features = np.zeros((n_shift, self.d_n)) # for padding
        temp = np.eye(n_shift)
        auxiliary_atom_features[:temp.shape[0], :temp.shape[1]] = temp
        F[0,n:n+n_shift,:] = auxiliary_atom_features
        # print('n',n,'n+n_shift',n+n_shift,auxiliary_atom_features.shape)

        d_e = len(self.possible_bond_types)
        E = np.zeros((d_e, self.max_atom, self.max_atom))
        for i in range(d_e):
            E[i,:n+n_shift,:n+n_shift] = np.eye(n+n_shift)
        for b in self.mol.GetBonds(): # self.mol, very important!! no aromatic
            begin_idx = b.GetBeginAtomIdx()
            end_idx = b.GetEndAtomIdx()
            bond_type = b.GetBondType()
            float_array = (bond_type == self.possible_bond_types).astype(float)
            try:
                assert float_array.sum() != 0
            except:
                print('error',bond_type)
            E[:, begin_idx, end_idx] = float_array
            E[:, end_idx, begin_idx] = float_array
        ob = {}
        if self.is_normalize:
            E = MoleculeEnv.normalize_adj(E)
        ob['adj'] = E
        ob['node'] = F
        ob = self.dict_to_np(ob)
        return ob

    def fit_terminate_condition(self, stop):
        """
        Define whether the molecule fits terminate condition.

        :param stop: stop mark in action.
        :return: True for terminate, False otherwise.
        """
        if self.is_conditional:
            terminate_condition = (self.mol.GetNumAtoms() >= self.max_atom-self.possible_atom_types.shape[0]-self.min_action or 
                                   self.counter >= self.max_action or 
                                   stop) and self.counter >= self.min_action
        else:
            terminate_condition = (self.mol.GetNumAtoms() >= self.max_atom-self.possible_atom_types.shape[0] or 
                                   self.counter >= self.max_action or 
                                   stop) and self.counter >= self.min_action
        return terminate_condition

    def check_valency(self):
        """
        Checks that no atoms in the mol have exceeded their possible
        valency
        :return: True if no valency issues, False otherwise
        """
        try:
            Chem.SanitizeMol(self.mol,
                    sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            return True
        except ValueError:
            return False
        
    # TODO(Bowen): check if need to sanitize again
    def get_final_smiles(self):
        """
        Returns a SMILES of the final molecule. Converts any radical
        electrons into hydrogens. Works only if molecule is valid
        :return: SMILES
        """
        m = convert_radical_electrons_to_hydrogens(self.mol)
        return Chem.MolToSmiles(m, isomericSmiles=True)

    def reset(self, smile=None):
        '''
        to avoid error, assume an atom already exists
        :return: ob
        '''
        if self.is_conditional:
            self.conditional = random.sample(self.conditional_list, 1)[0]
            self.mol = Chem.RWMol(Chem.MolFromSmiles(self.conditional[0]))
            Chem.SanitizeMol(self.mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        elif smile is not None:
            self.mol = Chem.RWMol(Chem.MolFromSmiles(smile))
            Chem.SanitizeMol(self.mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        else:
            self.mol = Chem.RWMol()
            # self._add_atom(np.random.randint(len(self.possible_atom_types)))  # random add one atom
            self._add_atom(0) # always add carbon first
        self.smile_list= [self.get_final_smiles()]
        self.counter = 0
        ob = self.get_observation()
        return ob
    
    def dict_to_np(self, ob):
        """
        Turn molecule observation to a combined numpy array for adaption to runner.

        :param ob: Molecule observation. Contains:
                'adj': d_e * N * N --- d_e for edge type num. 
                                        N for max atom num.
                'node': 1 * N * F --- F for atom features num.
        :return: Combined numpy array: d_e * N * (N + F)
        """
        adj = ob['adj']
        node = ob['node']
        assert adj.shape[1] == node.shape[1]
        node = np.repeat(node, adj.shape[0], 0)
        assert node.shape[0] == adj.shape[0]
        obarray = np.dstack((adj, node)) 
        return obarray
    
    def get_final_smiles(self):
        """
        Returns a SMILES of the final molecule. Converts any radical
        electrons into hydrogens. Works only if molecule is valid
        :return: SMILES
        """
        m = convert_radical_electrons_to_hydrogens(self.mol)
        return Chem.MolToSmiles(m, isomericSmiles=True)