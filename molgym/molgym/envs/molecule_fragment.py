import os
import copy
import random

import torch
import gym
from rdkit import Chem
import numpy as np

from molgym.envs.molecule import MoleculeEnv
from molgym.envs.critic import CriticMap
from molgym.envs.vocab import Vocab
from molgym.envs.molecule_spaces import FragmentActionTupleSpace, MolecularObDictSpace
from molgym.envs.utils import get_item, convert_radical_electrons_to_hydrogens

class ActionConflictException(Exception):
    def __init__(self, message, status):
        super().__init__(message, status)
        self.message = message
        self.status = status

class MoleculeFragmentEnv(MoleculeEnv):
    def __init__(self):
        self.set_hyperparams()
        self.criticmap = CriticMap().map

    def set_hyperparams(self,data_type='zinc',logp_ratio=1, qed_ratio=1,sa_ratio=1,
                        reward_step_total=1,is_normalize=0,reward_type='qed',reward_target=0.5,
                        has_scaffold=False,has_feature=False,is_conditional=False,conditional='low',
                        max_action=128,min_action=20,force_final=False,symmetric_action=True,max_motif_atoms=20,
                        max_atom=65, vocab_file_str="./molgym/molgym/dataset/share.txt", main_struct_file_str="./molgym/molgym/dataset/main_struct.txt"):
        
        super().set_hyperparams(data_type=data_type,logp_ratio=logp_ratio, qed_ratio=qed_ratio,sa_ratio=sa_ratio,
                                reward_step_total=reward_step_total,is_normalize=is_normalize,reward_type=reward_type,
                                reward_target=reward_target,has_scaffold=has_scaffold,has_feature=has_feature,
                                is_conditional=is_conditional,conditional=conditional, max_action=max_action,
                                min_action=min_action,force_final=force_final)
        print("MoleculeFragmentEnv")
        cwd = os.path.dirname(__file__)
        self.vocab, self.mof_dic = Vocab.get_cof_vocab(vocab_file_str)
        self.main_struct = Vocab.get_main_struct(main_struct_file_str)
        self.vocab_size = self.vocab.size()
        self.max_motif_atoms=max_motif_atoms
        self.max_atom = max_atom
        if symmetric_action:
            self.symmetric_action = symmetric_action
            self.symmetry = []
            self.end_points = [[],[]]
            self.last_motif = "C"
            self.last_connect = [0,0]

        # TODO dn   
        self.action_space = FragmentActionTupleSpace(self.max_atom, len(self.possible_atom_types), len(self.possible_bond_types), self.max_motif_atoms, self.vocab_size)
        self.observation_space = MolecularObDictSpace(self.max_atom, len(self.possible_atom_types), len(self.possible_bond_types), self.d_n)

    def step(self, action):
        ### init
        info = {}  # info we care about
        self.mol_old = copy.deepcopy(self.mol) # keep old mol
        self.symmetry_old = copy.deepcopy(self.symmetry)
        self.motif_old = copy.deepcopy(self.last_motif)
        self.connect_old = copy.deepcopy(self.last_connect)
        self.end_points_old = copy.deepcopy(self.end_points)
        total_atoms = self.mol.GetNumAtoms()

        add_atom_num = 1

        def potential_to_share(mol, motif, in_atom_idx, out_atom_idx):
            in_bond = -1
            out_bond = -1
            for bond in mol.GetAtomWithIdx(get_item(in_atom_idx)).GetBonds():
                if(bond.GetBondType()==Chem.rdchem.BondType.DOUBLE):
                    in_bond = bond.GetIdx()
                    break
            
            for bond in motif.GetAtomWithIdx(get_item(out_atom_idx)).GetBonds():
                if(bond.GetBondType()==Chem.rdchem.BondType.DOUBLE):
                    out_bond = bond.GetIdx()
                    break
            if(in_bond==-1 or out_bond==-1): return None
            else: return (in_bond, out_bond) 

            
            
        ### take action action 矩阵，每行四个元素，分别是动作的四个组成，之后的操作在加链路
        if action[4]==0 or self.counter < self.min_action: # not stop
            stop = False
            motif = Chem.RWMol(Chem.MolFromSmiles(self.vocab.vocab_list[action[0]]))
            Chem.SanitizeMol(motif, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            share_bonds = potential_to_share(self.mol, motif,action[1], action[2])
            if action[3]!=len(self.possible_bond_types) or share_bonds is None:
                add_atom_num = self._add_motif(action)
            else:
                add_atom_num = self._share_motif(action, share_bonds)

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
            self.symmetry = self.symmetry_old
            self.last_connect = self.connect_old
            self.last_motif = self.motif_old
            self.end_points = self.end_points_old

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
        d_e = len(self.possible_bond_types)
        E = np.zeros((d_e, self.max_atom, self.max_atom))
        for i in range(d_e):
            E[i,:n,:n] = np.eye(n)
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

    def reset(self,smile=None):
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
            while True:
                smiles, symmetry_list = random.choice(list(self.main_struct.items()))
                vocab_mol = Chem.MolFromSmiles(smiles)
                if vocab_mol:
                    break
            self.mol = vocab_mol
            self.symmetry = copy.deepcopy(symmetry_list)
            Chem.SanitizeMol(self.mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            # self._add_atom(np.random.randint(len(self.possible_atom_types)))  # random add one atom
        self.smile_list= [self.get_final_smiles()]
        self.counter = 0
        ob = self.get_observation()
        return ob
    
    def _connect_motifs(self, mol_1, mol_2, begin_atom_idx,
                                   end_atom_idx, bond_type):
        """
        Given two rdkit mol objects, begin and end atom indices of the new bond, the bond type, returns a new mol object
        that has the corresponding bond added. Note that the atom indices are based on the combined mol object, see below
        MUST PERFORM VALENCY CHECK AFTERWARDS
        :param mol_1:
        :param mol_2:
        :param begin_atom_idx:
        :param end_atom_idx:
        :param bond_type:
        :return: rdkit mol object
        """
        combined = Chem.CombineMols(mol_1, mol_2)
        rw_combined = Chem.RWMol(combined)
        
        # check that we have an atom index from each substructure
        grouped_atom_indices_combined = Chem.GetMolFrags(rw_combined)
        substructure_1_indices, substructure_2_indices = grouped_atom_indices_combined
        if(torch.is_tensor(begin_atom_idx)):begin_atom_idx = begin_atom_idx.item()
        if(torch.is_tensor(end_atom_idx)):end_atom_idx = end_atom_idx.item()
        end_atom_idx = end_atom_idx + mol_1.GetNumAtoms()
        if begin_atom_idx in substructure_1_indices and end_atom_idx in substructure_2_indices:
            try:    
                rw_combined.AddBond(begin_atom_idx, end_atom_idx, bond_type)
                return rw_combined.GetMol()
            except Exception as e:
                return mol_1
        else:
            return mol_1    
    # motif_idx, begin_atom_idx, end_atom_idx, bond_type
    def _add_motif(self, action):
        motif_idx = action[0]
        begin_atom_idx = action[1]
        end_atom_idx = action[2]
        bond_type = self.possible_bond_types[action[3] if action[3]!=len(self.possible_bond_types) else 0]
        motif_mol1 = Chem.RWMol(Chem.MolFromSmiles(self.vocab.vocab_list[motif_idx]))
        Chem.SanitizeMol(motif_mol1, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        motif_mol2 = Chem.RWMol(Chem.MolFromSmiles(self.vocab.vocab_list[motif_idx]))
        Chem.SanitizeMol(motif_mol2,sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        curnum = self.mol.GetNumAtoms()
        old_mol = copy.deepcopy(self.mol)
        self.mol =  self._connect_motifs(Chem.RWMol(self.mol), motif_mol1, begin_atom_idx, end_atom_idx, bond_type)
        newnum = self.mol.GetNumAtoms()
        if self.mol.GetNumAtoms()==curnum:
            return 0
        else:
            self.mol =  self._connect_motifs(Chem.RWMol(self.mol), motif_mol2, self.symmetry[begin_atom_idx], end_atom_idx, bond_type)
            if(self.mol.GetNumAtoms()==newnum):
                self.mol = old_mol
                return 0
            symmetry_tail = []
            self.end_points = [[],[]]
            self.last_motif = self.vocab.vocab_list[motif_idx]
            for atom in motif_mol1.GetAtoms():
                atom_idx1 = atom.GetIdx()+curnum
                atom_idx2 = atom_idx1+motif_mol1.GetNumAtoms()
                self.symmetry.append(atom_idx2)
                symmetry_tail.append(atom_idx1)
                self.end_points[0].append(atom_idx1)
                self.end_points[1].append(atom_idx2)
            self.symmetry.extend(symmetry_tail)
            self.last_connect = [end_atom_idx+curnum, self.symmetry[end_atom_idx+curnum]]
            return motif_mol1.GetNumAtoms()
        
    def _share_motif(self, action, share_bonds):
        if(action[3]!=len(self.possible_bond_types)):
            raise ActionConflictException("action requires no share edges!", 0)
        motif_idx = action[0]
        begin_atom_idx = action[1]
        end_atom_idx = action[2]
        motif_mol1 = Chem.RWMol(Chem.MolFromSmiles(self.vocab.vocab_list[motif_idx]))
        Chem.SanitizeMol(motif_mol1, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        curnum = self.mol.GetNumAtoms()
        old_mol = copy.deepcopy(self.mol)
        self.mol = self._glue_motifs(Chem.RWMol(self.mol), motif_mol1, share_bonds[0], share_bonds[1])
        newnum = self.mol.GetNumAtoms()
        if self.mol.GetNumAtoms() == curnum:
            return 0
        else:
            motif_mol2 = Chem.RWMol(Chem.MolFromSmiles(self.vocab.vocab_list[motif_idx]))
            Chem.SanitizeMol(motif_mol2,sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            sym_bond = self.mol.GetBondBetweenAtoms(self.symmetry[self.mol.GetBondWithIdx(share_bonds[0]).GetBeginAtomIdx()],
                self.symmetry[self.mol.GetBondWithIdx(share_bonds[0]).GetEndAtomIdx()])
            if(sym_bond is None):
                self.mol = old_mol
                return 0
            
            self.mol = self._glue_motifs(Chem.RWMol(self.mol), motif_mol2, sym_bond.GetIdx(), share_bonds[1])
            if(self.mol.GetNumAtoms()==newnum):
                self.mol = old_mol
                return 0
            symmetry_tail = []
            # TODO need to add another pair.
            self.last_connect = [begin_atom_idx, self.symmetry[begin_atom_idx]]
            self.end_points = [[],[]]
            self.last_motif = self.vocab.vocab_list[motif_idx]
            for idx in range(curnum, newnum):
                atom_idx1 = idx
                atom_idx2 = idx-curnum+newnum
                self.symmetry.append(atom_idx2)
                symmetry_tail.append(atom_idx1)
                self.end_points[0].append(atom_idx1)
                self.end_points[1].append(atom_idx2)
            self.symmetry.extend(symmetry_tail)
            return motif_mol1.GetNumAtoms()-2

    def _glue_motifs(self, mol_1, mol_2, bond_1_idx, bond_2_idx):
        mol_1_num = mol_1.GetNumAtoms()
        mol_1 = Chem.RWMol(mol_1)
        mol_1_atoms = [mol_1.GetBondWithIdx(bond_1_idx).GetBeginAtomIdx(), mol_1.GetBondWithIdx(bond_1_idx).GetEndAtomIdx()]
        mol_2_atoms = [mol_2.GetBondWithIdx(bond_2_idx).GetBeginAtomIdx(), mol_2.GetBondWithIdx(bond_2_idx).GetEndAtomIdx()]
        begindic = {}
        enddic = {}
        try:
            for idx in range(mol_2.GetNumBonds()):
                bond = mol_2.GetBondWithIdx(idx)
                if(bond.GetBeginAtomIdx()==mol_2_atoms[0] and bond.GetEndAtomIdx()!=mol_2_atoms[1]):
                    begindic[bond.GetEndAtomIdx()] = bond.GetBondType()   
                if(bond.GetEndAtomIdx()==mol_2_atoms[1] and bond.GetBeginAtomIdx()!=mol_2_atoms[0]):
                    enddic[bond.GetBeginAtomIdx()] = bond.GetBondType()
                if(bond.GetBeginAtomIdx()==mol_2_atoms[1]):
                    enddic[bond.GetEndAtomIdx()] = bond.GetBondType()
                if(bond.GetEndAtomIdx()==mol_2_atoms[0]):
                    begindic[bond.GetBeginAtomIdx()] = bond.GetBondType()
            combined = Chem.CombineMols(mol_1, mol_2)
            rw_combined = Chem.RWMol(combined)
            for idx in begindic:
                rw_combined.AddBond(mol_1_atoms[0], idx+mol_1_num, begindic[idx])
            for idx in enddic:
                rw_combined.AddBond(mol_1_atoms[1], idx+mol_1_num, enddic[idx])
            rw_combined.RemoveAtom(max(mol_2_atoms[0]+mol_1_num, mol_2_atoms[1]+mol_1_num))
            rw_combined.RemoveAtom(min(mol_2_atoms[0]+mol_1_num, mol_2_atoms[1]+mol_1_num))
            return rw_combined
        except Exception as e:
            return mol_1
    
    @staticmethod
    def get_observation_mol(mol, env_context):
        """
        ob['adj']:d_e*n*n --- 'E' 代表边类型的邻接矩阵
        ob['node']:1*n*d_n --- 'F' 1*原子类型*原子嵌入
        n = atom_num + atom_type_num 原子数目+原子种类数
        """
        try:
            Chem.SanitizeMol(mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except:
            pass
        n = mol.GetNumAtoms()
        data_type = env_context["data_type"]
        if data_type=='gdb':
            possible_atoms = ['C', 'N', 'O', 'S', 'Cl'] # gdb 13
        elif data_type=='zinc':
            possible_atoms = ['C', 'N', 'O', 'S', 'P', 'F', 'I', 'Cl',
                              'Br']
        possible_bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE] #, Chem.rdchem.BondType.AROMATIC (芳香键) 单键，双键，三键 smiles中是没有芳香键的
        atom_type_num = len(possible_atoms)    # 可能加的原子的类型数
        possible_atom_types = np.array(possible_atoms)
        possible_bond_types = np.array(possible_bonds, dtype=object)
        d_n = len(possible_atom_types)+6 if env_context["has_feature"] else len(possible_atom_types)
        #print("mol_atom_num   ", n)
        F = np.zeros((1, env_context["max_motif_atoms"], d_n))
        for a in mol.GetAtoms():
            atom_idx = a.GetIdx()
            atom_symbol = a.GetSymbol()
            if env_context["has_feature"]:
                formal_charge = a.GetFormalCharge()
                implicit_valence = a.GetImplicitValence()
                ring_atom = a.IsInRing()
                degree = a.GetDegree()
                hybridization = a.GetHybridization()
            # print(atom_symbol,formal_charge,implicit_valence,ring_atom,degree,hybridization)
            if env_context["has_feature"]:
                float_array = np.concatenate([(atom_symbol ==
                                               possible_atom_types),
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
                float_array = (atom_symbol == possible_atom_types).astype(float)
            F[0, atom_idx, :] = float_array

        d_e = len(possible_bond_types)
        max_motif_atoms = env_context["max_motif_atoms"]
        E = np.zeros((d_e, max_motif_atoms, max_motif_atoms))
        for i in range(d_e):
            E[i,:n,:n] = np.eye(n)
        for b in mol.GetBonds(): # mol, very important!! no aromatic
            begin_idx = b.GetBeginAtomIdx()
            end_idx = b.GetEndAtomIdx()
            bond_type = b.GetBondType()
            float_array = (bond_type == possible_bond_types).astype(float)
            try:
                assert float_array.sum() != 0
            except:
                print('error',bond_type)
            E[:, begin_idx, end_idx] = float_array
            E[:, end_idx, begin_idx] = float_array
        ob = {}
        if env_context["is_normalize"]:
            E = MoleculeEnv.normalize_adj(E)
        ob['adj'] = E
        ob['node'] = F
        return ob
    