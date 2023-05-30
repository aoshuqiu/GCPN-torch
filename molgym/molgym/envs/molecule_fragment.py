import os
import copy
import time
import random
import subprocess
import shutil

import torch
import gym
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
from rdkit.Chem.rdmolfiles import MolToSmiles, MolToXYZFile
import numpy as np


from molgym.envs.molecule import MoleculeEnv
from molgym.envs.critic import CriticMap
from molgym.envs.vocab import Vocab
from molgym.envs.molecule_spaces import FragmentActionTupleSpace, MolecularObDictSpace
from molgym.envs.utils import get_item, convert_radical_electrons_to_hydrogens, cd
from molgym.envs.episodic_memory import EpisodicMemory,similarity_to_memory

class ActionConflictException(Exception):
    def __init__(self, message, status):
        super().__init__(message, status)
        self.message = message
        self.status = status

class MoleculeFragmentEnv(MoleculeEnv):
    def __init__(self):
        RDLogger.DisableLog('rdApp.*')
        pass

    def set_hyperparams(self, device=torch.device('cpu'), data_type='zinc',logp_ratio=1, qed_ratio=1,sa_ratio=1,
                        reward_step_total=1,is_normalize=0,reward_type='qed',reward_target=0.5,
                        has_scaffold=False,has_feature=False,is_conditional=False,conditional='low',
                        max_action=128,min_action=5,force_final=False,symmetric_action=True,max_motif_atoms=20,
                        max_atom=65, vocab_file_strs=["./molgym/molgym/dataset/share.txt",], 
                        main_struct_file_str="./molgym/molgym/dataset/main_struct.txt",
                        zeoplusplus_path="/home/bachelor/zhangjinhang/molRL/zeo++-0.3/",
                        frameworks_gen_path="/home/bachelor/zhangjinhang/molRL/molppo/xyzs",
                        imgs_path="/home/bachelor/zhangjinhang/molRL/molppo/imgs", thresholds=None,
                        capture_logs=False, valid_coeff=1.0, symmetric_cnt=2, use_memory=False,**kwargs):
        
        super().set_hyperparams(device=device,data_type=data_type,logp_ratio=logp_ratio, qed_ratio=qed_ratio,sa_ratio=sa_ratio,
                                reward_step_total=reward_step_total,is_normalize=is_normalize,reward_type=reward_type,
                                reward_target=reward_target,has_scaffold=has_scaffold,has_feature=has_feature,
                                is_conditional=is_conditional,conditional=conditional, max_action=max_action,
                                min_action=min_action,force_final=force_final, valid_coeff=valid_coeff)
        print("MoleculeFragmentEnv")
        cwd = os.path.dirname(__file__)
        self.capture_logs = capture_logs
        self.symmetric_action = symmetric_action
        if self.symmetric_action:
            self.vocab, self.mof_dic = Vocab.get_cof_vocab(vocab_file_strs)
            self.main_struct = Vocab.get_main_struct(main_struct_file_str)
        else:
            self.vocab = Vocab.get_vocab_by_counter(vocab_file_strs,thresholds)
        self.vocab_size = self.vocab.size()
        self.max_motif_atoms=max_motif_atoms
        self.max_atom = max_atom
        self.criticmap = CriticMap(self.device).map
        # debug
        print("vocab_size: ", self.vocab_size)

        if self.symmetric_action:
            self.symmetric_cnt = symmetric_cnt
            self.frameworks_gen_path = frameworks_gen_path
            self.imgs_path = imgs_path
            if(not os.path.exists(self.frameworks_gen_path)):
                os.mkdir(self.frameworks_gen_path)
            if(not os.path.exists(self.imgs_path)):
                os.mkdir(self.imgs_path)
            self.zeoplusplus_path = zeoplusplus_path
            self.symmetry = []
            self.end_points = [[] for _ in range(self.symmetric_cnt)]
            self.last_motif = "C"
            self.last_connect = [0 for _ in range(self.symmetric_cnt)]

        # TODO dn   
        self.action_space = FragmentActionTupleSpace(self.max_atom, len(self.possible_atom_types), len(self.possible_bond_types), self.max_motif_atoms, self.vocab_size)
        self.observation_space = MolecularObDictSpace(self.max_atom, len(self.possible_atom_types), len(self.possible_bond_types), self.d_n)
        
        self.episodic_memory = EpisodicMemory(replacement="random")
        self.use_memory = use_memory
        if(self.use_memory):
            print("use_memory")

    def step(self, action):
        ### init
        info = {}  # info we care about
        self.mol_old = copy.deepcopy(self.mol) # keep old mol
        total_atoms = self.mol.GetNumAtoms()
        if self.symmetric_action:
            self.symmetry_old = copy.deepcopy(self.symmetry)
            self.motif_old = copy.deepcopy(self.last_motif)
            self.connect_old = copy.deepcopy(self.last_connect)
            self.end_points_old = copy.deepcopy(self.end_points)

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
            if self.symmetric_action:
                motif = Chem.RWMol(Chem.MolFromSmiles(self.vocab.vocab_list[action[0]]))
                Chem.SanitizeMol(motif, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                share_bonds = potential_to_share(self.mol, motif,action[1], action[2])
                if action[3]!=len(self.possible_bond_types) or share_bonds is None or self.symmetric_cnt>=2:
                    add_atom_num = self._add_motif(action)
                else:
                    add_atom_num = self._share_motif(action, share_bonds)
            else:
                add_atom_num = self._add_motif(action)
        else: # stop
            stop = True

        reward_memory = 0

         ### calculate intermediate rewards 
        if self.check_valency(): 
            if self.mol.GetNumAtoms()+self.mol.GetNumBonds()-self.mol_old.GetNumAtoms()-self.mol_old.GetNumBonds()>0:
                reward_step = self.reward_step_total/self.max_atom # successfully add node/edge
                reward_memory = similarity_to_memory(self.get_final_smiles(), self.episodic_memory)
                self.smile_list.append(self.get_final_smiles())
            else:
                reward_step = -self.reward_step_total/self.max_atom # edge exist
        else:
            reward_step = -self.reward_step_total/self.max_atom  # invalid action
            self.mol = self.mol_old
            if self.symmetric_action:
                self.symmetry = self.symmetry_old
                self.last_connect = self.connect_old
                self.last_motif = self.motif_old
                self.end_points = self.end_points_old
        ### calculate terminal rewards
        if self.fit_terminate_condition(stop) or self.force_final:
            reward_valid_func = self.criticmap['valid']
            final_mol = convert_radical_electrons_to_hydrogens(self.mol)
            final_mol = Chem.MolFromSmiles(Chem.MolToSmiles(final_mol, isomericSmiles=True))
            reward_valid = reward_valid_func(final_mol)
            try:
                if self.reward_type == "homo":
                    reward_cof, singleton = self.generate_framework(self.mol, 
                                                                    self.end_points,
                                                                    self.last_connect,
                                                                    self.reward_type)
                    if singleton != None: self.mol = singleton
                    reward_final = reward_cof
                    self.symmetry=[]
                elif self.reward_type == "GCMC":
                    reward_cof, singleton = self.generate_framework(self.mol,
                                                                    self.end_points,
                                                                    self.last_connect,
                                                                    self.reward_type)
                    if singleton != None: self.mol = singleton
                    reward_final = reward_cof
                    self.symmetry=[]
                else:
                    reward_final_func = self.criticmap[self.reward_type]
                    reward_final = reward_final_func(final_mol)
            except Exception as ex:
                # print(f"reward error : {ex}, {type(ex)}")
                reward_final = 0
            
            new = True # end of episode
            if self.force_final:
                reward = reward_final
            else:
                reward = reward_step + reward_final + self.valid_coeff * reward_valid
            
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

        if self.use_memory:
            reward += 0.5*reward_memory

        self.counter += 1
        if new:
            for smiles in self.smile_list:
                self.episodic_memory.add(smiles, "")  
            self.counter = 0
        
        return ob, reward, new, info
    
    def generate_framework(self, final_mol, end_points, last_connect, reward_type):
        # TODO add logs by(pid+time).txt to capture subprocess output.
        if self.capture_logs:
            if not os.path.exists("./zeo++_logs"):
                os.makedirs("./zeo++_logs")
            process_log_str = f"./zeo++_logs/{os.getpid()}_{time.time()}.txt"
            log_file = open(process_log_str, "w")
        final_mol = Chem.RWMol(final_mol)
        mol_copy = copy.deepcopy(final_mol)
        Chem.SanitizeMol(final_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        mol_len = len(final_mol.GetAtoms())
        mols = []
        slist = []
        max_sa = -1e10
        max_flat = -1e10
        max_mol = final_mol
        flat_mol = None
        def distance_last_connect(mol, end_1, end_2, idx):
            end_1 = get_item(end_1)
            end_2 = get_item(end_2)
            idx = get_item(idx)
            return min(len(Chem.rdmolops.GetShortestPath(mol, end_1, idx)),len(Chem.rdmolops.GetShortestPath(mol, end_2, idx)))

        for i in range(len(end_points[0])):
            try:
                mol_copy = copy.deepcopy(final_mol)
                for j in range(len(end_points)):
                    mol_copy.AddAtom(Chem.Atom("Br"))
                    mol_copy.AddBond(end_points[j][i], mol_len+j, order=Chem.rdchem.BondType.SINGLE)
                Chem.SanitizeMol(mol_copy, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                if(distance_last_connect(mol_copy, last_connect[0], last_connect[1], end_points[0][i])>max_sa):
                    max_mol = mol_copy
                    max_sa = max(max_sa,distance_last_connect(mol_copy, last_connect[0], last_connect[1], end_points[0][i]))
            except Exception as e:
                continue
            
        if(max_sa==-1e10 and max_flat==-1e10): return 0, None
        final_mol = flat_mol if flat_mol else max_mol

        # remove non-flat molecular:
        pbf_mol = copy.deepcopy(final_mol)
        AllChem.EmbedMolecule(pbf_mol, useRandomCoords=True)
        if(rdMolDescriptors.CalcPBF(pbf_mol)>0.1): return 0, None
        print(Chem.MolToSmiles(final_mol))
        try:
            smiles = MolToSmiles(final_mol)
            final_mol = Chem.MolFromSmiles(smiles)
            final_mol = Chem.AddHs(final_mol)
            Chem.SanitizeMol(final_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            AllChem.EmbedMolecule(final_mol, useRandomCoords=True)
            AllChem.MMFFOptimizeMolecule(final_mol)
        except Exception as e:
            # debug
            # print(e)
            return 0, None
        MolToXYZFile(final_mol,"myby.xyz")
        cwd_path = os.getcwd()
        with cd(self.frameworks_gen_path):
            command = [self.zeoplusplus_path+"molecule_to_abstract",
                cwd_path+"/myby.xyz","1",
                cwd_path+"/myby_1.xyz"]
            res = subprocess.run(command) if not self.capture_logs else \
                  subprocess.run(command, stdout=log_file, stderr=log_file)
            if res.returncode != 0:
                return 0 ,None
            command = [self.zeoplusplus_path+"framework_builder", 
                    self.zeoplusplus_path+"nets/hcb.cgd", "1", 
                    smiles, self.zeoplusplus_path+"builder_examples/building_blocks/N3_1.xyz", 
                    cwd_path+"/myby_1.xyz", "7"]
            res = subprocess.run(command) if not self.capture_logs else \
                  subprocess.run(command, stdout=log_file, stderr=log_file)
            command = [self.zeoplusplus_path+"network", 
                    "-cif", "./"+smiles+"_framework.cssr"]
            res = subprocess.run(command) if not self.capture_logs else \
                  subprocess.run(command, stdout=log_file, stderr=log_file)
            def try_remove(file_str):
                try:
                    os.remove(file_str)
                except:
                    pass
            try_remove(cwd_path+"/myby.xyz")
            try_remove(cwd_path+"/myby_1.xyz")
            try_remove("./"+smiles+"_net_full_edges.vtk")
            try_remove("./"+smiles+"_net_full_vertices.xyz")
            try_remove("./"+smiles+"_net_unit_cell.vtk")
            try_remove("./"+smiles+"_ratio.txt")
            try_remove("./"+smiles+"_framework.vtk")
            try_remove("./"+smiles+"_framework_labeled.cssr")
            try_remove("./"+smiles+"_framework.cssr")
            if res.returncode != 0:
                return -1, None
        if reward_type == "homo":
            critic = self.criticmap["homo"]
            reward = critic(self.frameworks_gen_path+"/"+smiles+"_framework.xyz")
            with cd(self.frameworks_gen_path):
                os.rename("./"+smiles+"_framework.xyz", "./"+smiles+"_framework_"+str(-reward)+".xyz")
                os.rename("./"+smiles+"_framework.cif", "./"+smiles+"_framework_"+str(-reward)+".cif")
        elif reward_type == "GCMC":
            shutil.move(self.frameworks_gen_path+"/"+smiles+"_framework.cif","/home/zhangjinhang/GCPN-torch/GCMC/"+smiles+"_framework.cif")
            critic = self.criticmap["GCMC"]
            reward = critic(smiles+"_framework")
        draw = Draw.MolToImage(final_mol)
        draw.save(self.imgs_path+"/"+smiles+"_"+str(-reward)+".jpg")
        final_mol = flat_mol if flat_mol else max_mol
        return reward, final_mol
    
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
        n = min(mol.GetNumAtoms(),self.max_atom)
        n_shift = len(self.possible_atom_types) # assume isolated nodes new nodes exist


        F = np.zeros((1, self.max_atom, self.d_n))
        for a in mol.GetAtoms():
            atom_idx = a.GetIdx()
            if atom_idx >= self.max_atom:
                continue
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
            if begin_idx >= self.max_atom or end_idx >= self.max_atom:
                continue
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
            if self.symmetric_action:
                while True:
                    smiles, symmetry_list = random.choice(list(self.main_struct.items()))
                    vocab_mol = Chem.MolFromSmiles(smiles)
                    if vocab_mol:
                        break
                self.mol = vocab_mol
                self.symmetry = copy.deepcopy(symmetry_list)
                end_points = [[] for _ in range(self.symmetric_cnt)]
                visited = set()
                for atom1, atom2_list in enumerate(symmetry_list):
                    if atom1 not in visited:
                        end_points[0].append(atom1)
                        visited.add(atom1)
                        for i in range(1, self.symmetric_cnt):
                            end_points[i].append(atom2_list[i-1])
                            visited.add(atom2_list[i-1])
                self.end_points = end_points
                self.last_motif = smiles
                self.last_connect = [0,]
                self.last_connect.extend(self.symmetry[0])
            
            else:
                while True:
                    smiles = random.choice(self.vocab.vocab_list)
                    vocab_mol = Chem.MolFromSmiles(smiles)
                    if vocab_mol:
                        break
                self.mol = vocab_mol
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
        begin_atom_idx = get_item(begin_atom_idx)
        end_atom_idx = get_item(end_atom_idx)
        end_atom_idx = end_atom_idx + mol_1.GetNumAtoms()
        if begin_atom_idx in substructure_1_indices and end_atom_idx in substructure_2_indices:
            try:    
                rw_combined.AddBond(begin_atom_idx, end_atom_idx, bond_type)
                return rw_combined.GetMol()
            except Exception as e:
                print("exception: ",e)
                return mol_1
        else:
            return mol_1    
    # motif_idx, begin_atom_idx, end_atom_idx, bond_type
    def _add_motif(self, action):
        if self.symmetric_action:
            motif_idx = action[0]
            begin_atom_idx = action[1]
            end_atom_idx = action[2]
            bond_type = self.possible_bond_types[action[3] if action[3]!=len(self.possible_bond_types) else 0]
            motif_mol_list = []
            originnum = self.mol.GetNumAtoms()
            old_mol = copy.deepcopy(self.mol)

            def _gen_connect_motif(begin_atom_id):
                motif_mol = Chem.RWMol(Chem.MolFromSmiles(self.vocab.vocab_list[motif_idx]))
                Chem.SanitizeMol(motif_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                self.mol =  self._connect_motifs(Chem.RWMol(self.mol), motif_mol, begin_atom_id, end_atom_idx, bond_type)
                newnum = self.mol.GetNumAtoms()
                return newnum
            newnum = _gen_connect_motif(begin_atom_idx)
            if newnum == originnum:
                return 0
            for i in range(self.symmetric_cnt-1):
                curnum = self.mol.GetNumAtoms()
                newnum = _gen_connect_motif(self.symmetry[begin_atom_idx][i])
                if newnum==curnum:
                    self.mol = old_mol
                    return 0
            self.last_motif = self.vocab.vocab_list[motif_idx]
            self.end_points = [[] for _ in range(self.symmetric_cnt)]
            motif_mol = Chem.RWMol(Chem.MolFromSmiles(self.vocab.vocab_list[motif_idx]))
            self.symmetry.extend([[] for _ in range(self.symmetric_cnt*motif_mol.GetNumAtoms())])
            for atom in motif_mol.GetAtoms():
                lis_atoms = []
                for i in range(self.symmetric_cnt):
                    self.end_points[i].append(atom.GetIdx()+originnum+i*motif_mol.GetNumAtoms())
                    lis_atoms.append(atom.GetIdx()+originnum+i*motif_mol.GetNumAtoms())
                for i in range(self.symmetric_cnt):
                    lis = copy.deepcopy(lis_atoms)
                    lis.remove(atom.GetIdx()+originnum+i*motif_mol.GetNumAtoms())
                    self.symmetry[atom.GetIdx()+originnum+i*motif_mol.GetNumAtoms()] = lis
            self.last_connect = [end_atom_idx+curnum]
            self.last_connect.extend(self.symmetry[end_atom_idx+curnum])
            return motif_mol.GetNumAtoms()
        else:
            motif_idx = action[0]
            begin_atom_idx = action[1]
            end_atom_idx = action[2]
            bond_type = self.possible_bond_types[action[3]] if action[3] < 3 else self.possible_bond_types[0]
            motif_mol = Chem.RWMol(Chem.MolFromSmiles(self.vocab.vocab_list[motif_idx]))
            Chem.SanitizeMol(motif_mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            self.mol =  self._connect_motifs(Chem.RWMol(self.mol), motif_mol, begin_atom_idx, end_atom_idx, bond_type)
            return motif_mol.GetNumAtoms()
        
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
    