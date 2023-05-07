from typing import Dict, List

from rdkit import Chem

class Vocab:
    def __init__(self, vocab_list, atom_num_limit=7):
        self.vocab_list = []
        for mol_smiles in vocab_list:
            mol = Chem.MolFromSmiles(mol_smiles)
            if mol and mol.GetNumAtoms()<atom_num_limit:
                try:
                    Chem.SanitizeMol(mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    for a in mol.GetAtoms():
                        if a.GetImplicitValence() > 0:
                            self.vocab_list.append(mol_smiles)
                            break
                except:
                    continue
        self.vocab_list.sort(key=lambda x: len(x), reverse=True)
        self.vmap = {x: i for i, x in enumerate(self.vocab_list)}
        self.length = len(self.vocab_list)
        # if one_hot_perpare:

    def get_vocab_idx(self, vocab):
        return self.vmap.get(vocab, -1)

    def __getitem__(self, idx):
        return self.vocab_list[idx]

    def size(self):
        return self.length


    @staticmethod
    def get_cof_vocab(file_paths: List[str]):
        vocab_set = set()
        dic = {}
        for file_path in file_paths:
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    strs = line.split(' ')
                    vocab_set.add(strs[0])
                    dic[strs[0]] = int(strs[1])
        return Vocab(vocab_set), dic
    
    @staticmethod
    def get_main_struct(file_path):
        main_struct = {}
        with open(file_path) as f:
            try:
                for line in f:
                    line = line.strip()
                    smiles, symmetry = line.split(' ')
                    lis = []
                    for cutstr in symmetry.split('|'):
                        lis.append([])
                        for numstr in cutstr.split(','):
                            lis[-1].append(int(numstr)) 
                    main_struct[smiles] = lis
            except:
                pass 
        return main_struct
    
    @staticmethod
    def get_vocab_by_counter(file_path_list: List[str], thresholds: List[int]):
        counters = []
        for file_path in file_path_list:
            counter = {}
            with open(file_path) as f:
                try:
                    for line in f:
                        line = line.strip()
                        smiles, num = line.split('\t')
                        counter[smiles] = int(num)
                except:
                    pass
            counters.append(counter)
        for threshold in thresholds:
            vocab_list = []
            for k, v in counter.items():
                if v >= threshold and k not in vocab_list:
                    vocab_list.append(k)
        return Vocab(vocab_list)
