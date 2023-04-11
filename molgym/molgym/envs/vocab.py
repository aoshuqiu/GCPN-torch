from typing import Dict

from rdkit import Chem

class Vocab:
    def __init__(self, vocab_set, vocab_count_dict: Dict[str, int]):
        self.vocab_list = []
        for mol_smiles in vocab_set:
            mol = Chem.MolFromSmiles(mol_smiles)
            if mol and mol.GetNumAtoms()<=15:
                try:
                    Chem.SanitizeMol(mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    for a in mol.GetAtoms():
                        if a.GetImplicitValence() > 0:
                            self.vocab_list.append(mol_smiles)
                            break
                except:
                    continue
        self.vocab_list.sort(key=lambda x:len(x),reverse=True)
        print(self.vocab_list)
        self.vmap = {x: i for i, x in enumerate(self.vocab_list)}
        self.length = len(self.vocab_list)
        self.vocab_count_dict = vocab_count_dict
        # if one_hot_perpare:

    def get_vocab_idx(self, vocab):
        return self.vmap.get(vocab, -1)

    def __getitem__(self, idx):
        return self.vocab_list[idx]

    def size(self):
        return self.length


    @staticmethod
    def get_cof_vocab(file_path):
        vocab_set = set()
        dic = {}
        with open(file_path) as f:
            for line in f:
                print(line)
                line = line.strip()
                strs = line.split(' ')
                print(strs)
                vocab_set.add(strs[0])
                dic[strs[0]] = int(strs[1])
        return Vocab(vocab_set, dic)
    
    @staticmethod
    def get_main_struct(file_path):
        main_struct = {}
        with open(file_path) as f:
            try:
                for line in f:
                    line = line.strip()
                    smiles, symmetry = line.split(' ')
                    lis = []
                    for numstr in symmetry.split(','):
                        lis.append(int(numstr))
                    main_struct[smiles] = lis
            except:
                pass 
        return main_struct