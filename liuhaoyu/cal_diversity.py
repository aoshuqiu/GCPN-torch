import csv
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import Geometry
from rdkit.Geometry import rdGeometry
from rdkit.DataStructs import cDataStructs
import os

motif_molset = []
with open('./molcule_cof_curi_sa_scale02_copy.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    num_count=0
    innovel_count=0
    for row in spamreader:
        if row[0] not in motif_molset:
            motif_molset.append(row[0])
            num_count+=1
        if num_count >= 3000:
            del(motif_molset[0])   
            num_count-=1   
dis_arr = []
for index, smiles in enumerate(motif_molset):
    idx = motif_molset.index(smiles)
    # from IPython.display import clear_output as clear
    # clear()
    if(index%100==0):
        print("完成度：{}%".format((float(idx)/len(motif_molset))*100))
    for i in range(idx+1 ,len(motif_molset)):
        m1 = Chem.MolFromSmiles(smiles)
        m2 = Chem.MolFromSmiles(motif_molset[i])
        bv1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=1024)
        bv2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=1024)
        dis_arr.append(1-cDataStructs.TanimotoSimilarity(bv1,bv2))
dis_arr = np.array(dis_arr)
print("生成分子的diversity为： ",np.mean(dis_arr))
