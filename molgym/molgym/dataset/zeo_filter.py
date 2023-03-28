import os
import dataset_utils
from dataset_utils import gdb_dataset, load_dataset
import pandas
from rdkit.Chem.rdmolfiles import MolToXYZFile
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
import subprocess

def arg_parser():
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def filter_arg_parser():
    parser = arg_parser()
    parser.add_argument("--opt", action="store_true")
    parser.add_argument("--xyzname", type=str, default="myby")
    parser.add_argument("--resname", type=str, default="zeo_mols")
    return parser

path = './250k_rndm_zinc_drugs_clean_sorted.smi'
count = 0
df = load_dataset(path)
dataset_len=df.shape[0]
count = 0
args = filter_arg_parser().parse_args()
for i in range(dataset_len):
    smiles = df["smiles"][i]
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, useRandomCoords=True)
    if args.opt:
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass
    MolToXYZFile(mol,args.xyzname+".xyz")
    command = ["/home/bachelor/zhangjinhang/molRL/zeo++-0.3/molecule_to_abstract","/home/bachelor/zhangjinhang/molRL/GCPNs/envs/atom-gym-molecule/gym_molecule/dataset/"+args.xyzname+".xyz","0","/home/bachelor/zhangjinhang/molRL/GCPNs/envs/atom-gym-molecule/gym_molecule/dataset/"+args.xyzname+"_1.xyz"]
    res = subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if res.returncode != 0:
        continue
    command = ["/home/bachelor/zhangjinhang/molRL/zeo++-0.3/framework_builder", "/home/bachelor/zhangjinhang/molRL/zeo++-0.3/nets/hcb.cgd", "1", "output", "/home/bachelor/zhangjinhang/molRL/zeo++-0.3/builder_examples/building_blocks/C.xyz", "/home/bachelor/zhangjinhang/molRL/GCPNs/envs/atom-gym-molecule/gym_molecule/dataset/"+args.xyzname+"_1.xyz"]
    res = subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if res.returncode == 0:
        count+=1
        print(smiles)
        with open("./"+args.resname+".csv", 'a') as f:
            f.write(smiles + "\n")
    if i % 1000 == 0:
        print("{:2%}".format(i/dataset_len))
    os.remove("./"+args.xyzname+".xyz")
    os.remove("./"+args.xyzname+"_1.xyz")
print("pass zeo++filter count:"+str(count))