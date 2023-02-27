import re

import numpy as np
from torch_geometric.data import Data

def get_atoms_type_and_xyz(filename):
    """
    :param filename: input molecule xyz file.
    :return: molecule's atom types and coordinates.
    """
    # 获取指定文件（xyz）中的分子各个原子的类型和坐标
    dataset=list()
    index=0
    with open(filename,'r') as fp:
        data=fp.read()
    with open(filename,'r') as fp:
        for line in fp.readlines():
            chem_table = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                  'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 
                  'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Te', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 
                  'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 
                  'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 
                  'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm','Md', 'No', 'Lr',
                  'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Uue']
                #   化学元素周期表，我们用元素的编号标识他的元素。
            p="(\w+)\s+(-*\d+.\d+|\d+e-\d+)\s+(-*\d+.\d+|\d+e-\d+)\s+(-*\d+.\d+|\d+e-\d+)"
            # 正则式匹配
            s=re.search(p,line)
            # 搜寻所有的x,y,z坐标并保存
            data=[]
            if s:
                k=0
                for i in s.groups():
                    if k==0:
                        k+=1
                        data.append(float(1+chem_table.index(i)))
                    else:
                        data.append(float(float(i)))
                        k+=1
            if len(data)>0:
                dataset.append(np.array(data))
    return np.array(dataset)