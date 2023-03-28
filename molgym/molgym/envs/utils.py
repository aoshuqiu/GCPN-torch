import csv
import os
import re
import copy
import itertools

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Descriptors import MolLogP, qed
import numpy as np
from torch_geometric.data import Data
import networkx as nx
from rdkit.rdBase import BlockLogs
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog

from molgym.envs.sascorer import calculateScore

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
            p=r"(\w+)\s+(-*\d+.\d+|\d+e-\d+)\s+(-*\d+.\d+|\d+e-\d+)\s+(-*\d+.\d+|\d+e-\d+)"
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

def reward_penalized_log_p(mol):
    """
    Reward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset
    :param mol: rdkit mol object
    :return: float
    """
    # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = MolLogP(mol)
    SA = -calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(
        Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle

def func_combine(funcs, weights):
    """
    Combine mutiple critic functions in to one function inner() with only one parameter ``mol``.

    :param funcs: Function list which need to be combined.
    :param weights: Weight for each function. 
    """
    def inner(mol):
        result = 0
        for i in range(len(funcs)):
            result += funcs[i](mol) * weights[i]
        return result
    return inner

def sa_reward(mol):
    """
    Get synthetic avaliblity reward for a mol

    :param mol: mol to be tested.
    :return: 0 to 1 higher for better synthetic avaliblity.
    """
    sa = -1 * calculateScore(mol)
    reward_sa = (sa + 10) / (10 - 1)
    return reward_sa


def reward_target_qed(mol, target,ratio=0.1,max=4):
    """
    Reward for a target log p
    :param mol: rdkit mol object
    :param target: float
    :return: float (-inf, max]
    """
    x = qed(mol)
    reward = -1 * np.abs((x - target)/ratio) + max
    return reward

def reward_target_new(mol, func,r_max1=4,r_max2=2.25,r_mid=2,r_min=-2,
                      x_start=500, x_mid=525):
    """
    Reward shaping for target reward.

    :param mol: molecule to be test.
    :param func: performance type.
    :param r_max1: max reward for equation 1, defaults to 4
    :param r_max2: max reward for equation 2, defaults to 2.25
    :param r_mid: middle number for reward, defaults to 2
    :param r_min: mini reward , defaults to -2
    :param x_start: mini performance scalar, defaults to 500
    :param x_mid: middle of performance scalar, defaults to 525
    """
    x = func(mol)
    return max((r_max1-r_mid)/(x_start-x_mid)*np.abs(x-x_mid)+r_max1, 
               (r_max2-r_mid)/(x_start-x_mid)*np.abs(x-x_mid)+r_max2,
                r_min)


def reward_valid(mol):
    """
    Reward base on molecule's validity, contains chemical validity, zinc_filter
    and steric strain filter.

    :param mol: molecule to be test.
    :return: max 2 if make through both filter, min -3 if not pass chemical validity.
    """
    reward_valid = 2
    if not check_chemical_validity(mol):
        return -3
    else:
        mol = convert_radical_electrons_to_hydrogens(mol)
        s = Chem.MolToSmiles(mol, isomericSmiles = True)
        mol = Chem.MolFromSmiles(s)
        if not steric_strain_filter(mol):
            reward_valid -= 1
        if not zinc_molecule_filter(mol):
            reward_valid -= 1
        return reward_valid




#检测化学有效性
def check_chemical_validity(mol):
    """
    Checks the chemical validity of the mol object. Existing mol object is
    not modified. Radicals pass this test.
    :return: True if chemically valid, False otherwise
    """
    s = Chem.MolToSmiles(mol, isomericSmiles=True)
    m = Chem.MolFromSmiles(s)  # implicitly performs sanitization
    if m:
        return True
    else:
        return False

#检测化合价
def check_valency(self):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    block = BlockLogs()
    try:
        Chem.SanitizeMol(self.mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        del block
        return True
    except ValueError:
        del block
        return False
    
### YES/NO filters ###
def zinc_molecule_filter(mol):
    """
    Flags molecules based on problematic functional groups as
    provided set of ZINC rules from
    http://blaster.docking.org/filtering/rules_default.txt.
    :param mol: rdkit mol object
    :return: Returns True if molecule is okay (ie does not match any of
    therules), False if otherwise
    """
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.ZINC)
    catalog = FilterCatalog(params)
    return not catalog.HasMatch(mol)


# TODO(Bowen): check 计算了一个角的能量限制，利用MMFF94力场，每个角的能量不能超过0.82，这实际是一种3维位置限制
def steric_strain_filter(mol, cutoff=0.82,
                         max_attempts_embed=20,
                         max_num_iters=200):
    """
    Flags molecules based on a steric energy cutoff after max_num_iters
    iterations of MMFF94 forcefield minimization. Cutoff is based on average
    angle bend strain energy of molecule
    :param mol: rdkit mol object
    :param cutoff: kcal/mol per angle . If minimized energy is above this
    threshold, then molecule fails the steric strain filter
    :param max_attempts_embed: number of attempts to generate initial 3d
    coordinates
    :param max_num_iters: number of iterations of forcefield minimization
    :return: True if molecule could be successfully minimized, and resulting
    energy is below cutoff, otherwise False
    """
    # check for the trivial cases of a single atom or only 2 atoms, in which
    # case there is no angle bend strain energy (as there are no angles!) 角弯曲应变能
    if mol.GetNumAtoms() <= 2:
        return True

    # make copy of input mol and add hydrogens
    m = copy.deepcopy(mol)
    m_h = Chem.AddHs(m)

    # generate an initial 3d conformer 生成初始化3d构象， 不行的说明不适合作为有效分子
    try:
        flag = AllChem.EmbedMolecule(m_h, maxAttempts=max_attempts_embed)
        if flag == -1:
            # print("Unable to generate 3d conformer")
            return False
    except: # to catch error caused by molecules such as C=[SH]1=C2OC21ON(N)OC(=O)NO
        # print("Unable to generate 3d conformer")
        return False

    # set up the forcefield
    AllChem.MMFFSanitizeMolecule(m_h)
    if AllChem.MMFFHasAllMoleculeParams(m_h):
        mmff_props = AllChem.MMFFGetMoleculeProperties(m_h)
        try:    # to deal with molecules such as CNN1NS23(=C4C5=C2C(=C53)N4Cl)S1
            ff = AllChem.MMFFGetMoleculeForceField(m_h, mmff_props)
        except:
            # print("Unable to get forcefield or sanitization error")
            return False
    else:
        # print("Unrecognized atom type")
        return False

    # minimize steric energy
    try:
        ff.Minimize(maxIts=max_num_iters)
    except:
        # print("Minimization error")
        return False


    # get the angle bend term contribution to the total molecule strain energy
    mmff_props.SetMMFFBondTerm(False)
    mmff_props.SetMMFFAngleTerm(True)
    mmff_props.SetMMFFStretchBendTerm(False)
    mmff_props.SetMMFFOopTerm(False)
    mmff_props.SetMMFFTorsionTerm(False)
    mmff_props.SetMMFFVdWTerm(False)
    mmff_props.SetMMFFEleTerm(False)

    ff = AllChem.MMFFGetMoleculeForceField(m_h, mmff_props)

    min_angle_e = ff.CalcEnergy()
    # print("Minimized angle bend energy: {}".format(min_angle_e))

    # find number of angles in molecule
    # TODO(Bowen): there must be a better way to get a list of all angles
    # from molecule... This is too hacky
    num_atoms = m_h.GetNumAtoms()
    atom_indices = range(num_atoms)
    angle_atom_triplets = itertools.permutations(atom_indices, 3)  # get all
    # possible 3 atom indices groups. Currently, each angle is represented by
    #  2 duplicate groups. Should remove duplicates here to be more efficient
    double_num_angles = 0
    for triplet in list(angle_atom_triplets):
        if mmff_props.GetMMFFAngleBendParams(m_h, *triplet):
            double_num_angles += 1
    num_angles = double_num_angles / 2  # account for duplicate angles

    avr_angle_e = min_angle_e / num_angles

    if avr_angle_e < cutoff:
        return True
    else:
        return False

# TODO(Bowen): check, esp if input is not radical 把自由基电子转化为氢
def convert_radical_electrons_to_hydrogens(mol):
    """
    Converts radical electrons in a molecule into bonds to hydrogens. Only
    use this if molecule is valid. Results a new mol object
    :param mol: rdkit mol object
    :return: rdkit mol object
    """
    m = copy.deepcopy(mol)
    if Chem.Descriptors.NumRadicalElectrons(m) == 0:  # not a radical 没有自由基
        return m
    else:  # a radical 原子有自由基
        for a in m.GetAtoms():
            num_radical_e = a.GetNumRadicalElectrons()
            if num_radical_e > 0:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(num_radical_e) #设置显性氢原子
    return m


def load_conditional(type='low'):
    """
    Load base fragment for conditional generation ,or to say 
    let the agent grows from a small molecule.
    :param type: low for low plogp, defaults to 'low'.
    :return: loaded molecule SMILES.
    """
    if type=='low':
        cwd = os.path.dirname(__file__)
        path = os.path.join(os.path.dirname(cwd), 'dataset',
                            'opt.test.logP-SA')
        import csv
        with open(path, 'r') as fp:
            reader = csv.reader(fp, delimiter=' ', quotechar='"')
            data = [row+[id] for id,row in enumerate(reader)]
        # print(len(data))
        # print(data[799])
    elif type=='high':
        cwd = os.path.dirname(__file__)
        path = os.path.join(os.path.dirname(cwd),'dataset',
                            'zinc_plogp_sorted.csv')
        import csv
        with open(path, 'r') as fp:
            reader = csv.reader(fp, delimiter=',', quotechar='"')
            data = [[row[1], row[0],id] for id, row in enumerate(reader)]
            # data = [row for id, row in enumerate(reader)]
            data = data[0:800]
    return data

def load_scaffold():
    """
    Load molecule fragment as generatation unit.

    :return: molecule fragments.
    """
    cwd = os.path.dirname(__file__) #返回文件路径
    path = os.path.join(os.path.dirname(cwd), 'dataset',
                       'vocab.txt')  # gdb 13 合成路径
    with open(path, 'r') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        data = [Chem.MolFromSmiles(row[0]) for row in reader]
        data = [mol for mol in data if mol.GetRingInfo().NumRings() == 1 and (mol.GetRingInfo().IsAtomInRingOfSize(0, 5) or mol.GetRingInfo().IsAtomInRingOfSize(0, 6))] #有一个环，0号原子在5元或6元环上，的分子
        for mol in data:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE) #单双键恢复成芳香键
        print('num of scaffolds:', len(data)) #脚手架数量
        return data