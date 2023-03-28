from typing import Dict, Tuple

import numpy as np
import gym
from gym.spaces import Space

class MolecularObDictSpace(Space):
    def __init__(self, max_atom_num, possible_atom_num, edge_type_num, node_feature_num):
        self.max_atom_num = max_atom_num
        self.possible_atom_num = possible_atom_num
        self.edge_type_num = edge_type_num
        self._shape = {}
        self._shape["node"] = (1, self.max_atom_num, node_feature_num)
        self._shape["adj"] = (self.edge_type_num, self.max_atom_num, self.max_atom_num)
        self.dtype = np.int32

    def sample(self) -> Dict[str, np.ndarray]:
        """
        Sample a random molecular graph from the space.
        :return: a random molecular graph.
                 node: (1, max_atom_num, node_feature_num) for the node feature.
                 adj: (max_atom_num, max_atom_num, edge_type_num) for the adjacency matrix.
        """
        node = np.zeros(self.shape["node"], dtype=self.dtype)
        rand_node_idx = np.random.randint(self.possible_atom_num, size=self.max_atom_num)
        node[:,:,:] = np.eye(self.possible_atom_num)[rand_node_idx]
        adj = np.zeros(self.shape["adj"], dtype=self.dtype)
        rand_adj_idx = np.random.randint(self.edge_type_num, size=(self.max_atom_num, self.max_atom_num))
        for i in range(self.max_atom_num):
            for j in range(i):
                adj[rand_adj_idx[i,j], i, j] = 1
                adj[rand_adj_idx[i,j], j, i] = 1
        return {"node": node, "adj": adj}
    
    def contains(self, x: Dict[str, np.ndarray]) -> bool:
        """
        Check if a molecular graph is valid.
        :param x: the molecular graph to be checked.
        :return: True if the molecular graph is valid, False otherwise.
        """
        if not isinstance(x, dict):
            return False
        if "node" not in x:
            return False
        if "adj" not in x:
            return False
        if x["node"].shape != self.shape["node"]:
            return False
        if x["adj"].shape != self.shape["adj"]:
            return False
        if x["node"].dtype != self.dtype:
            return False
        if x["adj"].dtype != self.dtype:
            return False
        return True
    
    def __repr__(self) -> str:
        return "MolecularObDictSpace({})".format(self.shape)
    
    def __eq__(self, other) -> bool:
        return np.array_equal(self.shape, other.shape)
    


class MolecularActionTupleSpace(Space):
    def __init__(self, max_atom_num, possible_atom_num, edge_type_num):
        self.max_atom_num = max_atom_num
        self.edge_type_num = edge_type_num
        self.possible_atom_num = possible_atom_num
        self._shape = (4,)
        self.dtype = np.int32

    def sample(self) -> Tuple[int,int,int,int]:
        """
        Sample a random action from the space.
        :return: a random action.
                 0: whether to stop.
                 1: choose a node from the current graph.
                 2: choose the other node to add an edge from the whole graph.
                 3: choose the edge type between the two nodes.
        """
        return (np.random.randint(2), 
                np.random.randint(self.max_atom_num-self.possible_atom_num), 
                np.random.randint(self.max_atom_num), 
                np.random.randint(self.edge_type_num))

    def contains(self, x: Tuple[int,int,int,int]) -> bool:
        """
        Check if an action is valid.
        :param x: the action to be checked.
        :return: True if the action is valid, False otherwise.
        """
        if not isinstance(x, tuple):
            return False
        if len(x) != 4:
            return False
        if x[0] not in [0, 1]:
            return False
        if x[1] not in range(self.max_atom_num-self.possible_atom_num):
            return False
        if x[2] not in range(self.max_atom_num):
            return False
        if x[3] not in range(self.edge_type_num):
            return False
        return True
    
    def __repr__(self) -> str:
        return "MolecularActionTupleSpace({})".format(self.shape)
    
    def __eq__(self, other) -> bool:
        return np.array_equal(self.shape, other.shape)
