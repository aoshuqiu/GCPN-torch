import torch
from torch import nn

class GCN(nn.Module):
    __constant__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    edge_num: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, edge_num: int, need_normalize=False, orthogonal_init=False) -> None:
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_num = edge_num
        self.weight = nn.Parameter(torch.Tensor(edge_num, in_features, out_features))
        self.orthogonal_init = orthogonal_init
        self.reset_parameters()
        self.need_normalize = need_normalize
        
    def reset_parameters(self) -> None:
        if self.orthogonal_init:
            nn.init.orthogonal_(self.weight)
        else:
            nn.init.xavier_uniform_(self.weight)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: B * E * A * in_F     weight: E * in_F * out_F
        .. math:: H^{(l+1)}=AGG(ReLU(\{\tilde{D}_i^{-\frac{1}{2}}\tilde{E}_i\tilde{D}_i^{-\frac{1}{2}}H^{(l)}W^{(l)}_i\}, \forall i \in (1,...,b))) 
        :param input: _description_
        :return: _description_
        """
        # weight: E * in_F * out_F     input:B * E * A * in_F
        node_embedding = torch.mean(torch.matmul(input, self.weight), 1).unsqueeze(1)
        if self.need_normalize:
            node_embedding = torch.nn.functional.normalize(node_embedding, dim=-1, p=2)
        return node_embedding
    
