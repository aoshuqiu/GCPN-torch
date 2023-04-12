from typing import List

import torch
from torch.distributions import Distribution, Categorical

class MultiCategorical(Distribution):
    def __init__(self, logits: torch.Tensor, split_list: List[int], validate_args=False):
        self.logits = logits
        self._param = self.logits
        self.split_list = split_list
        self.device = logits.device
        batch_shape = self._param.size()[:-1] if self._param.ndimension()>1 else torch.Size()
        self.logits_list = self.logits.split(self.split_list, dim=-1)
        super().__init__(batch_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        tensor_list = []
        for logit in self.logits_list:
            tensor_list.append(Categorical(logits=logit).sample(sample_shape))
        return torch.stack(tensor_list, dim=-1)
    
    def log_prob(self, value):
        if not isinstance(value, torch.Tensor) or value.device!=self.device:
            value = torch.tensor(value, device=self.device) 
        logp = torch.zeros(value.shape[0], device=self.device)
        values = value.split([1]*value.shape[-1],dim=-1)
        # dubug
        cnt = 0
        for logit, value in zip(self.logits_list, values):
            cnt+=1
            value = value.squeeze(-1)
            logp += Categorical(logits=logit).log_prob(value)
            # if torch.any(logp <= -20):
            #     print(f"logp<=-5 due to action[{cnt}]")
            #     for l, v in zip(logit, value):
            #         print("logit: ", l[int(v.item())])
            #     input()
                
        return logp
    
    def entropy(self):
        entropy = 0
        for logit in self.logits_list:
            entropy += Categorical(logits=logit).entropy()
        return entropy

            

