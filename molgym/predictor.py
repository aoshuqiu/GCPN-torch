from abc import ABCMeta, abstractmethod
import torch

class Predictor:
    """
    Base interface for molecule performance predictor such as homo, lumo, etc.
    """

    def __init__(self, model:torch.nn.Module, device:torch.device, save_file_str:str):
        """
        :param model: predictor's underlying torch model
        :param device: device to transfer model to
        :param save_file_str: file str for load state dict.
        """
        self.model = model
        self.device = device
        self.save_file_str = save_file_str

    def _load(self):
        self.model.load_state_dict(torch.load(self.save_file_str, map_location=self.device))

    def _model_to_device(self):
        self.model.to(self.device)



    def predict(self, mol):
        """
        Predict molecule performance.

        :param mol: molecule which need to be test
        """
        raise NotImplementedError('Implement me')
    
    @staticmethod
    @abstractmethod
    def factory() -> 'PredictorFactory':
        raise NotImplementedError('Implement me')
        

class PredictorFactory:
    def create(self) -> Predictor:
        raise NotImplementedError('Implement me')
