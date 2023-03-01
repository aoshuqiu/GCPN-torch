import torch
from torch_geometric.data import Data
from dig.threedgraph.method import SphereNet

from molgym.predictor import Predictor, PredictorFactory
from molgym.utils import get_atoms_type_and_xyz


class HomoPredictor(Predictor):
    
    def __call__(self, mol):
        """
        Predict homo scalar of molecule xyz file.

        :param mol: file path for input molecule xyz file.
        :return: homo of imput molecule.
        """
        self._load()
        self._model_to_device()
        self.model.eval()
        with torch.no_grad():
            return self.get_homo_of_a_COF(mol)

    def get_homo_of_a_COF(self, filename):
        """
        :param filename: Filename of a molecule xyz file.
        :return: Homo scalar for input molecule xyz file.
        """
        print("begin:"+filename)
        posdata= get_atoms_type_and_xyz(filename)
                # 从npy文件中读取数据

        pos=torch.tensor(posdata[:,1:] ,dtype=torch.float32)
        z=torch.tensor(posdata[:,0],dtype=torch.int64)
                
        data = Data(pos=pos,z=z,batch=torch.zeros_like(z))
        data.to(self.device)
        out = self.model(data)
        return out.double().item()
    
    @staticmethod
    def factory(energy_and_force=False, 
                cutoff=5.0, num_layers=4, hidden_channels=128, out_channels=1, int_emb_size=64,
                basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, 
                out_emb_channels=256, num_spherical=3, num_radial=6, envelope_exponent=5,
                num_before_skip=1, num_after_skip=2, num_output_layers=3):
        """
        Get predictorFactory follow Zican's Attribute.
        """
        return HomoPredictorFactory(energy_and_force, cutoff, num_layers, hidden_channels, out_channels, 
                               int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion, 
                               out_emb_channels, num_spherical, num_radial, envelope_exponent, num_before_skip, 
                               num_after_skip, num_output_layers)
        
    
class HomoPredictorFactory(PredictorFactory):

    def __init__(self,energy_and_force=False, 
                 cutoff=5.0, num_layers=4, hidden_channels=128, out_channels=1, int_emb_size=64,
                 basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, 
                 out_emb_channels=256, num_spherical=3, num_radial=6, envelope_exponent=5,
                 num_before_skip=1, num_after_skip=2, num_output_layers=3):
        super().__init__()
        self.model = SphereNet(energy_and_force, cutoff, num_layers, hidden_channels, out_channels, 
                               int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion, 
                               out_emb_channels, num_spherical, num_radial, envelope_exponent, num_before_skip, 
                               num_after_skip, num_output_layers)
    
    def create(self, save_file_str, device=torch.device('cuda:1')) -> HomoPredictor:
        return HomoPredictor(self.model, device, save_file_str)
        
