import unittest
from pathlib import Path
import torch
from torch_geometric.data import Data
from train_gnn_qm9 import CustomTransform, GNNModel

class GNNTests(unittest.TestCase):
    def test_single_atom_features_are_finite(self):
        data = Data(z=torch.tensor([1]), pos=torch.zeros((1,3)), edge_index=torch.empty((2,0), dtype=torch.long), y=torch.zeros((1,19)))
        result = CustomTransform(4)(data)
        self.assertEqual(tuple(result.x.shape), (1,8))
        self.assertEqual(tuple(result.edge_attr.shape), (0,4))
        self.assertTrue(torch.isfinite(result.x).all())

    def test_original_checkpoint_shape_and_inference(self):
        model = GNNModel()
        checkpoint = Path(__file__).resolve().parents[1] / 'gnn_qm9_homo_lumo.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        data = Data(z=torch.tensor([1,6]), pos=torch.tensor([[0.,0.,0.],[1.,0.,0.]]), edge_index=torch.tensor([[0,1],[1,0]]), y=torch.zeros((1,19)))
        graph = CustomTransform(4)(data)
        with torch.no_grad():
            prediction = model(graph.x, graph.edge_index, graph.edge_attr, torch.zeros(2,dtype=torch.long))
        self.assertEqual(tuple(prediction.shape),(1,1))
        self.assertTrue(torch.isfinite(prediction).all())

if __name__ == '__main__':
    unittest.main()
