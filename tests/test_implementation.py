import unittest
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from metrics.cad_prune import calculate_el2n, compute_cad_prune_scores, prune_dataset
from metrics.dcs import compute_dcs
from methods.ca2d import get_patches, stitch_patches, score_patches

class MockTeacher(torch.nn.Module):
    def forward(self, x):
        # Return dummy logits [batch, 2]
        return torch.randn(x.shape[0], 2)

class TestMetrics(unittest.TestCase):
    def test_el2n(self):
        outputs = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([0, 1])
        scores = calculate_el2n(outputs, targets)
        # Probabilities are [0.8, 0.2] and [0.3, 0.7]
        # Targets one-hot: [1, 0] and [0, 1]
        # Diff: [-0.2, 0.2] and [0.3, -0.3]
        # L2 norm: sqrt(0.04 + 0.04) = 0.2828 and sqrt(0.09 + 0.09) = 0.4243
        self.assertAlmostEqual(scores[0].item(), 0.2828, places=4)
        self.assertAlmostEqual(scores[1].item(), 0.4243, places=4)

    def test_cad_prune_scores(self):
        # K=10, J=3, W=2
        # k in [10-3-2, 10-3-1] => [5, 6]
        # uncertainties for k=5 (scores 5,6,7) and k=6 (scores 6,7,8)
        K, J, W = 10, 3, 2
        N = 5
        epoch_scores = torch.randn(K, N)
        scores = compute_cad_prune_scores(epoch_scores, K, J, W)
        self.assertEqual(scores.shape, (N,))

    def test_dcs(self):
        test_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
        dist_losses = [0.5, 0.4, 0.3, 0.2, 0.1] # Perfect correlation
        score = compute_dcs(test_losses, dist_losses)
        self.assertAlmostEqual(score, 1.0)

        dist_losses_inv = [0.1, 0.2, 0.3, 0.4, 0.5] # Perfect inverse correlation
        score_inv = compute_dcs(test_losses, dist_losses_inv)
        self.assertAlmostEqual(score_inv, -1.0)

class TestMethods(unittest.TestCase):
    def test_patches(self):
        images = torch.randn(2, 3, 32, 32)
        patch_size = 8
        patches = get_patches(images, patch_size)
        # 32/8 = 4 patches per side => 16 patches
        self.assertEqual(patches.shape, (2, 16, 3, 8, 8))

    def test_stitch(self):
        patch_size = 8
        grid_size = 4
        num_patches = grid_size * grid_size
        selected_patches = torch.randn(2 * num_patches, 3, patch_size, patch_size)
        distilled = stitch_patches(selected_patches, 2, patch_size, grid_size)
        self.assertEqual(distilled.shape, (2, 3, 32, 32))

    def test_score_patches_batched(self):
        # Test that score_patches works with batching
        patches = torch.randn(2, 4, 3, 8, 8)
        teacher = MockTeacher()
        targets = torch.tensor([0, 1])
        scores = score_patches(patches, teacher, targets, batch_size_eval=3)
        self.assertEqual(scores.shape, (2, 4))

if __name__ == '__main__':
    unittest.main()
