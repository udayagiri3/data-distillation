import numpy as np
from scipy.stats import spearmanr
import torch

def compute_dcs(test_losses, distillation_losses):
    """
    Compute the Distillation Correlation Score (DCS) for a given DD objective.

    Args:
        test_losses (list or np.array): List of downstream test losses for a set of subsets.
        distillation_losses (list or np.array): List of distillation objective losses for the same subsets.

    Returns:
        float: Spearman-rank correlation between test losses and distillation losses.
    """
    # Convert to numpy arrays if they are lists or tensors
    if isinstance(test_losses, torch.Tensor):
        test_losses = test_losses.cpu().numpy()
    if isinstance(distillation_losses, torch.Tensor):
        distillation_losses = distillation_losses.cpu().numpy()

    test_losses = np.array(test_losses)
    distillation_losses = np.array(distillation_losses)

    # Calculate Spearman rank correlation
    correlation, p_value = spearmanr(test_losses, distillation_losses)

    return correlation

def evaluate_objective_scalability(subsets_info, distillation_objective_fn, teacher_model):
    """
    A helper function to evaluate the scalability of a distillation objective.

    Args:
        subsets_info (list of dict): A list of dictionaries, each containing:
            - 'subset': The synthetic/coreset subset (torch.Tensor).
            - 'test_loss': The generalization error of a model trained on this subset.
        distillation_objective_fn (callable): A function that takes a subset and a teacher model,
                                             and returns the distillation loss.
        teacher_model: The teacher model used for distillation.

    Returns:
        float: The DCS score for the objective.
    """
    test_losses = []
    dist_losses = []

    for info in subsets_info:
        subset = info['subset']
        test_loss = info['test_loss']

        # Calculate the distillation loss for this subset
        dist_loss = distillation_objective_fn(subset, teacher_model)

        test_losses.append(test_loss)
        dist_losses.append(dist_loss)

    return compute_dcs(test_losses, dist_losses)

# Example of a simple distillation objective: BatchNorm statistics matching (like SRe2L)
def bn_matching_objective(subset, teacher_model):
    """
    Simplified BN matching objective.
    """
    teacher_model.eval()

    # We'd need to hook into the teacher model to get BN stats during a forward pass
    # For this implementation, we'll assume a simplified version that returns a scalar.
    # In a real scenario, this would involve comparing subset BN stats with teacher's saved stats.

    # Placeholder for actual implementation
    with torch.no_grad():
        # Suppose the objective is the MSE between some feature means
        # This is just for demonstration of how DCS would be used.
        return torch.rand(1).item()
