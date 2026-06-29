import torch
import numpy as np

def calculate_el2n(outputs, targets):
    """
    Calculate the EL2N (Error L2-Norm) score for a batch of samples.

    Args:
        outputs (torch.Tensor): Model outputs (logits or probabilities).
        targets (torch.Tensor): Ground truth labels (one-hot or indices).

    Returns:
        torch.Tensor: EL2N scores.
    """
    if len(targets.shape) == 1:
        # Convert indices to one-hot
        num_classes = outputs.shape[-1]
        targets_one_hot = torch.zeros_like(outputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
    else:
        targets_one_hot = targets

    # If outputs are logits, apply softmax
    if not torch.allclose(outputs.sum(dim=1), torch.ones(outputs.shape[0], device=outputs.device)):
        probs = torch.nn.functional.softmax(outputs, dim=1)
    else:
        probs = outputs

    return torch.norm(probs - targets_one_hot, p=2, dim=1)

def compute_cad_prune_scores(epoch_scores, K, J=6, W=2):
    """
    Compute CAD-Prune scores based on per-epoch sample scores.

    Args:
        epoch_scores (torch.Tensor): Tensor of shape (K, N) where K is total epochs
                                     and N is number of samples.
        K (int): Total compute budget (epochs).
        J (int): Range for calculating uncertainty (standard deviation).
        W (int): Window size to average uncertainty over.

    Returns:
        torch.Tensor: CAD-Prune scores for each sample.
    """
    # U_k(x) calculation
    # U_k(x) = std(scores from k to k+J-1)

    # We need uncertainty scores for k in [K-J-W, K-J-1]
    # For a given k, we use scores from [k, k+J-1]
    # The maximum index we need is (K-J-1) + J - 1 = K-2.

    uncertainties = []
    for k in range(K - J - W, K - J):
        # Slice scores for the range [k, k+J-1]
        window_scores = epoch_scores[k : k + J, :]
        # Calculate standard deviation along the epoch dimension
        u_k = torch.std(window_scores, dim=0, unbiased=True)
        uncertainties.append(u_k)

    # Average uncertainties over the window W
    cad_scores = torch.stack(uncertainties).mean(dim=0)
    return cad_scores

def prune_dataset(cad_scores, ipc, num_classes, labels):
    """
    Prune the dataset to keep 'ipc' samples per class based on CAD-Prune scores.

    Args:
        cad_scores (torch.Tensor): CAD-Prune scores for all samples.
        ipc (int): Images per class to keep.
        num_classes (int): Number of classes.
        labels (torch.Tensor): Labels for all samples.

    Returns:
        torch.Tensor: Indices of selected samples.
    """
    selected_indices = []
    for c in range(num_classes):
        class_indices = torch.where(labels == c)[0]
        class_scores = cad_scores[class_indices]

        # Select samples with top CAD-Prune scores (highest uncertainty at the end of training)
        # Note: The paper says "identify samples that correspond to an appropriate difficulty level"
        # and "selects ... from examples of suitable difficulty".
        # Typically "optimal difficulty" for a given compute.
        # Based on Fig 2(c), HL has an "optimal hardness".
        # CA2D "selects confident patches from examples of suitable difficulty".

        # For CAD-Prune as a coreset method, we'll sort and pick the top ones.
        _, top_idx = torch.topk(class_scores, min(ipc, len(class_indices)))
        selected_indices.append(class_indices[top_idx])

    return torch.cat(selected_indices)
