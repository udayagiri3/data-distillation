import torch
import torch.nn.functional as F

def get_patches(images, patch_size):
    """
    Divide images into non-overlapping patches.
    """
    batch_size, channels, height, width = images.shape
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # patches shape: (B, C, H_p, W_p, patch_size, patch_size)
    patches = patches.contiguous().view(batch_size, channels, -1, patch_size, patch_size)
    # patches shape: (B, C, num_patches, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4)
    # patches shape: (B, num_patches, C, patch_size, patch_size)
    return patches

def score_patches(patches, teacher_model, targets, batch_size_eval=128):
    """
    Score patches based on their 'impact' (e.g., loss or confidence).
    In RDED/CA2D, this is often the loss reduction or confidence of the patch.
    """
    batch_size, num_patches, channels, h, w = patches.shape
    flat_patches = patches.view(-1, channels, h, w)
    num_total_patches = flat_patches.shape[0]

    patch_confidences = []

    with torch.no_grad():
        # Expand targets to match patches
        expanded_targets = targets.repeat_interleave(num_patches)

        # Process patches in batches to avoid OOM
        for i in range(0, num_total_patches, batch_size_eval):
            batch_patches = flat_patches[i : i + batch_size_eval]
            batch_targets = expanded_targets[i : i + batch_size_eval]

            outputs = teacher_model(batch_patches)
            probs = F.softmax(outputs, dim=1)

            conf = probs[range(len(batch_targets)), batch_targets]
            patch_confidences.append(conf)

    patch_confidences = torch.cat(patch_confidences)
    return patch_confidences.view(batch_size, num_patches)

def stitch_patches(selected_patches, images_per_distilled, patch_size, grid_size):
    """
    Stitch selected patches into a single distilled image.
    """
    # selected_patches: (images_per_distilled * num_patches_needed, C, patch_size, patch_size)
    channels = selected_patches.shape[1]

    distilled_images = []
    patches_per_image = grid_size * grid_size

    for i in range(0, len(selected_patches), patches_per_image):
        image_patches = selected_patches[i:i+patches_per_image]
        # Reshape to (grid_size, grid_size, C, patch_size, patch_size)
        image_patches = image_patches.view(grid_size, grid_size, channels, patch_size, patch_size)
        # Permute and reshape to (C, grid_size * patch_size, grid_size * patch_size)
        image = image_patches.permute(2, 0, 3, 1, 4).contiguous()
        image = image.view(channels, grid_size * patch_size, grid_size * patch_size)
        distilled_images.append(image)

    return torch.stack(distilled_images)

def ca2d_distillation(dataset_images, dataset_labels, cad_scores, teacher_model,
                       ipc, patch_size, grid_size, num_classes):
    """
    CA2D (Compute-Aware Dataset Distillation) main loop.
    """
    distilled_dataset = []
    distilled_labels = []

    final_image_size = patch_size * grid_size

    for c in range(num_classes):
        class_indices = torch.where(dataset_labels == c)[0]
        class_images = dataset_images[class_indices]
        class_scores = cad_scores[class_indices]

        # 1. Identify samples of suitable difficulty using CAD-Prune
        # We pick the top IPC * (something) candidates to extract patches from
        # Or more simply, use CAD-Prune to rank candidates.
        # Paper says: "selects confident patches from examples of suitable difficulty"

        num_candidates = min(len(class_indices), ipc * 5) # Heuristic
        _, candidate_idx = torch.topk(class_scores, num_candidates)

        candidate_images = class_images[candidate_idx]
        candidate_labels = dataset_labels[class_indices[candidate_idx]]

        # 2. Extract patches from candidates
        all_patches = get_patches(candidate_images, patch_size)
        # all_patches: (num_candidates, patches_per_source_image, C, P, P)

        # 3. Score patches
        patch_scores = score_patches(all_patches, teacher_model, candidate_labels)
        # patch_scores: (num_candidates, patches_per_source_image)

        # 4. Select top patches across all candidates in this class
        flat_patches = all_patches.view(-1, 3, patch_size, patch_size)
        flat_scores = patch_scores.view(-1)

        patches_needed = ipc * (grid_size * grid_size)
        _, top_patch_indices = torch.topk(flat_scores, patches_needed)
        selected_patches = flat_patches[top_patch_indices]

        # 5. Stitch patches into distilled images
        distilled_images = stitch_patches(selected_patches, ipc, patch_size, grid_size)

        distilled_dataset.append(distilled_images)
        distilled_labels.append(torch.full((ipc,), c, dtype=torch.long))

    return torch.cat(distilled_dataset), torch.cat(distilled_labels)
