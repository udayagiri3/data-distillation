# Future Improvements for Data Distillation

Based on the paper "Rethinking Dataset Distillation: Hard Truths About Soft Labels", several avenues for future research and improvement are identified:

## 1. Extension to Multimodal and Other Tasks
The current study primarily focuses on image classification using ImageNet-1K and TinyImageNet. Future work could explore the application of CAD-Prune and CA2D to:
- **Multimodal tasks** (e.g., Vision-Language models, CLIP-style training).
- **Object Detection and Segmentation**: Investigating if patch-based distillation (like CA2D) generalizes to tasks requiring localization.
- **Natural Language Processing**: Applying compute-aware pruning to large language model pre-training or fine-tuning datasets.

## 2. Refinement of DCS for Non-Optimizable Objectives
The Distillation Correlation Score (DCS) is a powerful tool for evaluating distillation objectives. However, expressing non-optimizable or complex DD objectives (like those involving complicated matching heuristics) in a way that allows for DCS evaluation remains an open problem. Developing a more flexible framework for DCS could enable the evaluation of a wider range of methods.

## 3. Adaptive Windowing for CAD-Prune
In CAD-Prune, the window size `W` and range `J` are currently fixed (e.g., `W=2`, `J=6` for ImageNet-1K). An improvement could be to **adaptively determine these hyperparameters** based on the learning rate schedule or the convergence characteristics of the compute-matched run.

## 4. Combinatorial Optimization of Patch Selection
CA2D currently selects the top patches independently. A potential improvement is to use **combinatorial optimization or submodular maximization** to select a set of patches that collectively maximize diversity and coverage of the feature space, rather than just individual importance.

## 5. Investigation of Model-Stealing Scenarios
The paper suggests that under extensive teacher supervision (SL+KD), dataset quality and size have limited impact. This has significant implications for **model stealing**. Future work could systematically evaluate the vulnerability of different models to stealing attacks using varied distilled datasets under SL+KD.

## 6. Real-time Compute-Aware Distillation
Developing a version of CA2D that can **dynamically adjust the distilled set** as the student model trains, effectively providing a curriculum that stays aligned with the student's current "learnability" level.
