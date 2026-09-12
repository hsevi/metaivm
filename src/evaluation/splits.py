"""
Dataset-level train/val/test splitting for Neural IVM.

Critical: splits are at the dataset level, not the run level.
No dataset appears in more than one partition per split.
"""

import numpy as np


def make_splits(dataset_ids, n_splits=5, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=None):
    """
    Create dataset-level train/val/test splits.

    Args:
        dataset_ids: List of dataset identifiers.
        n_splits: Number of different random splits to generate.
        train_frac: Fraction of datasets for training.
        val_frac: Fraction of datasets for validation.
        test_frac: Fraction of datasets for testing.
        seed: Base random seed (each split uses seed + i).

    Returns:
        List of (train_ids, val_ids, test_ids) tuples.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "Fractions must sum to 1"

    dataset_ids = list(dataset_ids)
    n = len(dataset_ids)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    # Remainder goes to test, but ensure at least 1
    n_test = n - n_train - n_val
    if n_test < 1:
        # Redistribute: take from the largest partition
        n_train = max(1, n - 2)
        n_val = 1
        n_test = n - n_train - n_val

    assert n_train > 0 and n_val > 0 and n_test > 0, \
        f"Need at least 3 datasets for splitting (got n={n})"

    splits = []
    for i in range(n_splits):
        s = seed + i if seed is not None else i
        rng = np.random.default_rng(s)
        perm = rng.permutation(n)

        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]

        train_ids = [dataset_ids[j] for j in train_idx]
        val_ids = [dataset_ids[j] for j in val_idx]
        test_ids = [dataset_ids[j] for j in test_idx]

        splits.append((train_ids, val_ids, test_ids))

    return splits
