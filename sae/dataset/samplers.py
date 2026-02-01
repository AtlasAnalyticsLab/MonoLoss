"""Custom samplers for efficient LMDB data loading."""

import torch
import random
from torch.utils.data import Sampler
from typing import Iterator, List


class ContiguousBatchSampler(Sampler[List[int]]):
    """
    Yields batches of contiguous indices, shuffled at the batch level.

    Much faster for LMDB than random access because:
    - Sequential reads benefit from OS prefetching
    - Fewer disk seeks
    - Better cache utilization

    Args:
        n_samples: Total number of samples in dataset
        batch_size: Number of samples per batch
        drop_last: If True, drop the last incomplete batch
        shuffle: If True, shuffle batch order each epoch
    """
    def __init__(self, n_samples: int, batch_size: int, drop_last: bool = False, shuffle: bool = True):
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.num_batches = n_samples // batch_size
        self.remainder = n_samples % batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[List[int]]:
        # Get batch indices
        if self.shuffle:
            batch_order = torch.randperm(self.num_batches).tolist()
        else:
            batch_order = list(range(self.num_batches))

        # Yield contiguous batches in shuffled order
        for b in batch_order:
            start = b * self.batch_size
            yield list(range(start, start + self.batch_size))

        # Handle remainder
        if not self.drop_last and self.remainder > 0:
            tail = list(range(self.num_batches * self.batch_size, self.n_samples))
            if self.shuffle:
                random.shuffle(tail)
            yield tail

    def __len__(self) -> int:
        if self.drop_last or self.remainder == 0:
            return self.num_batches
        return self.num_batches + 1
