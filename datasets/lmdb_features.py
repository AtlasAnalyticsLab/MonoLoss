"""Generic LMDB feature dataset for pre-extracted embeddings."""

import lmdb
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Union
import torch

class LMDBFeatureDataset(Dataset):
    """Fast LMDB dataset using raw bytes (no pickle overhead) for vision features."""
    
    def __init__(self, lmdb_path: str, return_index: bool = False, verbose: bool = True):
        """
        Args:
            lmdb_path: Path to the LMDB directory.
            return_index: If True, __getitem__ returns (features, index).
            verbose: If True, print loading information.
        """
        self.lmdb_path = lmdb_path
        self.return_index = return_index
        
        # Open with optimized flags for read-only access
        self.env = lmdb.open(
            lmdb_path, 
            readonly=True, 
            lock=False,          # No locking for read-only
            readahead=False,     # Better for random access
            meminit=False        # Don't initialize memory
        )
        
        # Read metadata
        with self.env.begin() as txn:
            self.length = int(txn.get(b'__len__').decode('ascii'))
            shape_str = txn.get(b'__shape__').decode('ascii')
            self.feat_shape = eval(shape_str)
            self.feature_dim = self.feat_shape[-1] if len(self.feat_shape) > 0 else 0
        
        if verbose:
            print(f"Loading features from: {lmdb_path}")
            print(f"  Loaded {self.length} samples with shape {self.feat_shape}")
            print(f"=> Dataset has {self.length} samples.")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
        with self.env.begin() as txn:
            data = txn.get(str(idx).encode('ascii'))
            x = np.frombuffer(data, dtype=np.float32).reshape(self.feat_shape)
        
        # Normalize feature vector to unit norm
        norm = np.linalg.norm(x)
        if norm > 0:
            x = x / (norm + 1e-8)
        
        if self.return_index:
            return x, idx
        return x
    
    def get_feature_dim(self) -> int:
        """Return the feature dimension."""
        return self.feature_dim
    
    def close(self):
        """Close the LMDB environment."""
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
            self.env = None
    
    def __del__(self):
        self.close()

