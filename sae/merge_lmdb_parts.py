#!/usr/bin/env python3
"""
Merge multiple LMDB part files into a single LMDB database.

Usage:
    python scripts/merge_lmdb_parts.py \
        --input_dir /path/to/features/dataset_name \
        --pattern "pythia_410m_deduped_L11_post_mean_part*.lmdb" \
        --output pythia_410m_deduped_L11_post_mean.lmdb
"""

import os
import argparse
import glob
import lmdb
import numpy as np
from tqdm import tqdm


def get_lmdb_info(lmdb_path: str) -> tuple[int, tuple]:
    """Get length and feature shape from an LMDB database."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        length = int(txn.get(b'__len__').decode('ascii'))
        shape = eval(txn.get(b'__shape__').decode('ascii'))
    env.close()
    return length, shape


def iter_lmdb(lmdb_path: str, length: int):
    """Iterate over all features in an LMDB database."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        for i in range(length):
            key = str(i).encode('ascii')
            value = txn.get(key)
            if value is not None:
                yield value
    env.close()


def merge_lmdb_files(input_paths: list[str], output_path: str, map_size_gb: int = 300):
    """Merge multiple LMDB files into one."""

    # Sort input paths to ensure correct order (part1, part2, ...)
    input_paths = sorted(input_paths)

    print(f"Merging {len(input_paths)} LMDB files:")
    for p in input_paths:
        print(f"  - {p}")

    # Get info from all parts
    total_samples = 0
    feat_shape = None
    part_lengths = []

    for path in input_paths:
        length, shape = get_lmdb_info(path)
        part_lengths.append(length)
        total_samples += length
        if feat_shape is None:
            feat_shape = shape
        else:
            assert shape == feat_shape, f"Shape mismatch: {shape} vs {feat_shape}"
        print(f"  {os.path.basename(path)}: {length:,} samples")

    print(f"\nTotal samples to merge: {total_samples:,}")
    print(f"Feature shape: {feat_shape}")
    print(f"Output: {output_path}")

    # Create output directory
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Open output LMDB with fast write settings
    env = lmdb.open(
        output_path,
        map_size=map_size_gb * (1024 ** 3),
        writemap=True,
        map_async=True,
        sync=False,
        metasync=False
    )

    # Merge all parts with batched writes
    global_idx = 0
    batch_size = 100000
    pbar = tqdm(total=total_samples, desc="Merging", unit=" samples")

    for path, length in zip(input_paths, part_lengths):
        batch = []
        for value in iter_lmdb(path, length):
            batch.append((str(global_idx).encode('ascii'), value))
            global_idx += 1

            if len(batch) >= batch_size:
                with env.begin(write=True) as txn:
                    for key, val in batch:
                        txn.put(key, val)
                pbar.update(len(batch))
                batch = []

        # Write remaining batch
        if batch:
            with env.begin(write=True) as txn:
                for key, val in batch:
                    txn.put(key, val)
            pbar.update(len(batch))

    pbar.close()

    # Write metadata
    with env.begin(write=True) as txn:
        txn.put(b'__len__', str(global_idx).encode('ascii'))
        txn.put(b'__shape__', str(tuple(feat_shape)).encode('ascii'))

    # Final sync before close
    env.sync()
    env.close()

    print(f"\nMerged {global_idx:,} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge LMDB part files")
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing LMDB part files')
    parser.add_argument('--pattern', type=str, default="*_part*.lmdb",
                       help='Glob pattern for part files (default: *_part*.lmdb)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output filename (will be created in input_dir)')
    parser.add_argument('--map_size_gb', type=int, default=1500,
                       help='LMDB map size in GB (default: 1500)')

    args = parser.parse_args()

    # Find all part files
    pattern = os.path.join(args.input_dir, args.pattern)
    input_paths = sorted(glob.glob(pattern))

    if not input_paths:
        print(f"No files found matching: {pattern}")
        return

    output_path = os.path.join(args.input_dir, args.output)

    merge_lmdb_files(input_paths, output_path, args.map_size_gb)


if __name__ == '__main__':
    main()
