#!/usr/bin/env python3
"""
Extract text features from OpenWebText2 and save to LMDB format.

Supports two modes:
1. Text embedding models (GTE, E5, etc.) - sentence-level embeddings
2. LLM residual streams (Pythia) - activations at specific layers

Usage examples:
    # GTE embeddings
    python extract_features_text.py \
        --input_dir /path/to/openwebtext2 \
        --output_dir ./features/owt2/gte \
        --model_type embedding \
        --model_name Alibaba-NLP/gte-large-en-v1.5 \
        --batch_size 64 \
        --max_samples 1000000

    # Pythia residual stream (layer 11)
    python extract_features_text.py \
        --input_dir /path/to/openwebtext2 \
        --output_dir ./features/owt2/pythia410m_L11 \
        --model_type pythia \
        --model_name EleutherAI/pythia-410m-deduped \
        --layers 11 \
        --pool mean \
        --batch_size 16 \
        --max_samples 1000000
"""

import os
import gc
import json
import argparse
import glob
from pathlib import Path
from typing import Iterator, Optional

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
import zstandard as zstd
from tqdm.auto import tqdm


def iter_jsonl_zst(file_path: str) -> Iterator[dict]:
    """Iterate over a zstd-compressed JSONL file."""
    dctx = zstd.ZstdDecompressor()
    with open(file_path, 'rb') as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = reader.read().decode('utf-8')
            for line in text_stream.strip().split('\n'):
                if line:
                    yield json.loads(line)


def count_openwebtext_samples(input_dir: str) -> int:
    """Count total samples in OpenWebText directory."""
    files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl.zst")))
    if not files:
        raise ValueError(f"No .jsonl.zst files found in {input_dir}")

    total = 0
    pbar = tqdm(files, desc="Counting samples", unit=" files")
    for fpath in pbar:
        for record in iter_jsonl_zst(fpath):
            text = record.get('text', '')
            if isinstance(text, str) and len(text.strip()) > 0:
                total += 1
        pbar.set_postfix({"total": f"{total:,}"})
    return total


def iter_openwebtext_dir(input_dir: str, max_samples: Optional[int] = None) -> Iterator[str]:
    """Iterate over all jsonl.zst files in the OpenWebText directory."""
    files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl.zst")))
    if not files:
        raise ValueError(f"No .jsonl.zst files found in {input_dir}")

    count = 0
    for fpath in files:
        for record in iter_jsonl_zst(fpath):
            text = record.get('text', '')
            if isinstance(text, str) and len(text.strip()) > 0:
                yield text.strip()
                count += 1
                if max_samples is not None and count >= max_samples:
                    return


def load_hf_dataset(dataset_path: str, split: str = 'train', cache_dir: str = None):
    """Load HuggingFace dataset."""
    import sys
    # Remove current dir from path to avoid local datasets module
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path = [p for p in sys.path if p != script_dir and p != '']
    from datasets import load_dataset

    print(f"Loading HuggingFace dataset from {dataset_path}...")
    ds = load_dataset(
        dataset_path,
        split=split,
        cache_dir=cache_dir,
        download_mode="reuse_cache_if_exists",
    )
    print(f"Dataset loaded: {len(ds):,} samples")
    return ds


def batch_iterator(iterator: Iterator, batch_size: int):
    """Yield batches from an iterator."""
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class EmbeddingExtractor:
    """Extract embeddings using sentence-transformer style models (GTE, E5, etc.)."""

    def __init__(self, model_name: str, device: str = 'cuda', max_length: int = 512):
        from transformers import AutoTokenizer, AutoModel

        self.device = torch.device(device)
        self.max_length = max_length
        self.model_name = model_name

        print(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            trust_remote_code=True
        ).to(self.device).eval()

        # Get embedding dimension
        with torch.no_grad():
            dummy = self.tokenizer("test", return_tensors="pt", padding=True, truncation=True)
            dummy = {k: v.to(self.device) for k, v in dummy.items()}
            out = self.model(**dummy)
            if hasattr(out, 'last_hidden_state'):
                self.dim = out.last_hidden_state.shape[-1]
            elif hasattr(out, 'pooler_output'):
                self.dim = out.pooler_output.shape[-1]
            else:
                self.dim = out[0].shape[-1]
        print(f"Embedding dimension: {self.dim}")

    @torch.no_grad()
    def extract(self, texts: list[str]) -> np.ndarray:
        """Extract embeddings for a batch of texts."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
            outputs = self.model(**inputs)

        # Mean pooling over tokens (masked)
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs[0]

        mask = inputs['attention_mask'].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Normalize to unit norm
        pooled = F.normalize(pooled, p=2, dim=-1)

        return pooled.float().cpu().numpy()


class PythiaExtractor:
    """Extract residual stream activations from Pythia models."""

    def __init__(
        self,
        model_name: str,
        layers: list[int],
        site: str = 'post',  # 'pre' or 'post'
        pool: str = 'mean',  # 'mean' or 'last'
        device: str = 'cuda',
        max_length: int = 256
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.device = torch.device(device)
        self.layers = layers
        self.site = site
        self.pool = pool
        self.max_length = max_length
        self.model_name = model_name

        print(f"Loading Pythia model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if device == 'cuda' else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(self.device).eval()

        self.dim = self.model.config.hidden_size
        print(f"Hidden dimension: {self.dim}, extracting layers: {layers}")

        # Storage for captured activations
        self.captured = {L: None for L in layers}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        def make_hook(layer_id: int):
            def hook_fn(module, inp, out):
                if self.site == 'pre':
                    self.captured[layer_id] = inp[0].detach()
                else:
                    self.captured[layer_id] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
            return hook_fn

        for L in self.layers:
            block = self.model.gpt_neox.layers[L]
            self.hooks.append(block.register_forward_hook(make_hook(L)))

    def cleanup(self):
        """Remove hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    @torch.no_grad()
    def extract(self, texts: list[str]) -> np.ndarray:
        """Extract residual stream activations for a batch of texts.

        Returns:
            If single layer: (batch, dim)
            If multiple layers: (batch, n_layers, dim)
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Reset captured
        for L in self.layers:
            self.captured[L] = None

        with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
            _ = self.model(**inputs, use_cache=False)

        attn_mask = inputs['attention_mask']

        features_per_layer = []
        for L in self.layers:
            hs = self.captured[L]  # (B, T, dim)
            if hs is None:
                raise RuntimeError(f"Hook failed for layer {L}")

            if self.pool == 'mean':
                mask = attn_mask.unsqueeze(-1).float()
                pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:  # last
                last_idx = attn_mask.sum(dim=1) - 1
                pooled = hs[torch.arange(hs.size(0), device=hs.device), last_idx, :]

            # Normalize to unit norm
            pooled = F.normalize(pooled.float(), p=2, dim=-1)
            features_per_layer.append(pooled.cpu().numpy())

        if len(self.layers) == 1:
            return features_per_layer[0]  # (B, dim)
        else:
            return np.stack(features_per_layer, axis=1)  # (B, n_layers, dim)


def write_to_lmdb(
    output_path: str,
    iterator: Iterator[str],
    extractor,
    batch_size: int,
    total_samples: int,
    max_samples: Optional[int] = None,
    map_size_gb: int = 100
):
    """Extract features and write to LMDB."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    env = lmdb.open(
        output_path,
        map_size=map_size_gb * (1024 ** 3),
        writemap=True,
        map_async=True
    )

    total_written = 0
    feat_shape = None

    # Use max_samples if set, otherwise use total_samples from counting
    pbar_total = max_samples if max_samples else total_samples
    pbar = tqdm(desc="Extracting", total=pbar_total, unit=" samples",
                dynamic_ncols=True, smoothing=0.1)

    for batch_texts in batch_iterator(iterator, batch_size):
        features = extractor.extract(batch_texts)  # (B, dim) or (B, n_layers, dim)

        if feat_shape is None:
            feat_shape = features.shape[1:]  # exclude batch dim
            tqdm.write(f"Feature shape per sample: {feat_shape}")

        with env.begin(write=True) as txn:
            for i, feat in enumerate(features):
                key = str(total_written + i).encode('ascii')
                txn.put(key, feat.astype(np.float32).tobytes())

        total_written += len(features)
        pbar.update(len(features))

        if max_samples is not None and total_written >= max_samples:
            break

    pbar.close()

    # Write metadata
    with env.begin(write=True) as txn:
        txn.put(b'__len__', str(total_written).encode('ascii'))
        txn.put(b'__shape__', str(tuple(feat_shape)).encode('ascii'))

    env.close()
    print(f"Wrote {total_written} samples to {output_path}")
    print(f"Feature shape: {feat_shape}")


def main():
    parser = argparse.ArgumentParser(description="Extract text features to LMDB")

    # Input/output
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Directory containing .jsonl.zst files (for OpenWebText2)')
    parser.add_argument('--hf_dataset', type=str, default=None,
                       help='Path to HuggingFace dataset (local or hub)')
    parser.add_argument('--text_field', type=str, default='text',
                       help='Field name containing text in HF dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for LMDB database')
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Dataset name (used in output path). If not provided, inferred from input')

    # Model selection
    parser.add_argument('--model_type', type=str, choices=['embedding', 'pythia'], default='embedding',
                       help='Model type: embedding (GTE/E5) or pythia (residual stream)')
    parser.add_argument('--model_name', type=str, default='Alibaba-NLP/gte-large-en-v1.5',
                       help='HuggingFace model name')

    # Pythia-specific
    parser.add_argument('--layers', type=str, default='11',
                       help='Comma-separated layer indices for Pythia (e.g., "1,11" or "3,7,11,15,19,23")')
    parser.add_argument('--site', type=str, choices=['pre', 'post'], default='post',
                       help='Residual stream position: pre (entering block) or post (leaving block)')
    parser.add_argument('--pool', type=str, choices=['mean', 'last'], default='mean',
                       help='Pooling strategy over tokens')

    # Processing
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=256,
                       help='Max sequence length for tokenization')
    parser.add_argument('--start_index', type=int, default=0,
                       help='Start index for dataset slicing (for splitting into parts)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process')
    parser.add_argument('--part_suffix', type=str, default=None,
                       help='Suffix to add to output filename (e.g., "_part1")')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--map_size_gb', type=int, default=100,
                       help='LMDB map size in GB')

    args = parser.parse_args()

    # Validate input
    if args.input_dir is None and args.hf_dataset is None:
        parser.error("Must specify either --input_dir or --hf_dataset")
    if args.input_dir is not None and args.hf_dataset is not None:
        parser.error("Specify only one of --input_dir or --hf_dataset")

    # Infer dataset name
    if args.dataset_name:
        dataset_name = args.dataset_name
    elif args.input_dir:
        dataset_name = os.path.basename(args.input_dir.rstrip('/'))
    else:
        dataset_name = os.path.basename(args.hf_dataset.rstrip('/'))

    # Build output path
    model_tag = args.model_name.split('/')[-1].replace('-', '_')
    if args.model_type == 'pythia':
        layers_str = args.layers.replace(',', '_')
        output_name = f"{model_tag}_L{layers_str}_{args.site}_{args.pool}"
    else:
        output_name = model_tag

    # Add part suffix if specified
    if args.part_suffix:
        output_name = f"{output_name}{args.part_suffix}"
    output_path = os.path.join(args.output_dir, dataset_name, f"{output_name}.lmdb")
    print(f"Output: {output_path}")

    # Create extractor
    if args.model_type == 'embedding':
        extractor = EmbeddingExtractor(
            model_name=args.model_name,
            device=args.device,
            max_length=args.max_length
        )
    else:
        layers = [int(x) for x in args.layers.split(',') if x.strip()]
        extractor = PythiaExtractor(
            model_name=args.model_name,
            layers=layers,
            site=args.site,
            pool=args.pool,
            device=args.device,
            max_length=args.max_length
        )

    # Load data based on input type
    if args.input_dir:
        # OpenWebText2 style: directory of .jsonl.zst files
        print("Counting total samples...")
        total_samples = count_openwebtext_samples(args.input_dir)
        print(f"Total samples: {total_samples:,}")
        text_iter = iter_openwebtext_dir(args.input_dir, max_samples=args.max_samples)
    else:
        # HuggingFace dataset
        ds = load_hf_dataset(args.hf_dataset, cache_dir=os.environ.get('HF_DATASETS_CACHE'))
        total_samples = len(ds)

        # Apply slicing if start_index or max_samples specified
        end_index = total_samples
        if args.start_index > 0 or args.max_samples is not None:
            start = args.start_index
            if args.max_samples is not None:
                end_index = min(start + args.max_samples, total_samples)
            ds = ds.select(range(start, end_index))
            total_samples = len(ds)
            print(f"Processing slice [{start}:{end_index}] = {total_samples:,} samples")

        text_iter = (ex[args.text_field] for ex in ds)

    write_to_lmdb(
        output_path=output_path,
        iterator=text_iter,
        extractor=extractor,
        batch_size=args.batch_size,
        total_samples=total_samples,
        max_samples=args.max_samples,
        map_size_gb=args.map_size_gb
    )

    # Cleanup
    if hasattr(extractor, 'cleanup'):
        extractor.cleanup()

    torch.cuda.empty_cache()
    gc.collect()

    print("Done!")


if __name__ == '__main__':
    main()
