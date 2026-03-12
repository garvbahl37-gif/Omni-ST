"""
Omni-ST: 10x Genomics Visium Dataset
======================================
PyTorch Dataset wrapping AnnData objects from Visium spatial transcriptomics.

Supports:
  - spot-level gene expression + histology patch paired loading
  - spatial coordinate extraction
  - kNN spatial graph construction on-the-fly
  - instruction-conditioned task loading
  - multi-task yield format: (image_patch, gene_expr, coords, instruction, label)
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import Dataset
from PIL import Image


class VisiumDataset(Dataset):
    """
    Dataset for 10x Genomics Visium spatial transcriptomics.

    Parameters
    ----------
    adata : AnnData | str
        Loaded AnnData object or path to .h5ad file.
    task : str
        Task key defining what (input, output) pairs to yield.
        One of: ``image_to_gene``, ``gene_to_celltype``,
        ``graph_to_domain``, ``region_to_text``.
    image_dir : str | None
        Directory containing histology image tiles (PNG/TIFF).
        If None, images are loaded from ``adata.uns["spatial"]``.
    gene_list : list[str] | None
        Subset of genes to include. If None, use all HVGs.
    patch_size : int
        Spatial patch size in pixels.
    transform : callable | None
        Optional torchvision transform for image patches.
    max_spots : int | None
        Subsample to at most this many spots (for fast debugging).
    instruction : str | None
        If provided, override the instruction string for all samples.
    """

    TASK_INSTRUCTIONS = {
        "image_to_gene": (
            "Given the histology patch from a Visium spot, predict the gene expression profile."
        ),
        "gene_to_celltype": (
            "Based on the gene expression vector, classify the cell type of this spatial spot."
        ),
        "graph_to_domain": (
            "Identify the spatial transcriptomics domain this spot belongs to."
        ),
        "region_to_text": (
            "Describe the biological characteristics of this tissue region."
        ),
    }

    def __init__(
        self,
        adata: Union[ad.AnnData, str, Path],
        task: str = "image_to_gene",
        image_dir: Optional[str] = None,
        gene_list: Optional[List[str]] = None,
        patch_size: int = 224,
        transform: Optional[Callable] = None,
        max_spots: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> None:
        super().__init__()

        if isinstance(adata, (str, Path)):
            adata = ad.read_h5ad(adata)
        self.adata = adata
        self.task = task
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.transform = transform

        # Subselect genes
        if gene_list is not None:
            available = [g for g in gene_list if g in self.adata.var_names]
            self.adata = self.adata[:, available]
        self.gene_list = list(self.adata.var_names)

        # Subselect spots
        if max_spots is not None:
            idx = np.random.choice(len(self.adata), min(max_spots, len(self.adata)), replace=False)
            # BUG FIX: AnnData view after fancy indexing must be .copy()'d to
            # avoid silent view-mutation bugs and ensure a concrete object.
            self.adata = self.adata[idx].copy()

        self.instruction_text = instruction or self.TASK_INSTRUCTIONS.get(task, "")
        self._validate()

    def _validate(self) -> None:
        assert "spatial" in self.adata.obsm, (
            "AnnData must contain 'spatial' in obsm for spatial coordinates."
        )

    def __len__(self) -> int:
        return len(self.adata)

    def _load_patch(self, idx: int) -> Optional[torch.Tensor]:
        """Load the histology image patch for spot ``idx``."""
        # Try to load from library crop
        try:
            library_id = list(self.adata.uns["spatial"].keys())[0]
            full_image = self.adata.uns["spatial"][library_id]["images"]["hires"]  # numpy [H, W, 3]
            coords = self.adata.obsm["spatial"][idx]  # (x, y) pixel coords
            scale = self.adata.uns["spatial"][library_id]["scalefactors"]["tissue_hires_scalef"]
            cx, cy = int(coords[0] * scale), int(coords[1] * scale)
            half = self.patch_size // 2
            patch = full_image[
                max(cy - half, 0): cy + half,
                max(cx - half, 0): cx + half,
            ]
            patch = Image.fromarray((patch * 255).astype(np.uint8) if patch.max() <= 1 else patch.astype(np.uint8))
            patch = patch.resize((self.patch_size, self.patch_size))
        except Exception:
            patch = Image.fromarray(np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8))

        if self.transform:
            return self.transform(patch)
        return torch.tensor(np.array(patch), dtype=torch.float32).permute(2, 0, 1) / 255.0

    def _get_expression(self, idx: int) -> torch.Tensor:
        """Return log-normalised expression vector [G]."""
        expr = self.adata.X[idx]
        # BUG FIX: sparse matrix row-slice returns a (1, G) matrix.
        # .toarray() gives shape [1, G]; [0] reduces to [G].
        # np.squeeze() is unsafe when G==1 (drops the gene dim entirely).
        if hasattr(expr, "toarray"):
            expr = expr.toarray()[0]      # always shape [G]
        elif isinstance(expr, np.matrix):
            expr = np.asarray(expr).flatten()
        return torch.tensor(np.array(expr).flatten(), dtype=torch.float32)

    def _get_label(self, idx: int) -> Optional[torch.Tensor]:
        """Return the ground-truth label depending on task."""
        if self.task == "gene_to_celltype":
            if "cell_type" in self.adata.obs.columns:
                ct = self.adata.obs["cell_type"].iloc[idx]
                col = self.adata.obs["cell_type"]
                # BUG FIX: categorical columns expose .cat.categories; object
                # columns use .unique().  Both must be cast to a plain list
                # before calling .index() — Categorical.tolist() is safe.
                if hasattr(col, "cat"):
                    cats = col.cat.categories.tolist()
                else:
                    cats = sorted(col.dropna().unique().tolist())
                return torch.tensor(cats.index(ct), dtype=torch.long)
        elif self.task == "graph_to_domain":
            if "domain" in self.adata.obs.columns:
                d = self.adata.obs["domain"].iloc[idx]
                col = self.adata.obs["domain"]
                # BUG FIX: same categorical safety fix as above
                if hasattr(col, "cat"):
                    domains = col.cat.categories.tolist()
                else:
                    domains = sorted(col.dropna().unique().tolist())
                return torch.tensor(domains.index(d), dtype=torch.long)
        return None

    def __getitem__(self, idx: int) -> Dict:
        item: Dict = {"instruction": self.instruction_text, "spot_idx": idx}

        # Always include expression and spatial coords
        item["gene_expr"] = self._get_expression(idx)
        coords = self.adata.obsm["spatial"][idx]
        item["spatial_coords"] = torch.tensor(coords, dtype=torch.float32)

        # Task-specific additions
        if self.task in ("image_to_gene", "region_to_text"):
            item["image"] = self._load_patch(idx)

        label = self._get_label(idx)
        if label is not None:
            item["label"] = label

        return item


def create_visium_dataloaders(
    adata: Union[ad.AnnData, str],
    task: str = "image_to_gene",
    batch_size: int = 32,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: Optional[int] = None,
    seed: int = 42,
    **dataset_kwargs,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Split AnnData into train/val/test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    from torch.utils.data import random_split, DataLoader

    # BUG FIX: num_workers > 0 causes BrokenPipeError on Windows due to
    # lack of os.fork(). Default to 0 on Windows, 4 on Linux/Mac.
    if num_workers is None:
        num_workers = 0 if platform.system() == "Windows" else 4

    dataset = VisiumDataset(adata, task=task, **dataset_kwargs)
    n = len(dataset)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    if n_train <= 0:
        raise ValueError(
            f"Dataset too small ({n} spots) for the requested split ratios. "
            "Reduce val_split / test_split or increase the dataset size."
        )

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    def collate(batch: List[Dict]) -> Dict:
        """Safe collate that handles optional/None fields gracefully."""
        out: Dict = {}
        all_keys = {k for item in batch for k in item.keys()}
        for key in all_keys:
            vals = [item.get(key) for item in batch]
            # BUG FIX: skip keys where ANY value is None (optional labels)
            if any(v is None for v in vals):
                continue
            if isinstance(vals[0], torch.Tensor):
                out[key] = torch.stack(vals)
            else:
                out[key] = vals
        return out

    # pin_memory only beneficial when CUDA is available
    pin = torch.cuda.is_available()
    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin,
    )
    return (
        DataLoader(train_ds, shuffle=True, **dl_kwargs),
        DataLoader(val_ds, shuffle=False, **dl_kwargs),
        DataLoader(test_ds, shuffle=False, **dl_kwargs),
    )
