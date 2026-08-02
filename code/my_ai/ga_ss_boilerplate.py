"""GA + SS boilerplate — reusable interfaces extracted from ss_train.py.

Provides:
  - mutate_class_labels:  label mutation as pure data transform
  - bc_train:             unified BC interface (init_params + data → new_params)
  - setup_output_dir:     create output directories with logger
  - init_csv / write_csv_row:  CSV helpers
  - get_device:           return available torch device
  - setup_signal_handler: Ctrl+C handler that stops after current generation

Usage:
    from my_ai.ga_ss_boilerplate import (
        mutate_class_labels, bc_train,
        setup_output_dir, init_csv, write_csv_row,
        get_device, setup_signal_handler,
    )
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _CODE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import signal
import csv
from datetime import datetime

import numpy as np

from utils.logger import get_logger


# ═══════════════════════════════════════════════════════════════
# 1. Label mutation
# ═══════════════════════════════════════════════════════════════

def mutate_class_labels(
    cls_labels: torch.Tensor,
    head_logits: torch.Tensor,
    p_mutate: float = 0.1,
    temperature: float = 3.0,
    seed: int = 0,
    category_weights: list[float] | None = None,
) -> torch.Tensor:
    """Mutate class labels via temperature sampling from head logits.
    Pure data transform (no model forward/backward).

    Each label is independently mutated with probability **p_mutate**.
    Mutated labels are sampled from softmax(head_logits / temperature)
    with category rebalancing weights applied (rare classes get higher
    sampling probability).

    Preserved optimisations from ss_train bc_mutate_population:
        - logit clamping to [-50, 50] (prevents extreme logits from
          making temperature sampling effectively deterministic)
        - per-head independent mutation masks
        - category rebalancing weights (towers=1.0, rare=2.5)

    Args:
        cls_labels: (B, N_heads) original class labels.
        head_logits: (B, N_heads, 24) head logits for sampling.
        p_mutate: per-label mutation probability.
        temperature: sampling temperature (higher = more uniform).
        seed: random seed for reproducibility.
        category_weights: 24-element weight vector. If None, uses default
            weights (tower classes 0-15 → 1.0, rare classes 16-23 → 2.5).

    Returns:
        (B, N_heads) mutated class labels (same dtype/device as input).
    """
    import torch
    import torch.nn.functional as F

    if category_weights is None:
        w = torch.ones(24, dtype=head_logits.dtype, device=head_logits.device)
        w[16] = 2.5      # demolish (1 class)
        w[17:21] = 2.5   # super weapons (4 classes)
        w[21:23] = 2.5   # base upgrades (2 classes)
        w[23] = 2.5      # HOLD (1 class)
    else:
        w = torch.tensor(category_weights, dtype=head_logits.dtype,
                         device=head_logits.device)

    B, NH = cls_labels.shape
    mutated = cls_labels.clone()
    rng = torch.Generator(device=head_logits.device)
    rng.manual_seed(seed)

    for hi in range(NH):
        # Which samples to mutate (per-head independent mask)
        mask = torch.rand(B, generator=rng, device=head_logits.device) < p_mutate
        if not mask.any():
            continue

        logits_i = head_logits[mask, hi, :]          # (n_mutate, 24)
        if temperature <= 0:
            # Uniform random: sharp mutation, any action can replace any other
            sampled = torch.randint(0, 24, (logits_i.size(0),),
                                    generator=rng, device=head_logits.device)
        else:
            logits_i = torch.clamp(logits_i, -50.0, 50.0)
            logits_i = logits_i + torch.log(w) * temperature  # category rebalancing
            probs = F.softmax(logits_i / temperature, dim=-1)
            sampled = torch.multinomial(probs, 1, generator=rng).squeeze(-1)
        mutated[mask, hi] = sampled.to(mutated.dtype)

    return mutated


def mutate_class_labels_soft(
    cls_labels: torch.Tensor,
    head_logits: torch.Tensor,
    p_mutate: float = 0.1,
    temperature: float = 3.0,
    seed: int = 0,
    category_weights: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mutate both hard labels AND head_logits consistently.

    For mutated frames, the logit value of the old top class is swapped
    with the logit value of the newly sampled class, so the soft target
    distribution shifts toward the new action while preserving the
    relative structure of the original distribution.

    Returns:
        (mutated_labels, mutated_head_logits) — each (B, N_heads) or (B, N_heads, 24).
    """
    import torch
    import torch.nn.functional as F

    if category_weights is None:
        w = torch.ones(24, dtype=head_logits.dtype, device=head_logits.device)
        w[16] = 2.5
        w[17:21] = 2.5
        w[21:23] = 2.5
        w[23] = 2.5
    else:
        w = torch.tensor(category_weights, dtype=head_logits.dtype, device=head_logits.device)

    B, NH = cls_labels.shape
    mutated_labels = cls_labels.clone()
    mutated_logits = head_logits.clone()
    rng = torch.Generator(device=head_logits.device)
    rng.manual_seed(seed)

    for hi in range(NH):
        mask = torch.rand(B, generator=rng, device=head_logits.device) < p_mutate
        if not mask.any():
            continue

        logits_i = head_logits[mask, hi, :]
        if temperature <= 0:
            sampled = torch.randint(0, 24, (logits_i.size(0),),
                                    generator=rng, device=head_logits.device)
        else:
            logits_i = torch.clamp(logits_i, -50.0, 50.0)
            logits_i = logits_i + torch.log(w) * temperature
            probs = F.softmax(logits_i / temperature, dim=-1)
            sampled = torch.multinomial(probs, 1, generator=rng).squeeze(-1)

        # Swap logits between old top class and new class
        old_top = cls_labels[mask, hi]  # original top class per frame
        for row in range(sampled.size(0)):
            old_c, new_c = int(old_top[row]), int(sampled[row])
            if old_c != new_c:
                tmp = mutated_logits[mask, hi, :][row, old_c].clone()
                mutated_logits[mask, hi, :][row, old_c] = mutated_logits[mask, hi, :][row, new_c]
                mutated_logits[mask, hi, :][row, new_c] = tmp

        mutated_labels[mask, hi] = sampled.to(mutated_labels.dtype)

    return mutated_labels, mutated_logits


# ═══════════════════════════════════════════════════════════════
# 1.5 Action-map mutation
# ═══════════════════════════════════════════════════════════════

def mutate_action_map(
    action_map: torch.Tensor,
    p_mutate: float = 0.1,
    noise_std: float = 0.05,
    seed: int = 0,
) -> torch.Tensor:
    """Mutate action maps by adding spatial noise and re-normalizing.

    Each 19x19 spatial map per class is a probability distribution.
    Mutation adds Gaussian noise and re-normalizes to keep valid distributions.

    Args:
        action_map: (B, NUM_CLASSES, 19, 19) action probability maps.
        p_mutate: per-sample mutation probability.
        noise_std: std of added Gaussian noise.
        seed: random seed.

    Returns:
        (B, NUM_CLASSES, 19, 19) mutated action maps.
    """
    import torch

    B, C, H, W = action_map.shape
    mutated = action_map.clone()
    rng = torch.Generator(device=action_map.device)
    rng.manual_seed(seed)

    mask = torch.rand(B, generator=rng, device=action_map.device) < p_mutate
    if not mask.any():
        return mutated

    noise = torch.randn((mask.sum().item(), C, H, W),
                        generator=rng, device=action_map.device,
                        dtype=action_map.dtype) * noise_std
    mutated[mask] = mutated[mask] + noise
    mutated[mask] = torch.clamp(mutated[mask], min=0.0)

    # Re-normalize each class's spatial map to sum to 1
    for b in range(mask.sum().item()):
        for c in range(C):
            total = mutated[mask][b, c].sum()
            if total > 0:
                mutated[mask][b, c] /= total

    return mutated


# ═══════════════════════════════════════════════════════════════
# 2. BC unified interface
# ═══════════════════════════════════════════════════════════════

def bc_train(
    init_params: np.ndarray,
    model_template: AntWarNetwork,
    dataset,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-3,
    batch_size: int = 64,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    lambda_class: float = 1.0,
    lambda_map: float = 1.0,
    lambda_div: float = 0.0,
    lambda_soft: float = 0.0,
    bias_decay: float = 0.0,
    log=None,
) -> np.ndarray:
    """Train a model via BC from **init_params**, return new parameter vector.

    Creates a fresh model clone (same architecture as **model_template**),
    loads **init_params**, runs ``ss_supervised_update``, and returns the
    trained parameter vector.

    Does **not** modify ``model_template`` or ``init_params``.

    Args:
        init_params: starting parameter vector (flat np.ndarray).
        model_template: model instance whose architecture to clone.
        dataset: SSDataset or compatible Dataset.
        device: torch device for training.
        epochs: BC training epochs.
        lr: learning rate.
        batch_size: DataLoader batch size.
        weight_decay: AdamW weight decay (0 = Adam, >0 = AdamW).
        label_smoothing: label smoothing factor for CE loss.
        lambda_class: class CE loss weight.
        lambda_map: action-map KL loss weight.
        lambda_div: head diversity loss weight (0 = disabled).
        lambda_soft: soft-target KL loss weight (0 = disabled).
        bias_decay: extra L2 penalty on policy head biases (0 = disabled).
        log: optional logger.

    Returns:
        Trained parameter vector (flat np.ndarray, same shape as init_params).
    """
    from my_ai.network import AntWarNetwork
    from my_ai.ss_train import ss_supervised_update

    # Clone architecture from template (handles small/large/custom)
    no_bn = getattr(model_template, "no_bn", False)
    model = AntWarNetwork(
        num_resblocks=model_template.num_resblocks,
        num_heads=model_template.num_heads,
        latent_dim=model_template.LATENT_DIM,
        no_bn=no_bn,
    )
    model.set_parameters_from_vector(init_params)

    # Copy BN running stats from template (freeze pretrained stats)
    tmpl_sd = model_template.state_dict()
    for name, buf in model.state_dict().items():
        if "running_mean" in name or "running_var" in name:
            buf.copy_(tmpl_sd[name])

    ss_supervised_update(
        model, dataset, device=device,
        epochs=epochs, lr=lr, batch_size=batch_size,
        weight_decay=weight_decay, label_smoothing=label_smoothing,
        lambda_class=lambda_class, lambda_map=lambda_map,
        lambda_div=lambda_div, lambda_soft=lambda_soft,
        bias_decay=bias_decay, log=log,
    )

    # Restore BN stats (prevent BC from corrupting them)
    for name, buf in model.state_dict().items():
        if "running_mean" in name or "running_var" in name:
            buf.copy_(tmpl_sd[name])

    return model.get_parameters_as_vector()


# ═══════════════════════════════════════════════════════════════
# 3. Boilerplate utilities
# ═══════════════════════════════════════════════════════════════

def setup_output_dir(args) -> tuple[Path, Path]:
    """Create output directories and logger.

    Args:
        args: parsed arguments (must have ``out_dir`` attribute).

    Returns:
        (out_dir, bc_dir, log)
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("training_history") / f"ga_ss_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    bc_dir = out_dir / "bc_data"
    bc_dir.mkdir(parents=True, exist_ok=True)

    log = get_logger(out_dir / "train.log")
    from utils.logger import redirect_stderr_to_log
    redirect_stderr_to_log(log)

    # Save config
    with open(out_dir / "config.txt", "w") as f:
        for key, val in sorted(vars(args).items()):
            f.write(f"{key}={val}\n")
        f.write(f"timestamp={ts}\n")

    return out_dir, bc_dir, log


def init_csv(csv_path: Path, header: list[str], first_row: list | None = None):
    """Create a CSV file with header and optional first data row."""
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        if first_row is not None:
            w.writerow(first_row)


def write_csv_row(csv_path: Path, row: list):
    """Append a row to an existing CSV file."""
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)


def get_device():
    """Return cuda if available, else cpu."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_signal_handler(interrupted_ref: list, log=None):
    """Set up Ctrl+C handler that stops after current generation.

    Args:
        interrupted_ref: a mutable single-element list (e.g. ``[False]``)
            that will be set to ``[True]`` on first Ctrl+C.
            Second Ctrl+C forces immediate exit.
        log: optional logger.

    Example:
        >>> interrupted = [False]
        >>> setup_signal_handler(interrupted, log)
        >>> for gen in range(generations):
        ...     if interrupted[0]:
        ...         break
    """
    def _handler(signum, frame):
        if interrupted_ref[0]:
            if log:
                log.print("Forced exit.")
            sys.exit(1)
        interrupted_ref[0] = True
        if log:
            log.print("")
            log.print(key="interrupt",
                      value="Ctrl+C received, stopping after current generation...")

    signal.signal(signal.SIGINT, _handler)
    try:
        signal.signal(signal.SIGTERM, _handler)
    except AttributeError:
        pass


# ═══════════════════════════════════════════════════════════════
# 6. Merge two SSDataset instances
# ═══════════════════════════════════════════════════════════════

def merge_datasets(ds1: SSDataset, ds2: SSDataset) -> SSDataset:
    """Merge two SSDataset instances by concatenating their data arrays.

    Both datasets should already have HOLD filtering and oversampling applied.
    The merged dataset skips re-applying those transforms.

    Args:
        ds1: first SSDataset
        ds2: second SSDataset

    Returns:
        New SSDataset with data from both inputs concatenated.
    """
    from my_ai.ss_train import SSDataset

    merged = SSDataset.__new__(SSDataset)
    merged.board = np.concatenate([ds1.board, ds2.board], axis=0)
    merged.stats = np.concatenate([ds1.stats, ds2.stats], axis=0)
    merged.class_label = np.concatenate([ds1.class_label, ds2.class_label], axis=0)
    merged.action_map = np.concatenate([ds1.action_map, ds2.action_map], axis=0)
    merged.head_logits = np.concatenate([ds1.head_logits, ds2.head_logits], axis=0)

    # Optional fields: only keep if both have them
    if ds1.class_scores is not None and ds2.class_scores is not None:
        merged.class_scores = np.concatenate([ds1.class_scores, ds2.class_scores], axis=0)
    else:
        merged.class_scores = None

    if ds1.value is not None and ds2.value is not None:
        merged.value = np.concatenate([ds1.value, ds2.value], axis=0)
    else:
        merged.value = None

    return merged


# ═══════════════════════════════════════════════════════════════
# 7. Sub-sample an SSDataset
# ═══════════════════════════════════════════════════════════════

def subsample_dataset(ds: SSDataset, n: int, rng: np.random.Generator) -> SSDataset:
    """Randomly sample n frames from an SSDataset (without replacement).

    Useful for mutation: pick a small random subset of frames to mutate,
    instead of mutating the entire dataset.

    Args:
        ds: source SSDataset
        n: number of frames to sample
        rng: random number generator

    Returns:
        New SSDataset with n frames (or fewer if ds has fewer than n frames).
    """
    from my_ai.ss_train import SSDataset

    n = min(n, len(ds))
    idx = rng.choice(len(ds), size=n, replace=False)
    idx.sort()

    sub = SSDataset.__new__(SSDataset)
    sub.board = ds.board[idx]
    sub.stats = ds.stats[idx]
    sub.class_label = ds.class_label[idx]
    sub.action_map = ds.action_map[idx]
    sub.head_logits = ds.head_logits[idx]

    if ds.class_scores is not None:
        sub.class_scores = ds.class_scores[idx]
    else:
        sub.class_scores = None

    if ds.value is not None:
        sub.value = ds.value[idx]
    else:
        sub.value = None

    return sub
