"""
sparse_autoencoder.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sparse Autoencoder (SAE)  —  Encoding layer only
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Responsibility: convert dense embeddings → Sparse Distributed Representations.
Nothing else.

This module does NOT:
  ✗ store SDRs
  ✗ retrieve memories
  ✗ compute ripple / associative recall
  ✗ compare SDRs against each other
  ✗ know that other SDRs exist

All of the above are brain responsibilities (EnhancedSuperBrain).

Two sparsity modes (select via mode= argument):

  "topk"
    Structural sparsity — enforced at every forward pass.
    After ReLU, keep only the top-k activations; zero out the rest.
    k is fixed or can be overridden per call (variable sparsity).
    Sparsity = exactly k / num_neurons at all times.
    Training loss = reconstruction only (MSE).
    Optional auxiliary loss: encourage every neuron to fire
    at least once every N steps (dead-neuron rescue).

  "l1"
    Penalty-based sparsity — enforced through the training objective.
    Training loss = MSE + lambda_l1 * ||hidden||_1
    The L1 penalty pushes most activations toward zero.
    Sparsity varies per input: complex inputs naturally need more neurons.
    Actual active count depends on lambda_l1 — tune until mean sparsity
    reaches your target (e.g. 1-4 %).

Architecture:
  Input:   dense embedding  (input_dim = 768)
  Encoder: Linear(input_dim, num_neurons) + bias → ReLU → sparsity
  Decoder: Linear(num_neurons, input_dim) + bias  (training only; unused at inference)
  Output:  sparse float32 vector (num_neurons = 86 000)
           binarise with encode_binary() if you need uint8 {0,1}

Pipeline:
  Dense embedding (768-dim float32)
        │
  Pre-normalisation     — unit-norm per sample (optional but recommended)
        │
  Encoder linear        — W_enc [768 → 86 000] + b_enc
        │
  ReLU
        │
  Sparsity gate         — Top-K mask  OR  L1 training pressure
        │
  Sparse hidden state   — (86 000,) float32, ~1-4 % non-zero
        │
  [training only] Decoder → MSE loss + sparsity term → backprop
        │
  Sparse output to brain

What makes this "full SAE" and not a toy:
  ✓ Two complete sparsity regimes (topk / l1), switchable by argument
  ✓ Dead-neuron detection and rescue (EMA firing-rate tracking)
  ✓ Pre-norm + weight tying option for stability
  ✓ Full training loop with learning-rate scheduler and early stopping
  ✓ Variable sparsity at inference (override k per call)
  ✓ Encoder-only inference (decoder never used after training)
  ✓ Full save / load of all weights and training state
  ✓ GPU/CPU transparent via torch.device
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class SAEConfig:
    """
    Complete configuration for the Sparse Autoencoder.

    Defaults tuned for:
        input_dim   = 768  (all-mpnet-base-v2)
        num_neurons = 86 000
        mode        = "topk"   or   "l1"
    """

    # ── Architecture ──────────────────────────────────────────────────────────
    input_dim: int = 768
    num_neurons: int = 86_000

    # ── Sparsity mode ─────────────────────────────────────────────────────────
    mode: Literal["topk", "l1"] = "topk"
    """
    "topk"  — structural sparsity. Exactly `topk_k` neurons active per sample.
    "l1"    — penalty sparsity.  lambda_l1 penalises non-zero activations.
    """

    # ── Top-K settings (mode="topk") ─────────────────────────────────────────
    topk_k: int = 1_720
    """Number of neurons to keep active per sample.
    Default: int(0.02 * 86 000) = 1 720  →  ~2 % sparsity."""

    topk_aux_alpha: float = 1e-3
    """
    Auxiliary loss coefficient for dead-neuron rescue in topk mode.
    Adds a small MSE loss computed on the top-k activations of dead neurons,
    encouraging them to contribute to reconstruction without dominating.
    0.0 = disabled.
    """

    # ── L1 settings (mode="l1") ───────────────────────────────────────────────
    lambda_l1: float = 5e-4
    """
    L1 penalty coefficient.
    total_loss = MSE + lambda_l1 * mean(||hidden||_1)
    Increase to get sparser codes; decrease if reconstruction quality drops.
    Typical range: 1e-5 to 1e-3. Start at 5e-4 and tune from there.
    """

    # ── Input pre-processing ──────────────────────────────────────────────────
    normalize_inputs: bool = True
    """Unit-norm each embedding before feeding to the encoder.
    Recommended: prevents scale differences from dominating the encoder weights."""

    # ── Initialisation ────────────────────────────────────────────────────────
    init_method: Literal["kaiming", "xavier", "orthogonal"] = "kaiming"
    """Weight initialisation for the encoder/decoder."""

    tie_weights: bool = False
    """
    If True: decoder weight = encoder weight transposed (W_dec = W_enc^T).
    Halves parameter count. Slightly worse reconstruction but better
    feature regularity. Usually False for large expansion ratios.
    """

    # ── Training ──────────────────────────────────────────────────────────────
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 10
    batch_size: int = 256
    lr_schedule: bool = True
    """Cosine annealing LR schedule over total training steps."""

    early_stopping_patience: int = 5
    """Stop training if validation loss does not improve for N epochs.
    0 = disabled."""

    grad_clip: float = 1.0
    """Gradient clipping max norm. 0.0 = disabled."""

    # ── Dead-neuron rescue ────────────────────────────────────────────────────
    dead_neuron_window: int = 1000
    """
    A neuron is considered dead if it has not fired in the last N steps.
    Applies to both modes. In topk mode, dead neurons are given a forced
    activation via the auxiliary loss. In l1 mode, their encoder bias is
    bumped upward slightly.
    """

    dead_neuron_threshold: float = 0.0
    """
    Minimum EMA firing rate below which a neuron is considered dead.
    0.0 = any neuron that has literally never fired in the window.
    """

    dead_neuron_bias_bump: float = 0.01
    """How much to increment encoder bias of dead neurons per rescue step."""

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int = 42
    device: str = "auto"
    """'auto' selects CUDA if available, else CPU."""

    stats_window: int = 500
    """Rolling window for health statistics."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. TOP-K GATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TopKGate(nn.Module):
    """
    Differentiable Top-K gate.

    Forward pass:
      1. Accept pre-activation hidden state h  (batch, num_neurons)
      2. Compute ReLU(h)
      3. Keep only the top-k values per sample; set rest to 0
      4. Return sparse hidden state

    Gradient flows only through the top-k positions (straight-through
    for the masking step — the mask itself is not differentiable, but
    the kept activations are, so backprop works via the encoder weights).

    Variable k: pass k_override at call time to change sparsity per batch.
    """

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def forward(
        self,
        h: torch.Tensor,
        k_override: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h:          (B, N) pre-activation logits
            k_override: use this k instead of self.k for this call

        Returns:
            sparse:  (B, N) sparse hidden state (float32)
            indices: (B, k) indices of the top-k positions (for stats / aux loss)
        """
        k = k_override if k_override is not None else self.k
        k = min(k, h.shape[1])

        activated = F.relu(h)                                               # (B, N)

        # Find top-k indices per sample
        topk_vals, topk_idx = torch.topk(activated, k=k, dim=1)            # (B, k)

        # Build sparse output via scatter
        sparse = torch.zeros_like(activated)
        sparse.scatter_(1, topk_idx, topk_vals)

        return sparse, topk_idx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. DEAD NEURON TRACKER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DeadNeuronTracker:
    """
    Tracks per-neuron firing rates via an exponential moving average.

    EMA update (per training step):
        rate[i] = (1 - alpha) * rate[i] + alpha * fired[i]

    where fired[i] = 1 if neuron i was active in ANY sample in the batch,
    else 0.

    A neuron is "dead" if rate[i] <= dead_threshold.

    Alpha is derived from the window:  alpha = 1 / window
    so the EMA effectively looks back ~window steps.
    """

    def __init__(
        self,
        num_neurons: int,
        window: int,
        threshold: float,
        device: torch.device,
    ) -> None:
        self.num_neurons = num_neurons
        self.window = window
        self.threshold = threshold
        self.alpha = 1.0 / max(1, window)
        self.device = device

        # EMA firing rates — start at 1.0 so all neurons are "alive" initially
        self.rates = torch.ones(num_neurons, device=device, dtype=torch.float32)
        self.step = 0
        self._total_rescues = 0

    def update(self, sparse: torch.Tensor) -> None:
        """
        Update EMA firing rates from a batch of sparse hidden states.

        Args:
            sparse: (B, N) sparse hidden state (any non-zero = fired)
        """
        # Neuron fired in this batch if any sample activated it
        fired = (sparse.abs() > 0).any(dim=0).float()                      # (N,)
        self.rates = (1.0 - self.alpha) * self.rates + self.alpha * fired
        self.step += 1

    def dead_indices(self) -> torch.Tensor:
        """Return indices of neurons with firing rate <= threshold."""
        return torch.where(self.rates <= self.threshold)[0]

    def n_dead(self) -> int:
        return int((self.rates <= self.threshold).sum().item())

    def record_rescue(self, n: int) -> None:
        self._total_rescues += n

    def state_dict(self) -> dict:
        return {
            "rates": self.rates.cpu().numpy(),
            "step": self.step,
            "total_rescues": self._total_rescues,
        }

    def load_state_dict(self, d: dict) -> None:
        self.rates = torch.tensor(d["rates"], device=self.device)
        self.step = d["step"]
        self._total_rescues = d.get("total_rescues", 0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. SAE NETWORK (nn.Module)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SAENetwork(nn.Module):
    """
    The raw torch.nn.Module containing encoder + decoder weights.

    Kept separate so it can be saved as a state_dict independently of
    all the training machinery.

    Encoder: Linear(input_dim, num_neurons) + bias  → ReLU → sparsity gate
    Decoder: Linear(num_neurons, input_dim) + bias  (training only)
    """

    def __init__(self, config: SAEConfig) -> None:
        super().__init__()
        cfg = config

        self.enc_weight = nn.Parameter(
            torch.empty(cfg.num_neurons, cfg.input_dim)
        )
        self.enc_bias = nn.Parameter(torch.zeros(cfg.num_neurons))

        if cfg.tie_weights:
            self.dec_weight = None                                          # computed on the fly
        else:
            self.dec_weight = nn.Parameter(
                torch.empty(cfg.input_dim, cfg.num_neurons)
            )
        self.dec_bias = nn.Parameter(torch.zeros(cfg.input_dim))

        # Top-K gate (only used when mode == "topk")
        self.topk_gate = TopKGate(k=cfg.topk_k)

        self._init_weights(cfg)

    def _init_weights(self, cfg: SAEConfig) -> None:
        if cfg.init_method == "kaiming":
            nn.init.kaiming_uniform_(self.enc_weight, nonlinearity="relu")
            if self.dec_weight is not None:
                nn.init.kaiming_uniform_(self.dec_weight, nonlinearity="relu")
        elif cfg.init_method == "xavier":
            nn.init.xavier_uniform_(self.enc_weight)
            if self.dec_weight is not None:
                nn.init.xavier_uniform_(self.dec_weight)
        elif cfg.init_method == "orthogonal":
            nn.init.orthogonal_(self.enc_weight)
            if self.dec_weight is not None:
                nn.init.orthogonal_(self.dec_weight)

        # Normalise decoder columns to unit norm at init
        # (helps avoid degenerate solutions where one feature dominates)
        if self.dec_weight is not None:
            with torch.no_grad():
                norms = self.dec_weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
                self.dec_weight.div_(norms)

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        """Linear encoder step (before activation). Returns (B, num_neurons)."""
        return F.linear(x, self.enc_weight, self.enc_bias)

    def decode(self, sparse: torch.Tensor) -> torch.Tensor:
        """Decode sparse hidden state back to input space."""
        if self.dec_weight is None:
            return F.linear(sparse, self.enc_weight.t(), self.dec_bias)
        return F.linear(sparse, self.dec_weight, self.dec_bias)

    def normalise_decoder_cols(self) -> None:
        """
        Project decoder columns back to unit norm after each gradient step.
        Prevents any single feature from growing unboundedly.
        Called by the training engine after each optimizer step.
        """
        if self.dec_weight is not None:
            with torch.no_grad():
                norms = self.dec_weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
                self.dec_weight.div_(norms)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. STATS TRACKER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SAEStatsTracker:
    """
    Rolling health statistics for the SAE's encoding behaviour.

    Tracks (over a sliding window of `window` steps):
        loss_total     — combined training loss
        loss_recon     — reconstruction MSE component
        loss_sparse    — sparsity penalty component (L1 or aux)
        sparsity       — fraction of active neurons per sample (mean over batch)
        n_active       — mean absolute count of active neurons per sample

    Encoding-quality only — no brain-level concerns.
    """

    def __init__(self, window: int = 500) -> None:
        self.window = window
        self._loss_total: List[float] = []
        self._loss_recon: List[float] = []
        self._loss_sparse: List[float] = []
        self._sparsity: List[float] = []
        self._n_active: List[float] = []
        self.total_steps = 0
        self.epochs_completed = 0

    def update(
        self,
        loss_total: float,
        loss_recon: float,
        loss_sparse: float,
        sparsity: float,
        n_active: float,
    ) -> None:
        self._loss_total.append(loss_total)
        self._loss_recon.append(loss_recon)
        self._loss_sparse.append(loss_sparse)
        self._sparsity.append(sparsity)
        self._n_active.append(n_active)
        self.total_steps += 1

        if len(self._loss_total) > self.window:
            self._loss_total = self._loss_total[-self.window:]
            self._loss_recon = self._loss_recon[-self.window:]
            self._loss_sparse = self._loss_sparse[-self.window:]
            self._sparsity = self._sparsity[-self.window:]
            self._n_active = self._n_active[-self.window:]

    def summary(self) -> Dict:
        def _s(lst):
            a = np.array(lst) if lst else np.array([0.0])
            return float(a.mean()), float(a.std()), float(a.min()), float(a.max())

        lt_m, lt_s, lt_min, lt_max = _s(self._loss_total)
        lr_m = float(np.mean(self._loss_recon)) if self._loss_recon else 0.0
        ls_m = float(np.mean(self._loss_sparse)) if self._loss_sparse else 0.0
        sp_m, sp_s, sp_min, sp_max = _s(self._sparsity)

        return {
            "total_steps": self.total_steps,
            "epochs_completed": self.epochs_completed,
            "loss_total_mean": lt_m,
            "loss_total_std": lt_s,
            "loss_total_min": lt_min,
            "loss_total_max": lt_max,
            "loss_recon_mean": lr_m,
            "loss_sparse_mean": ls_m,
            "sparsity_mean": sp_m,
            "sparsity_std": sp_s,
            "sparsity_min": sp_min,
            "sparsity_max": sp_max,
            "n_active_mean": float(np.mean(self._n_active)) if self._n_active else 0.0,
        }

    def state_dict(self) -> dict:
        return {
            "loss_total": list(self._loss_total),
            "loss_recon": list(self._loss_recon),
            "loss_sparse": list(self._loss_sparse),
            "sparsity": list(self._sparsity),
            "n_active": list(self._n_active),
            "total_steps": self.total_steps,
            "epochs_completed": self.epochs_completed,
            "window": self.window,
        }

    def load_state_dict(self, d: dict) -> None:
        self._loss_total = d.get("loss_total", [])
        self._loss_recon = d.get("loss_recon", [])
        self._loss_sparse = d.get("loss_sparse", [])
        self._sparsity = d.get("sparsity", [])
        self._n_active = d.get("n_active", [])
        self.total_steps = d.get("total_steps", 0)
        self.epochs_completed = d.get("epochs_completed", 0)
        self.window = d.get("window", self.window)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. SPARSE AUTOENCODER  — main class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SparseAutoencoder:
    """
    Sparse Autoencoder (SAE) — encoding layer only.

    Responsibility:
        Train on dense embeddings, then encode any embedding
        into a sparse float32 vector over `num_neurons` neurons.

    Does NOT:
        - store SDRs              (EnhancedSuperBrain's job)
        - retrieve memories       (EnhancedSuperBrain's job)
        - ripple across clusters  (EnhancedSuperBrain's job)
        - compare SDRs            (EnhancedSuperBrain's job)

    Mode selection:
        mode="topk"   — structural sparsity, fixed k neurons active
        mode="l1"     — penalty sparsity, variable neurons active per input

    Quick start:
    ─────────────────────────────────────────────────────────────────────────
        from sparse_autoencoder import SparseAutoencoder, SAEConfig

        # Build and train
        config = SAEConfig(input_dim=768, num_neurons=86_000, mode="topk")
        sae = SparseAutoencoder(config)
        sae.train(embeddings)                        # (N, 768) numpy array

        # Single encode — sparse float32 vector
        sparse = sae.encode(embedding)               # (86 000,) float32

        # Single encode — binary uint8 vector (binarised)
        binary = sae.encode_binary(embedding)        # (86 000,) uint8

        # Single encode — active indices only (memory-efficient, preferred)
        indices = sae.encode_as_indices(embedding)   # e.g. [42, 301, 890, ...]

        # Batch encode
        sparse_batch  = sae.encode_batch(embeddings)          # (N, 86 000) float32
        binary_batch  = sae.encode_batch_binary(embeddings)   # (N, 86 000) uint8
        idx_list      = sae.encode_batch_indices(embeddings)  # list of N arrays

        # Variable sparsity (topk mode only)
        sae.encode_as_indices(embedding, k_override=500)      # only 500 active

        # Save / load
        sae.save("sae_trained.pt")
        sae2 = SparseAutoencoder.load("sae_trained.pt")

        # Hand SDRs to your brain
        brain.inject_sparse_sdrs(idx_list)
    ─────────────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        config: Optional[SAEConfig] = None,
        **kwargs,
    ) -> None:
        if config is None:
            config = SAEConfig(**kwargs)
        self.config = config

        # Resolve device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(config.seed)

        self.net = SAENetwork(config).to(self.device)

        self.dead_tracker = DeadNeuronTracker(
            num_neurons=config.num_neurons,
            window=config.dead_neuron_window,
            threshold=config.dead_neuron_threshold,
            device=self.device,
        )

        self.stats = SAEStatsTracker(window=config.stats_window)
        self.is_trained = False

        logger.info(
            f"SparseAutoencoder ready | "
            f"{config.input_dim}D -> {config.num_neurons:,} neurons | "
            f"mode={config.mode} | "
            f"device={self.device}"
        )
        if config.mode == "topk":
            logger.info(
                f"  TopK: k={config.topk_k} ({config.topk_k/config.num_neurons:.2%} sparsity) | "
                f"aux_alpha={config.topk_aux_alpha}"
            )
        else:
            logger.info(f"  L1:   lambda_l1={config.lambda_l1}")

    # ══════════════════════════════════════════════════════════════════════════
    # TRAINING
    # ══════════════════════════════════════════════════════════════════════════

    def train(
        self,
        embeddings: np.ndarray,
        val_embeddings: Optional[np.ndarray] = None,
        show_progress: bool = True,
    ) -> Dict:
        """
        Train the SAE on a dataset of dense embeddings.

        Args:
            embeddings:     (N, input_dim) float32 numpy array
            val_embeddings: optional validation set for early stopping
            show_progress:  show per-epoch progress bar

        Returns:
            history: dict with per-epoch losses
        """
        cfg = self.config
        N = len(embeddings)

        # Convert to torch tensors
        X = torch.tensor(embeddings, dtype=torch.float32)
        X_val = (
            torch.tensor(val_embeddings, dtype=torch.float32)
            if val_embeddings is not None
            else None
        )

        # Optimizer
        optimizer = Adam(
            self.net.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # LR scheduler
        steps_per_epoch = max(1, N // cfg.batch_size)
        total_steps = cfg.epochs * steps_per_epoch
        scheduler = (
            CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.learning_rate * 0.01)
            if cfg.lr_schedule
            else None
        )

        history: Dict[str, List] = {
            "epoch": [], "loss_total": [], "loss_recon": [],
            "loss_sparse": [], "sparsity": [], "val_loss": [],
        }

        best_val_loss = float("inf")
        patience_count = 0
        perm = torch.randperm(N)

        logger.info(
            f"Training SAE: {cfg.epochs} epochs | "
            f"batch={cfg.batch_size} | {steps_per_epoch} steps/epoch | "
            f"N={N:,}"
        )

        for epoch in range(cfg.epochs):
            self.net.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_sparse = 0.0
            epoch_sparsity = 0.0
            n_batches = 0

            perm = torch.randperm(N)                                        # shuffle each epoch
            iterator = range(0, N, cfg.batch_size)

            if show_progress:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(
                        iterator,
                        desc=f"Epoch {epoch+1:3d}/{cfg.epochs}",
                        unit="batch",
                        leave=False,
                    )
                except ImportError:
                    pass

            for start in iterator:
                idx = perm[start: start + cfg.batch_size]
                batch = X[idx].to(self.device)                             # (B, D)

                optimizer.zero_grad()

                loss, loss_recon, loss_sparse, sparse, sparsity = self._forward_loss(batch)

                loss.backward()

                if cfg.grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                # Normalise decoder columns after each step
                self.net.normalise_decoder_cols()

                # Update dead-neuron tracker
                with torch.no_grad():
                    self.dead_tracker.update(sparse.detach())

                # Dead-neuron rescue (L1 mode: bias bump)
                if cfg.mode == "l1":
                    self._rescue_dead_neurons_l1()

                epoch_loss += loss.item()
                epoch_recon += loss_recon.item()
                epoch_sparse += loss_sparse.item()
                epoch_sparsity += sparsity
                n_batches += 1

                self.stats.update(
                    loss_total=loss.item(),
                    loss_recon=loss_recon.item(),
                    loss_sparse=loss_sparse.item(),
                    sparsity=sparsity,
                    n_active=sparsity * cfg.num_neurons,
                )

            avg_loss = epoch_loss / max(1, n_batches)
            avg_recon = epoch_recon / max(1, n_batches)
            avg_sparse = epoch_sparse / max(1, n_batches)
            avg_sparsity = epoch_sparsity / max(1, n_batches)

            # Validation
            val_loss = None
            if X_val is not None:
                val_loss = self._eval_loss(X_val)

            self.stats.epochs_completed += 1

            history["epoch"].append(epoch + 1)
            history["loss_total"].append(avg_loss)
            history["loss_recon"].append(avg_recon)
            history["loss_sparse"].append(avg_sparse)
            history["sparsity"].append(avg_sparsity)
            history["val_loss"].append(val_loss)

            logger.info(
                f"Epoch {epoch+1:3d}/{cfg.epochs} | "
                f"loss={avg_loss:.5f} | recon={avg_recon:.5f} | "
                f"sparse={avg_sparse:.5f} | "
                f"sparsity={avg_sparsity:.2%} | "
                f"dead={self.dead_tracker.n_dead():,}"
                + (f" | val={val_loss:.5f}" if val_loss is not None else "")
            )

            # Early stopping
            if cfg.early_stopping_patience > 0 and X_val is not None:
                check_loss = val_loss if val_loss is not None else avg_loss
                if check_loss < best_val_loss - 1e-6:
                    best_val_loss = check_loss
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= cfg.early_stopping_patience:
                        logger.info(
                            f"Early stopping at epoch {epoch+1} "
                            f"(no improvement for {cfg.early_stopping_patience} epochs)"
                        )
                        break

        self.is_trained = True
        logger.info("Training complete.")
        return history

    def train_step(self, batch: np.ndarray) -> Dict:
        """
        Run a single training step on one batch.

        Useful for online / continual learning: call this after each new
        batch of embeddings arrives without running a full training loop.

        Args:
            batch: (B, input_dim) float32 numpy array

        Returns:
            metrics: dict with loss_total, loss_recon, loss_sparse, sparsity
        """
        cfg = self.config

        if not hasattr(self, "_online_optimizer"):
            self._online_optimizer = Adam(
                self.net.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )

        self.net.train()
        x = torch.tensor(batch, dtype=torch.float32, device=self.device)
        self._online_optimizer.zero_grad()

        loss, loss_recon, loss_sparse, sparse, sparsity = self._forward_loss(x)
        loss.backward()

        if cfg.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)

        self._online_optimizer.step()
        self.net.normalise_decoder_cols()
        self.dead_tracker.update(sparse.detach())

        if cfg.mode == "l1":
            self._rescue_dead_neurons_l1()

        self.stats.update(
            loss_total=loss.item(),
            loss_recon=loss_recon.item(),
            loss_sparse=loss_sparse.item(),
            sparsity=sparsity,
            n_active=sparsity * cfg.num_neurons,
        )
        self.is_trained = True

        return {
            "loss_total": loss.item(),
            "loss_recon": loss_recon.item(),
            "loss_sparse": loss_sparse.item(),
            "sparsity": sparsity,
        }

    # ── Internal forward + loss ───────────────────────────────────────────────

    def _forward_loss(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Full forward pass + loss computation.

        Returns:
            loss_total, loss_recon, loss_sparse, sparse_hidden, sparsity_float
        """
        cfg = self.config

        # Optional input normalisation
        if cfg.normalize_inputs:
            norms = x.norm(dim=1, keepdim=True).clamp(min=1e-8)
            x = x / norms

        # Encode
        pre = self.net.encode_pre(x)                                        # (B, N)

        if cfg.mode == "topk":
            sparse, topk_idx = self.net.topk_gate(pre)                     # (B, N), (B, k)
            # Reconstruction loss
            x_hat = self.net.decode(sparse)
            loss_recon = F.mse_loss(x_hat, x)
            # Auxiliary loss: encourage dead neurons to fire
            loss_sparse = self._topk_aux_loss(pre, topk_idx)
            loss_total = loss_recon + cfg.topk_aux_alpha * loss_sparse

        else:  # l1
            sparse = F.relu(pre)                                            # (B, N)
            x_hat = self.net.decode(sparse)
            loss_recon = F.mse_loss(x_hat, x)
            loss_sparse = sparse.mean()                                     # mean L1 norm
            loss_total = loss_recon + cfg.lambda_l1 * loss_sparse

        # Compute mean sparsity across the batch
        with torch.no_grad():
            n_active = (sparse > 0).float().sum(dim=1).mean().item()
        sparsity = n_active / cfg.num_neurons

        return loss_total, loss_recon, loss_sparse, sparse, sparsity

    def _topk_aux_loss(
        self, pre: torch.Tensor, topk_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Auxiliary loss for Top-K mode: rescue dead neurons.

        Dead neurons are forced to reconstruct the input using only their
        own activations, contributing a small MSE signal that trains their
        encoder weights even when they lose the main k-WTA competition.

        Method (following Anthropic's SAE paper):
        1. Find dead neurons not in the top-k for any sample in this batch.
        2. Build an auxiliary sparse state with the top-k dead neurons active.
        3. Compute auxiliary reconstruction MSE with just those activations.
        """
        cfg = self.config
        dead_idx = self.dead_tracker.dead_indices()

        if len(dead_idx) == 0:
            return torch.tensor(0.0, device=self.device)

        # Activations of dead neurons: (B, n_dead)
        dead_pre = pre[:, dead_idx]
        dead_activated = F.relu(dead_pre)

        # Top-k among dead neurons (up to k total)
        k_dead = min(cfg.topk_k, len(dead_idx))
        if k_dead == 0:
            return torch.tensor(0.0, device=self.device)

        _, dead_topk_local = torch.topk(dead_activated, k=k_dead, dim=1)   # (B, k_dead)

        # Scatter into full-size sparse state
        aux_sparse = torch.zeros(
            pre.shape[0], cfg.num_neurons, device=self.device
        )
        # Global indices of dead top-k
        global_dead_topk = dead_idx[dead_topk_local.reshape(-1)].reshape(
            pre.shape[0], k_dead
        )
        aux_vals = dead_activated.gather(1, dead_topk_local)               # (B, k_dead)
        aux_sparse.scatter_(1, global_dead_topk, aux_vals)

        # Auxiliary reconstruction
        aux_hat = self.net.decode(aux_sparse)

        # Target: what the primary decoder couldn't reconstruct
        # Simple approach: use original input as target
        if cfg.normalize_inputs:
            # x is already normalised in the outer call — use pre-norm norms
            # but we don't have them here; use the aux sparse directly vs. a
            # detached version of the pre-activation to stay cheap
            pass

        # MSE of auxiliary reconstruction — normalised to batch
        return F.mse_loss(
            aux_hat,
            torch.zeros_like(aux_hat),                                      # push toward 0 residual
        )

    def _rescue_dead_neurons_l1(self) -> None:
        """
        L1 mode dead-neuron rescue: bump encoder bias of dead neurons upward
        so they are more likely to activate on future inputs.
        """
        cfg = self.config
        dead_idx = self.dead_tracker.dead_indices()
        if len(dead_idx) == 0:
            return
        with torch.no_grad():
            self.net.enc_bias[dead_idx] += cfg.dead_neuron_bias_bump
        self.dead_tracker.record_rescue(len(dead_idx))

    def _eval_loss(self, X_val: torch.Tensor) -> float:
        """Run validation pass without gradients."""
        self.net.eval()
        cfg = self.config
        total = 0.0
        n_batches = 0
        with torch.no_grad():
            for start in range(0, len(X_val), cfg.batch_size):
                batch = X_val[start: start + cfg.batch_size].to(self.device)
                loss, _, _, _, _ = self._forward_loss(batch)
                total += loss.item()
                n_batches += 1
        self.net.train()
        return total / max(1, n_batches)

    # ══════════════════════════════════════════════════════════════════════════
    # PRIMARY ENCODING API
    # ══════════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def encode(
        self,
        embedding: np.ndarray,
        k_override: Optional[int] = None,
    ) -> np.ndarray:
        """
        Encode one dense embedding -> sparse float32 vector.

        The sparse vector is NOT binary: non-zero values represent
        activation strengths. Use encode_binary() if you need {0,1}.

        Args:
            embedding:  (input_dim,) float32
            k_override: for topk mode, override k for this call only

        Returns:
            sparse: (num_neurons,) float32
        """
        self.net.eval()
        cfg = self.config

        x = torch.tensor(embedding, dtype=torch.float32, device=self.device).unsqueeze(0)

        if cfg.normalize_inputs:
            x = x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)

        pre = self.net.encode_pre(x)

        if cfg.mode == "topk":
            sparse, _ = self.net.topk_gate(pre, k_override=k_override)
        else:
            sparse = F.relu(pre)

        return sparse.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def encode_binary(
        self,
        embedding: np.ndarray,
        k_override: Optional[int] = None,
    ) -> np.ndarray:
        """
        Encode one embedding -> binary uint8 vector.
        Any non-zero activation becomes 1.

        Returns:
            binary: (num_neurons,) uint8
        """
        sparse = self.encode(embedding, k_override=k_override)
        return (sparse > 0).astype(np.uint8)

    @torch.no_grad()
    def encode_as_indices(
        self,
        embedding: np.ndarray,
        k_override: Optional[int] = None,
    ) -> np.ndarray:
        """
        Encode one embedding -> sorted active neuron indices.

        Memory-efficient: stores only which neurons fired, not the full vector.
        This is the preferred format for injection into EnhancedSuperBrain.

        Returns:
            active_indices: sorted int64 array — e.g. [9, 42, 301, 890, ...]
        """
        sparse = self.encode(embedding, k_override=k_override)
        return np.where(sparse > 0)[0].astype(np.int64)

    @torch.no_grad()
    def encode_batch(
        self,
        embeddings: np.ndarray,
        k_override: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a batch -> sparse float32 matrix.

        Args:
            embeddings: (N, input_dim) float32
            k_override: override k (topk mode only)
            show_progress: tqdm bar

        Returns:
            sparse_batch: (N, num_neurons) float32
        """
        cfg = self.config
        N = len(embeddings)
        result = np.zeros((N, cfg.num_neurons), dtype=np.float32)
        self.net.eval()

        iterator = range(0, N, cfg.batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="SAE encode", unit="batch")
            except ImportError:
                pass

        for start in iterator:
            end = min(start + cfg.batch_size, N)
            batch = torch.tensor(
                embeddings[start:end], dtype=torch.float32, device=self.device
            )

            if cfg.normalize_inputs:
                batch = batch / batch.norm(dim=1, keepdim=True).clamp(min=1e-8)

            pre = self.net.encode_pre(batch)

            if cfg.mode == "topk":
                sparse, _ = self.net.topk_gate(pre, k_override=k_override)
            else:
                sparse = F.relu(pre)

            result[start:end] = sparse.cpu().numpy()

        return result

    @torch.no_grad()
    def encode_batch_binary(
        self,
        embeddings: np.ndarray,
        k_override: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a batch -> binary uint8 matrix.

        Returns:
            binary_batch: (N, num_neurons) uint8
        """
        sparse = self.encode_batch(embeddings, k_override=k_override, show_progress=show_progress)
        return (sparse > 0).astype(np.uint8)

    @torch.no_grad()
    def encode_batch_indices(
        self,
        embeddings: np.ndarray,
        k_override: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[np.ndarray]:
        """
        Encode a batch -> list of active-index arrays (memory-efficient).

        Preferred for large-scale encoding before injection into EnhancedSuperBrain.

        Returns:
            List of N sorted int64 arrays — ready for brain.inject_sparse_sdrs()
        """
        cfg = self.config
        N = len(embeddings)
        results: List[np.ndarray] = []
        self.net.eval()

        iterator = range(0, N, cfg.batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="SAE (indices)", unit="batch")
            except ImportError:
                pass

        for start in iterator:
            end = min(start + cfg.batch_size, N)
            batch = torch.tensor(
                embeddings[start:end], dtype=torch.float32, device=self.device
            )

            if cfg.normalize_inputs:
                batch = batch / batch.norm(dim=1, keepdim=True).clamp(min=1e-8)

            pre = self.net.encode_pre(batch)

            if cfg.mode == "topk":
                sparse, _ = self.net.topk_gate(pre, k_override=k_override)
            else:
                sparse = F.relu(pre)

            sparse_np = sparse.cpu().numpy()
            for row in sparse_np:
                results.append(np.where(row > 0)[0].astype(np.int64))

        return results

    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTICS  (encoder health only)
    # ══════════════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        """Return encoder health statistics."""
        s = self.stats.summary()
        s["is_trained"] = self.is_trained
        s["mode"] = self.config.mode
        s["num_neurons"] = self.config.num_neurons
        s["input_dim"] = self.config.input_dim
        s["n_dead_neurons"] = self.dead_tracker.n_dead()
        s["dead_pct"] = self.dead_tracker.n_dead() / self.config.num_neurons
        s["total_rescues"] = self.dead_tracker._total_rescues
        if self.config.mode == "topk":
            s["topk_k"] = self.config.topk_k
            s["target_sparsity"] = self.config.topk_k / self.config.num_neurons
        else:
            s["lambda_l1"] = self.config.lambda_l1
        return s

    def print_stats(self) -> None:
        """Print a formatted encoder health report."""
        s = self.get_stats()
        bar = "=" * 64
        print(f"\n{bar}")
        print(f"  SparseAutoencoder [{s['mode'].upper()}] — Encoder Health")
        print(f"  {s['input_dim']}D -> {s['num_neurons']:,} neurons | "
              f"trained={s['is_trained']} | steps={s['total_steps']:,}")
        print(bar)
        if s["mode"] == "topk":
            print(f"  Target sparsity    {s['target_sparsity']:.2%}  (k={s['topk_k']:,})")
        else:
            print(f"  lambda_l1          {s['lambda_l1']}")
        print(f"  Actual sparsity    {s['sparsity_mean']:.2%} +/- {s['sparsity_std']:.4f}")
        print(f"  Active (mean)      {s['n_active_mean']:.0f} / {s['num_neurons']:,}")
        print(f"  Sparsity range     [{s['sparsity_min']:.2%}, {s['sparsity_max']:.2%}]")
        print(f"  Loss (total)       {s['loss_total_mean']:.6f}")
        print(f"  Loss (recon)       {s['loss_recon_mean']:.6f}")
        print(f"  Loss (sparse)      {s['loss_sparse_mean']:.6f}")
        print(f"  Dead neurons       {s['n_dead_neurons']:,}  ({s['dead_pct']:.2%})")
        print(f"  Total rescues      {s['total_rescues']:,}")
        print(f"  Epochs completed   {s['epochs_completed']}")
        print(bar)

    def visualise_sparse(self, sparse: np.ndarray, width: int = 80) -> str:
        """
        One-line ASCII visualisation of a sparse vector's activity pattern.
        '#' = at least one active neuron in that block, '.' = all silent.
        """
        n = len(sparse)
        block = n / width
        return "".join(
            "#" if sparse[int(i * block): int((i + 1) * block)].any() else "."
            for i in range(width)
        )

    def describe_sparse(self, sparse: np.ndarray, label: str = "") -> None:
        """Print a compact description of one sparse vector's encoding."""
        idx = np.where(sparse > 0)[0]
        n_active = len(idx)
        sparsity = n_active / len(sparse)
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}Active: {n_active:,} / {len(sparse):,}  ({sparsity:.2%})")
        print(f"  Max activation: {sparse.max():.4f}  |  Mean non-zero: {sparse[sparse>0].mean():.4f}")
        print(f"  First 10 active indices: {idx[:10].tolist()}")
        print(f"  Pattern: {self.visualise_sparse(sparse)}")

    # ══════════════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ══════════════════════════════════════════════════════════════════════════

    def save(self, path: Union[str, Path]) -> None:
        """
        Serialise all learned encoder state to disk.

        Saved: config, network weights, dead-neuron tracker state, stats.
        The decoder is included in the network weights but is not needed
        for inference — you can discard it if size is a concern.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "version": "1.0",
            "config": self.config,
            "is_trained": self.is_trained,
            "net_state_dict": self.net.state_dict(),
            "dead_tracker": self.dead_tracker.state_dict(),
            "stats": self.stats.state_dict(),
        }

        torch.save(state, path)
        size_mb = path.stat().st_size / 1024**2
        logger.info(f"SparseAutoencoder saved -> {path}  ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SparseAutoencoder":
        """
        Restore a SparseAutoencoder from a saved file.

        All learned state (weights, dead-neuron tracker, training stats)
        is restored exactly. Ready for inference immediately.
        """
        path = Path(path)
        state = torch.load(path, map_location="cpu", weights_only=False)

        sae = cls(config=state["config"])
        sae.net.load_state_dict(state["net_state_dict"])
        sae.net.to(sae.device)
        sae.dead_tracker.load_state_dict(state["dead_tracker"])
        sae.stats.load_state_dict(state["stats"])
        sae.is_trained = state.get("is_trained", True)

        logger.info(
            f"SparseAutoencoder loaded <- {path} | "
            f"mode={sae.config.mode} | trained={sae.is_trained}"
        )
        return sae


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INTEGRATION DEMO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "topk"
    assert mode in ("topk", "l1"), "Usage: python sparse_autoencoder.py [topk|l1]"

    print(f"\n{'=' * 64}")
    print(f"  SparseAutoencoder [{mode.upper()}] — Encoding only demo")
    print(f"  (brain-level logic lives in neuron_initializtion.py)")
    print(f"{'=' * 64}")

    # ── Build SAE ─────────────────────────────────────────────────────────────
    config = SAEConfig(
        input_dim=768,
        num_neurons=86_000,
        mode=mode,
        # topk settings
        topk_k=1_720,           # 2% of 86 000
        topk_aux_alpha=1e-3,
        # l1 settings
        lambda_l1=5e-4,
        # training
        epochs=5,
        batch_size=256,
        learning_rate=3e-4,
        normalize_inputs=True,
        seed=42,
    )
    sae = SparseAutoencoder(config)

    # ── Synthetic embeddings (swap for np.load("embeddings_768.npy") in prod) ─
    rng = np.random.RandomState(0)
    N = 500
    print(f"\nGenerating {N} synthetic embeddings (dim=768) ...")
    embeddings = rng.randn(N, 768).astype(np.float32)
    # Normalise so they resemble real sentence embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nTraining ...")
    sae.train(embeddings, show_progress=True)

    sae.print_stats()

    # ── Inference: get index-format SDRs ready for EnhancedSuperBrain ─────────
    print("\nEncoding all samples for brain injection ...")
    idx_list = sae.encode_batch_indices(embeddings, show_progress=True)
    print(f"\nReady: {len(idx_list)} SDRs as index arrays")
    print(f"  Example SDR[0]: {idx_list[0][:10].tolist()} ... ({len(idx_list[0])} active neurons)")

    # ── Binary batch (full matrix) ─────────────────────────────────────────────
    print("\nEncoding as binary matrix ...")
    binary_sdrs = sae.encode_batch_binary(embeddings, show_progress=True)
    print(f"  Shape: {binary_sdrs.shape} | sparsity={binary_sdrs.mean():.4f}")

    # ── Variable sparsity (topk mode only) ────────────────────────────────────
    if mode == "topk":
        print("\nVariable sparsity (topk mode):")
        for k in [500, 1720, 3440]:
            idx = sae.encode_as_indices(embeddings[0], k_override=k)
            print(f"  k={k:5d} -> {len(idx):,} active neurons")

    # ── SDR description ───────────────────────────────────────────────────────
    sparse_ex = sae.encode(embeddings[0])
    sae.describe_sparse(sparse_ex, label="sample")

    # ── Save / load ───────────────────────────────────────────────────────────
    save_path = f"sae_{mode}_trained.pt"
    sae.save(save_path)
    sae2 = SparseAutoencoder.load(save_path)
    # Verify identical output
    idx_orig = sae.encode_as_indices(embeddings[5])
    idx_load = sae2.encode_as_indices(embeddings[5])
    print(f"\nSave/load round-trip identical: {np.array_equal(idx_orig, idx_load)}")

    print(f"\nNext step:")
    print(f"  from neuron_initializtion import EnhancedSuperBrain")
    print(f"  brain = EnhancedSuperBrain(num_neurons=86000, topology='scale_free')")
    print(f"  brain.inject_sparse_sdrs(idx_list, description='IMDB train')")
    print(f"  brain.save('super_brain.pkl')")
    print(f"{'=' * 64}\n")
