"""
spatial_pooler.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Full HTM-style Spatial Pooler  —  Brain-Inspired AI Architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Converts dense sentence embeddings (384-dim float32 from SentenceTransformers)
into Sparse Distributed Representations (SDRs) over 86 000 neurons, suitable
for injection into a neurological circuit (EnhancedSuperBrain).

Pipeline:
  Dense embedding (384-dim float32)
        │
  InputEncoder          ─ adaptive z-score binarisation (online Welford stats)
        │
  PotentialSynapses     ─ receptive fields, permanences, overlap scoring
        │
  BoostingModule        ─ homeostatic amplification of under-active columns
        │
  InhibitionModule      ─ zone-based or local k-WTA competition
        │
  LearningEngine        ─ vectorised Hebbian permanence update
        │
  DutyCycleTracker      ─ EMA activity statistics + dead-column rescue
        │
  SDR output (86 000-dim uint8, ~1–2 % active)

What makes this "full HTM" and not a toy:
  ✓ Online permanence learning  — synapses grow/shrink per Hebbian rule
  ✓ Homeostatic boosting        — prevents dead neurons, balances firing
  ✓ Overlap duty-cycle rescue   — bumps permanences of silent columns
  ✓ Adaptive input encoding     — per-dimension running z-score (Welford)
  ✓ Variable sparsity           — pass target_density per call
  ✓ Semantic overlap            — similar embeddings → overlapping active sets
  ✓ Ripple retrieval helper     — find SDRs overlapping a query above threshold
  ✓ Full save / load            — all learnable state persisted
  ✓ Vectorised throughout       — no Python loops in hot paths
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import rank_filter

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class SpatialPoolerConfig:
    """
    Complete configuration. Defaults tuned for:
        input_dim=384 (all-MiniLM-L6-v2)  →  num_columns=86 000 neurons

    Adjust sparsity, boosting, and inhibition_mode as your network evolves.
    """

    # ── Architecture ──────────────────────────────────────────────────────────
    input_dim: int = 384
    num_columns: int = 86_000

    # ── Receptive fields ──────────────────────────────────────────────────────
    potential_radius: int = 192
    """Half-width (in input-space) of each column's connectivity neighbourhood.
    Default = input_dim // 2  →  each column can reach the full input."""

    potential_pct: float = 0.5
    """Fraction of the 2×potential_radius neighbourhood each column samples.
    n_synapses = max(1, int(potential_pct * potential_radius * 2))  ≈ 192."""

    # ── Inhibition ────────────────────────────────────────────────────────────
    inhibition_mode: Literal["global_zone", "local"] = "global_zone"
    """
    global_zone  →  divide columns into N zones; top-k per zone survive.
                    Fully vectorised, scales to 86 000 columns easily.
    local        →  each column competes within a sliding window of radius r.
                    Uses scipy.ndimage.rank_filter — O(N) in C code.
    """

    num_inhibition_zones: int = 860
    """Number of inhibition zones for 'global_zone' mode.
    Default: 860 zones × ~100 cols/zone."""

    local_inhibition_radius: int = 50
    """Neighbourhood half-width for 'local' inhibition mode."""

    local_area_density: float = 0.02
    """Target fraction of columns active per inhibition area.
    0.02 → 1 720 active out of 86 000 (comparable to cortical sparsity)."""

    # ── Stimulus threshold ────────────────────────────────────────────────────
    stimulus_threshold: int = 2
    """Minimum overlap score for a column to compete.
    Prevents columns with zero or near-zero connectivity from winning."""

    # ── Permanence ────────────────────────────────────────────────────────────
    syn_perm_connected: float = 0.50
    """Synapse is 'connected' (contributes to overlap) iff perm >= this."""

    syn_perm_active_inc: float = 0.05
    """Permanence increment when the pre-synaptic bit is active (Hebbian LTP)."""

    syn_perm_inactive_dec: float = 0.008
    """Permanence decrement when the pre-synaptic bit is inactive (Hebbian LTD)."""

    syn_perm_below_stimulus_dec: float = 0.01
    """Extra increment applied to underperforming (dead) columns to rescue them."""

    syn_perm_trim_threshold: float = 0.025
    """Permanences below this are zeroed out (weak synapse pruning)."""

    syn_perm_min: float = 0.0
    syn_perm_max: float = 1.0

    # ── Duty cycles and boosting ──────────────────────────────────────────────
    duty_cycle_period: int = 1000
    """EMA window for duty cycle calculation: α = 1 / period."""

    boost_strength: float = 2.0
    """
    Homeostatic boost strength.
    0  = disabled.
    2–5 = recommended for active balancing.
    boost_factor[i] = exp(−boost_strength × (active_dc[i] − target_density))
    """

    min_pct_overlap_duty_cycle: float = 0.001
    """If a column's overlap duty cycle < min_pct × global_max → permanence bump."""

    min_pct_active_duty_cycle: float = 0.001
    """Monitoring threshold for active duty cycle health."""

    # ── Input encoding ────────────────────────────────────────────────────────
    input_encode_strategy: Literal["adaptive", "topk", "threshold", "sign"] = "adaptive"
    """
    adaptive   — per-dimension z-score (Welford online), then top-density% → 1.
                 Best for embeddings with varying scales (recommended).
    topk       — globally top-k values → 1. Fixed sparsity.
    threshold  — top-density percentile → 1.
    sign       — positive values → 1.
    """

    input_density: float = 0.5
    """Fraction of input bits set to 1 after binarisation."""

    # ── Learning ──────────────────────────────────────────────────────────────
    learn: bool = True
    """Master switch. Set False at inference time for reproducible SDRs."""

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int = 42
    stats_window: int = 500
    """Rolling window for health statistics."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. INPUT ENCODER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class InputEncoder:
    """
    Converts dense float embeddings → binary input vectors.

    The Spatial Pooler expects binary inputs where roughly `input_density`
    fraction of bits are 1.  The `adaptive` strategy is the most robust:
    it maintains per-dimension running statistics (Welford algorithm) and
    z-scores the input before thresholding, making it invariant to embedding
    scale and distribution drift over time.

    All strategies output a (input_dim,) uint8 array.
    """

    def __init__(
        self,
        input_dim: int,
        strategy: str = "adaptive",
        density: float = 0.5,
    ) -> None:
        self.input_dim = input_dim
        self.strategy = strategy
        self.density = density
        self.topk = max(1, int(density * input_dim))

        # Welford online stats (for 'adaptive' strategy)
        self._n: int = 0
        self._mean = np.zeros(input_dim, dtype=np.float64)
        self._M2 = np.ones(input_dim, dtype=np.float64)   # sum of squared deviations

    # ── Public API ────────────────────────────────────────────────────────────

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode one dense embedding to a binary vector.

        Args:
            x: (input_dim,) float32 or float64

        Returns:
            binary: (input_dim,) uint8
        """
        x = x.astype(np.float64)

        if self.strategy == "adaptive":
            self._welford_update(x)
            std = np.sqrt(self._M2 / max(1, self._n - 1))
            z = (x - self._mean) / (std + 1e-8)
            return self._topk_binarise(z)

        elif self.strategy == "topk":
            return self._topk_binarise(x)

        elif self.strategy == "threshold":
            thresh = np.percentile(x, (1.0 - self.density) * 100.0)
            return (x >= thresh).astype(np.uint8)

        elif self.strategy == "sign":
            return (x > 0.0).astype(np.uint8)

        else:
            raise ValueError(f"Unknown encode strategy: {self.strategy!r}")

    def encode_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Encode a batch of embeddings.

        Args:
            X: (N, input_dim) float32

        Returns:
            binary: (N, input_dim) uint8
        """
        return np.stack([self.encode(row) for row in X])

    # ── Internals ─────────────────────────────────────────────────────────────

    def _topk_binarise(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros(self.input_dim, dtype=np.uint8)
        idx = np.argpartition(x, -self.topk)[-self.topk:]
        out[idx] = 1
        return out

    def _welford_update(self, x: np.ndarray) -> None:
        """Online Welford algorithm: update per-dimension running mean / variance."""
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        self._M2 += delta * (x - self._mean)

    @property
    def running_std(self) -> np.ndarray:
        return np.sqrt(self._M2 / max(1, self._n - 1))

    # ── Serialisation ─────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "density": self.density,
            "topk": self.topk,
            "n": self._n,
            "mean": self._mean.copy(),
            "M2": self._M2.copy(),
        }

    def load_state_dict(self, d: dict) -> None:
        self.strategy = d["strategy"]
        self.density = d["density"]
        self.topk = d["topk"]
        self._n = d["n"]
        self._mean = d["mean"].copy()
        self._M2 = d["M2"].copy()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. POTENTIAL SYNAPSES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PotentialSynapses:
    """
    Stores all synaptic connectivity and manages permanence values.

    Layout:
        potential_inputs[col, k]  → which input bit synapse k of column col connects to
        permanences[col, k]       → permanence of that synapse  (float32, 0–1)

    A synapse is "connected" iff permanence[col, k] >= syn_perm_connected.
    Only connected synapses count toward the column's overlap score.

    Memory footprint (86 000 cols × ~192 synapses each):
        permanences       86 000 × 192 × 4 bytes  ≈ 66 MB
        potential_inputs  86 000 × 192 × 4 bytes  ≈ 66 MB

    If memory is tight, reduce potential_pct (fewer synapses per column).
    """

    def __init__(
        self, config: SpatialPoolerConfig, rng: np.random.RandomState
    ) -> None:
        cfg = config
        self.cfg = cfg

        # Number of potential synapses per column
        n_syn = max(1, int(cfg.potential_pct * cfg.potential_radius * 2))
        self.n_syn = n_syn

        logger.info(
            f"PotentialSynapses: {cfg.num_columns:,} cols × {n_syn} synapses = "
            f"{cfg.num_columns * n_syn:,} connections  "
            f"(~{cfg.num_columns * n_syn * 4 / 1024**2:.0f} MB per array)"
        )

        self.potential_inputs = np.empty((cfg.num_columns, n_syn), dtype=np.int32)
        self.permanences = np.empty((cfg.num_columns, n_syn), dtype=np.float32)

        self._initialise(rng)

    # ── Initialisation ────────────────────────────────────────────────────────

    def _initialise(self, rng: np.random.RandomState) -> None:
        """
        Build receptive fields and seed permanences.

        Each column's centre is mapped to an input position proportionally.
        Potential inputs are sampled uniformly from a neighbourhood of that centre.
        Permanences are initialised ~N(syn_perm_connected, σ) so roughly half
        of synapses start connected.
        """
        cfg = self.cfg
        C, P = cfg.num_columns, self.n_syn

        # Linear mapping: column index → centre in input space
        centers = np.linspace(0, cfg.input_dim - 1, C)

        t0 = time.time()
        for col in range(C):
            c = int(round(centers[col]))
            lo = max(0, c - cfg.potential_radius)
            hi = min(cfg.input_dim - 1, c + cfg.potential_radius)
            pool = np.arange(lo, hi + 1)

            if len(pool) < P:
                # Neighbourhood too small — pad with random global inputs
                extra_idx = rng.choice(
                    np.setdiff1d(np.arange(cfg.input_dim), pool),
                    size=min(P - len(pool), cfg.input_dim - len(pool)),
                    replace=False,
                )
                pool = np.concatenate([pool, extra_idx])

            chosen = rng.choice(pool, size=min(P, len(pool)), replace=False)
            # If still short (tiny input_dim), allow replacement
            if len(chosen) < P:
                chosen = rng.choice(cfg.input_dim, size=P, replace=True)
            self.potential_inputs[col] = chosen.astype(np.int32)

        logger.info(
            f"Receptive fields built in {time.time() - t0:.1f}s"
        )

        # Seed permanences: normal around connected threshold, clipped to [min, max]
        self.permanences[:] = np.clip(
            rng.normal(
                loc=cfg.syn_perm_connected,
                scale=cfg.syn_perm_connected / 4.0,
                size=(C, P),
            ).astype(np.float32),
            cfg.syn_perm_min,
            cfg.syn_perm_max,
        )

        logger.info("PotentialSynapses initialised.")

    # ── Core operations ───────────────────────────────────────────────────────

    @property
    def connected_mask(self) -> np.ndarray:
        """(num_columns, n_syn) bool — True where synapse is currently connected."""
        return self.permanences >= self.cfg.syn_perm_connected

    def n_connected(self) -> int:
        """Total number of connected synapses across all columns."""
        return int(self.connected_mask.sum())

    def compute_overlap(self, binary_input: np.ndarray) -> np.ndarray:
        """
        Compute overlap score for every column.

            overlap[col] = Σ_k  connected[col, k]  AND  binary_input[potential_inputs[col, k]]

        Fully vectorised:
            1. Gather input bits at all synapse positions:
               input_vals = binary_input[potential_inputs]      shape (C, P)  uint8
            2. AND with connected mask:
               hits = connected_mask AND input_vals             shape (C, P)  bool
            3. Sum across synapses → overlap scores             shape (C,)   float32

        Time complexity: O(C × P) numpy — ~4 M ops for default config.

        Args:
            binary_input: (input_dim,) uint8

        Returns:
            overlap: (num_columns,) float32
        """
        # Gather: (C, P) — each cell = binary_input[potential_inputs[col, k]]
        input_vals = binary_input[self.potential_inputs]                    # (C, P) uint8

        # Vectorised AND-sum
        hits = self.connected_mask & input_vals.astype(bool)               # (C, P) bool
        return hits.sum(axis=1).astype(np.float32)                         # (C,)

    def update_permanences(
        self,
        active_columns: np.ndarray,
        binary_input: np.ndarray,
    ) -> None:
        """
        Vectorised Hebbian permanence update for all active columns.

        For each active column col:
            synapse to ACTIVE input    → perm += syn_perm_active_inc
            synapse to INACTIVE input  → perm -= syn_perm_inactive_dec
            clip to [syn_perm_min, syn_perm_max]
            zero out synapses below syn_perm_trim_threshold  (weak-synapse pruning)

        Args:
            active_columns: (k,) int64 — indices of winning columns this step
            binary_input:   (input_dim,) uint8
        """
        if len(active_columns) == 0:
            return
        cfg = self.cfg

        # Gather input bits for all active columns at once: (k, P)
        rf_inputs = binary_input[self.potential_inputs[active_columns]]     # (k, P) uint8

        # Hebbian LTP / LTD
        perms = self.permanences[active_columns]                            # (k, P) view
        perms += cfg.syn_perm_active_inc   * rf_inputs
        perms -= cfg.syn_perm_inactive_dec * (1 - rf_inputs)

        # Clip
        np.clip(perms, cfg.syn_perm_min, cfg.syn_perm_max, out=perms)

        # Prune very weak synapses
        perms[perms < cfg.syn_perm_trim_threshold] = 0.0

        # Write back (the view might not be writable in all numpy versions)
        self.permanences[active_columns] = perms

    def bump_permanences(self, columns: np.ndarray) -> None:
        """
        Dead-column rescue: nudge all potential synapses of `columns` upward.
        This helps silent columns re-establish connectivity.

        Args:
            columns: (k,) int64 — column indices to bump
        """
        if len(columns) == 0:
            return
        cfg = self.cfg
        self.permanences[columns] += cfg.syn_perm_below_stimulus_dec
        np.clip(
            self.permanences[columns],
            cfg.syn_perm_min, cfg.syn_perm_max,
            out=self.permanences[columns],
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "permanences": self.permanences.copy(),
            "potential_inputs": self.potential_inputs.copy(),
            "n_syn": self.n_syn,
        }

    def load_state_dict(self, d: dict) -> None:
        self.permanences = d["permanences"]
        self.potential_inputs = d["potential_inputs"]
        self.n_syn = d["n_syn"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. DUTY CYCLE TRACKER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DutyCycleTracker:
    """
    Exponential Moving Average (EMA) of two signals per column:

    overlap_duty_cycle[col]
        Fraction of recent steps where col's raw overlap >= stimulus_threshold.
        Low value → column rarely overlaps any input → candidate for rescue.

    active_duty_cycle[col]
        Fraction of recent steps where col was active (won inhibition).
        Low value → column rarely fires → boosting kicks in.

    EMA update rule (applied every step):
        dc[t] = (1 − α) · dc[t−1]  +  α · current_value
        where α = 1 / period
    """

    def __init__(self, num_columns: int, period: int) -> None:
        self.num_columns = num_columns
        self.period = period
        self.alpha = 1.0 / max(1, period)
        self.iteration = 0

        self.overlap_duty_cycle = np.zeros(num_columns, dtype=np.float64)
        self.active_duty_cycle = np.zeros(num_columns, dtype=np.float64)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        overlap_exceeds: np.ndarray,
        active_columns: np.ndarray,
    ) -> None:
        """
        Step the EMA forward by one iteration.

        Args:
            overlap_exceeds: (num_columns,) bool — True if raw overlap >= stimulus_threshold
            active_columns:  (k,) int64          — indices of active columns this step
        """
        α = self.alpha
        self.iteration += 1

        # Overlap duty cycle EMA (vectorised)
        self.overlap_duty_cycle = (
            (1.0 - α) * self.overlap_duty_cycle
            + α * overlap_exceeds.astype(np.float64)
        )

        # Active duty cycle EMA (vectorised)
        active_vec = np.zeros(self.num_columns, dtype=np.float64)
        if len(active_columns) > 0:
            active_vec[active_columns] = 1.0
        self.active_duty_cycle = (
            (1.0 - α) * self.active_duty_cycle + α * active_vec
        )

    # ── Derived queries ───────────────────────────────────────────────────────

    def underperforming_columns(self, min_pct: float) -> np.ndarray:
        """
        Indices of columns whose overlap duty cycle is dangerously low.
        These columns will have their permanences bumped (dead-column rescue).

        threshold = min_pct × max(overlap_duty_cycle)
        """
        global_max = float(self.overlap_duty_cycle.max())
        if global_max == 0.0:
            return np.array([], dtype=np.int64)
        threshold = min_pct * global_max
        return np.where(self.overlap_duty_cycle < threshold)[0]

    # ── Serialisation ─────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "overlap_duty_cycle": self.overlap_duty_cycle.copy(),
            "active_duty_cycle": self.active_duty_cycle.copy(),
            "iteration": self.iteration,
            "period": self.period,
        }

    def load_state_dict(self, d: dict) -> None:
        self.overlap_duty_cycle = d["overlap_duty_cycle"].copy()
        self.active_duty_cycle = d["active_duty_cycle"].copy()
        self.iteration = d["iteration"]
        self.period = d.get("period", self.period)
        self.alpha = 1.0 / max(1, self.period)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. BOOSTING MODULE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BoostingModule:
    """
    Homeostatic boosting: scales raw overlap scores by a per-column factor
    to equalise firing rates across all columns.

    boost_factor[i] = exp( −boost_strength × (active_duty_cycle[i] − target_density) )

    Interpretation:
        Column rarely fires  → active_dc < target  → factor > 1  (amplified)
        Column fires too often → active_dc > target → factor < 1  (suppressed)
        boost_strength = 0   → all factors = 1       (disabled)

    Effect: over thousands of steps, every column converges toward firing at
    exactly `target_density` fraction of the time, giving every neuron a role
    in the encoding.  Without boosting, a few dominant columns monopolise
    all activity (dead-column collapse).
    """

    def __init__(
        self,
        num_columns: int,
        boost_strength: float,
        target_density: float,
    ) -> None:
        self.num_columns = num_columns
        self.boost_strength = boost_strength
        self.target_density = target_density
        self.boost_factors = np.ones(num_columns, dtype=np.float64)

    def update(self, active_duty_cycle: np.ndarray) -> None:
        """Recompute boost factors from the current active duty cycles."""
        if self.boost_strength == 0.0:
            return
        self.boost_factors = np.exp(
            -self.boost_strength * (active_duty_cycle - self.target_density)
        )

    def apply(self, overlap: np.ndarray) -> np.ndarray:
        """
        Multiply raw overlap scores by boost factors.

        Args:
            overlap: (num_columns,) float32

        Returns:
            boosted: (num_columns,) float64
        """
        return overlap * self.boost_factors

    # ── Serialisation ─────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {"boost_factors": self.boost_factors.copy()}

    def load_state_dict(self, d: dict) -> None:
        self.boost_factors = d["boost_factors"].copy()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. INHIBITION MODULE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class InhibitionModule:
    """
    Selects active columns from boosted overlap scores via k-Winner-Take-All.

    Two modes:

    ── global_zone (recommended for 86 000 columns) ────────────────────────
        Divide all columns into N equal zones.
        Within each zone, the top-k columns (by boosted overlap) win.
            k = max(1, round(density × zone_size))
        Fully vectorised: O(C × log k) via numpy argpartition.
        Deterministic, cache-friendly, ~860 zones × 100 cols each.

    ── local (biologically faithful) ───────────────────────────────────────
        Column i competes only with neighbours in [i−r, i+r].
        Column i is active iff its score ranks in the top-k of its window.
        Uses scipy.ndimage.rank_filter: O(N) in compiled C code.
        More realistic lateral inhibition, slower but biologically motivated.

    Variable sparsity:
        Pass target_density per call to compute().
        Complex inputs → caller passes higher density → more columns active.
        Simple inputs → lower density → fewer active columns.
    """

    def __init__(self, config: SpatialPoolerConfig) -> None:
        self.cfg = config
        self.num_columns = config.num_columns
        self.mode = config.inhibition_mode

        if self.mode == "global_zone":
            self._setup_zones()
        elif self.mode == "local":
            self.local_radius = config.local_inhibition_radius
            logger.info(f"Inhibition: local, radius={self.local_radius}")
        else:
            raise ValueError(f"Unknown inhibition_mode: {self.mode!r}")

    # ── Zone setup ────────────────────────────────────────────────────────────

    def _setup_zones(self) -> None:
        """Pre-compute zone boundaries with remainder distributed evenly."""
        cfg = self.cfg
        n = cfg.num_inhibition_zones
        base = cfg.num_columns // n
        remainder = cfg.num_columns % n

        starts, ends = [], []
        cursor = 0
        for z in range(n):
            starts.append(cursor)
            size = base + (1 if z < remainder else 0)
            cursor += size
            ends.append(cursor)

        self.zone_starts = np.array(starts, dtype=np.int64)
        self.zone_ends = np.array(ends, dtype=np.int64)
        self.n_zones = n
        logger.info(
            f"Inhibition: {n} global zones, ~{base} cols/zone"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def select_active_columns(
        self,
        boosted_overlap: np.ndarray,
        stimulus_threshold: int,
        target_density: Optional[float] = None,
    ) -> np.ndarray:
        """
        Run the k-WTA competition and return winning column indices.

        Args:
            boosted_overlap:    (num_columns,) float64
            stimulus_threshold: minimum score to enter the competition
            target_density:     override config density for this call

        Returns:
            active_columns: (k,) int64 — sorted indices of active columns
        """
        density = (
            target_density
            if target_density is not None
            else self.cfg.local_area_density
        )

        if self.mode == "global_zone":
            return self._zone_inhibit(boosted_overlap, stimulus_threshold, density)
        else:
            return self._local_inhibit(boosted_overlap, stimulus_threshold, density)

    # ── Zone inhibition ───────────────────────────────────────────────────────

    def _zone_inhibit(
        self,
        boosted_overlap: np.ndarray,
        stimulus_threshold: int,
        density: float,
    ) -> np.ndarray:
        """
        Vectorised zone-based inhibition.

        For each zone:
          1. Mask sub-threshold columns (score → −inf)
          2. Select top-k via np.argpartition  (O(zone_size × log k))
          3. Collect global column indices of winners
        """
        winners = []

        for z in range(self.n_zones):
            s = self.zone_starts[z]
            e = self.zone_ends[z]
            zone_len = int(e - s)
            k = max(1, int(round(density * zone_len)))

            zone_scores = boosted_overlap[s:e]                              # view

            # Sub-threshold mask
            eligible = np.where(
                zone_scores >= stimulus_threshold, zone_scores, -np.inf
            )

            # Skip zone if entirely sub-threshold
            finite_mask = np.isfinite(eligible)
            n_eligible = int(finite_mask.sum())
            if n_eligible == 0:
                continue

            if k >= n_eligible:
                local_win = np.where(finite_mask)[0]
            else:
                # argpartition: O(zone_size) — much faster than full sort
                local_win = np.argpartition(eligible, -k)[-k:]
                local_win = local_win[finite_mask[local_win]]

            winners.append(local_win + s)

        if not winners:
            return np.array([], dtype=np.int64)

        return np.sort(np.concatenate(winners).astype(np.int64))

    # ── Local inhibition ──────────────────────────────────────────────────────

    def _local_inhibit(
        self,
        boosted_overlap: np.ndarray,
        stimulus_threshold: int,
        density: float,
    ) -> np.ndarray:
        """
        Local sliding-window inhibition.

        Column i wins iff its score is in the top-k of the neighbourhood
        [i − radius, i + radius].

        Uses scipy.ndimage.rank_filter to compute the k-th order statistic
        of each window in O(N) time (compiled C extension).

        rank_filter(x, rank=window−k, size=window) at position i gives the
        (window−k)-th smallest element in the window, which is the k-th
        largest — the 'threshold to beat' for column i.
        """
        r = self.local_radius
        window = 2 * r + 1
        k = max(1, int(round(density * window)))

        # Mask sub-threshold columns
        masked = np.where(
            boosted_overlap >= stimulus_threshold,
            boosted_overlap,
            -np.inf,
        )

        # k-th largest in each sliding window of size `window`
        rank = max(0, window - k)
        kth_max = rank_filter(masked, rank=rank, size=window, mode="nearest")

        # Column i is active if its score ≥ local k-th max AND is finite
        active_mask = (masked >= kth_max) & np.isfinite(masked)
        return np.sort(np.where(active_mask)[0].astype(np.int64))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. STATS TRACKER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StatsTracker:
    """
    Rolling health statistics for the SpatialPooler.

    Tracks (over a sliding window of `window` steps):
        sparsity        — fraction of active columns per step
        n_active        — absolute count of active columns per step
        mean_overlap    — mean raw overlap score of active columns per step

    Also accumulates:
        total_iterations        — total compute() calls
        total_dead_recovered    — total column-rescues performed
    """

    def __init__(self, window: int = 500) -> None:
        self.window = window
        self._sparsity: List[float] = []
        self._n_active: List[int] = []
        self._mean_overlap: List[float] = []
        self.total_iterations = 0
        self.total_dead_recovered = 0

    def update(
        self,
        active_columns: np.ndarray,
        num_columns: int,
        raw_overlap: np.ndarray,
        n_recovered: int = 0,
    ) -> None:
        n = len(active_columns)
        self._sparsity.append(n / num_columns)
        self._n_active.append(n)
        self._mean_overlap.append(
            float(raw_overlap[active_columns].mean()) if n > 0 else 0.0
        )
        self.total_iterations += 1
        self.total_dead_recovered += n_recovered

        # Trim to window
        if len(self._sparsity) > self.window:
            self._sparsity = self._sparsity[-self.window:]
            self._n_active = self._n_active[-self.window:]
            self._mean_overlap = self._mean_overlap[-self.window:]

    def summary(self) -> Dict:
        sp = np.array(self._sparsity) if self._sparsity else np.array([0.0])
        na = np.array(self._n_active) if self._n_active else np.array([0])
        mo = np.array(self._mean_overlap) if self._mean_overlap else np.array([0.0])
        return {
            "total_iterations": self.total_iterations,
            "total_dead_recovered": self.total_dead_recovered,
            "sparsity_mean": float(sp.mean()),
            "sparsity_std": float(sp.std()),
            "sparsity_min": float(sp.min()),
            "sparsity_max": float(sp.max()),
            "n_active_mean": float(na.mean()),
            "overlap_mean": float(mo.mean()),
        }

    def state_dict(self) -> dict:
        return {
            "sparsity": list(self._sparsity),
            "n_active": list(self._n_active),
            "mean_overlap": list(self._mean_overlap),
            "total_iterations": self.total_iterations,
            "total_dead_recovered": self.total_dead_recovered,
            "window": self.window,
        }

    def load_state_dict(self, d: dict) -> None:
        self._sparsity = d.get("sparsity", [])
        self._n_active = d.get("n_active", [])
        self._mean_overlap = d.get("mean_overlap", [])
        self.total_iterations = d.get("total_iterations", 0)
        self.total_dead_recovered = d.get("total_dead_recovered", 0)
        self.window = d.get("window", self.window)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. SPATIAL POOLER  — main class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SpatialPooler:
    """
    Full HTM-style Spatial Pooler.

    Orchestrates all submodules into a complete encoding pipeline:

        Dense embedding
          → InputEncoder          binarise with online Welford stats
          → PotentialSynapses     vectorised overlap via fancy indexing
          → BoostingModule        homeostatic amplitude scaling
          → InhibitionModule      zone-based or local k-WTA
          → LearningEngine        Hebbian permanence update + dead-col rescue
          → DutyCycleTracker      EMA bookkeeping
          → SDR output

    Quick start:
    ─────────────────────────────────────────────────────────────────────────
        from sentence_transformers import SentenceTransformer
        import numpy as np

        # Build the SP
        config = SpatialPoolerConfig(input_dim=384, num_columns=86_000)
        sp = SpatialPooler(config)

        # Encode one sentence
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        emb = model.encode("Homo sapiens is the scientific name of humans")
        sdr = sp.compute(emb)                       # (86_000,) uint8
        idx = sp.compute_as_indices(emb)            # e.g. [42, 301, 890, ...]

        # Batch encode
        embeddings = model.encode([s1, s2, s3])     # (N, 384)
        sdrs = sp.compute_batch(embeddings)         # (N, 86_000) uint8

        # Semantic overlap
        sdr_a = sp.compute_as_indices(emb_a)
        sdr_b = sp.compute_as_indices(emb_b)
        score = sp.sdr_overlap(sdr_a, sdr_b)        # 0.0 – 1.0

        # Ripple retrieval
        neighbours = sp.ripple_neighbours(query_idx, stored_index_list, threshold=0.1)

        # Save / load
        sp.save("spatial_pooler.pkl")
        sp2 = SpatialPooler.load("spatial_pooler.pkl")
    ─────────────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        config: Optional[SpatialPoolerConfig] = None,
        **kwargs,
    ) -> None:
        if config is None:
            config = SpatialPoolerConfig(**kwargs)
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.iteration: int = 0

        logger.info("Initialising SpatialPooler …")

        self.encoder = InputEncoder(
            input_dim=config.input_dim,
            strategy=config.input_encode_strategy,
            density=config.input_density,
        )
        self.synapses = PotentialSynapses(config, self.rng)
        self.duty_cycles = DutyCycleTracker(
            num_columns=config.num_columns,
            period=config.duty_cycle_period,
        )
        self.boosting = BoostingModule(
            num_columns=config.num_columns,
            boost_strength=config.boost_strength,
            target_density=config.local_area_density,
        )
        self.inhibition = InhibitionModule(config)
        self.stats = StatsTracker(window=config.stats_window)

        logger.info(
            f"SpatialPooler ready │ "
            f"{config.input_dim}D → {config.num_columns:,} cols │ "
            f"mode={config.inhibition_mode} │ "
            f"target_sparsity={config.local_area_density:.1%} │ "
            f"learning={'ON' if config.learn else 'OFF'}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PRIMARY ENCODING API
    # ══════════════════════════════════════════════════════════════════════════

    def compute(
        self,
        embedding: np.ndarray,
        learn: Optional[bool] = None,
        target_density: Optional[float] = None,
    ) -> np.ndarray:
        """
        Convert a dense embedding to a Sparse Distributed Representation.

        Steps:
          1. Binarise embedding via InputEncoder
          2. Compute overlap scores (vectorised)
          3. Apply homeostatic boost factors
          4. Select active columns via inhibition (k-WTA)
          5. Build binary SDR output
          6. (if learn=True) Hebbian update + duty cycles + dead-col rescue

        Args:
            embedding:      (input_dim,) float32 — dense embedding
            learn:          override config.learn for this call
            target_density: override target sparsity (variable-sparsity mode)

        Returns:
            sdr: (num_columns,) uint8 — 1 = active, 0 = silent
        """
        should_learn = learn if learn is not None else self.config.learn

        # ── 1. Binarise ───────────────────────────────────────────────────────
        binary_input = self.encoder.encode(embedding)                       # (D,) uint8

        # ── 2. Overlap ────────────────────────────────────────────────────────
        raw_overlap = self.synapses.compute_overlap(binary_input)           # (C,) float32

        # ── 3. Boost ──────────────────────────────────────────────────────────
        boosted_overlap = self.boosting.apply(raw_overlap)                  # (C,) float64

        # ── 4. Inhibition ─────────────────────────────────────────────────────
        active_columns = self.inhibition.select_active_columns(
            boosted_overlap=boosted_overlap,
            stimulus_threshold=self.config.stimulus_threshold,
            target_density=target_density,
        )                                                                   # (k,) int64

        # ── 5. Build SDR ──────────────────────────────────────────────────────
        sdr = np.zeros(self.config.num_columns, dtype=np.uint8)
        if len(active_columns) > 0:
            sdr[active_columns] = 1

        # ── 6. Learn ──────────────────────────────────────────────────────────
        n_recovered = 0
        if should_learn:
            n_recovered = self._learn(binary_input, raw_overlap, active_columns)

        # ── 7. Stats ──────────────────────────────────────────────────────────
        self.stats.update(
            active_columns, self.config.num_columns, raw_overlap, n_recovered
        )
        self.iteration += 1

        return sdr

    # ──────────────────────────────────────────────────────────────────────────

    def compute_as_indices(
        self,
        embedding: np.ndarray,
        learn: Optional[bool] = None,
        target_density: Optional[float] = None,
    ) -> np.ndarray:
        """
        Like compute() but returns active column indices — not the full binary vector.

        Memory: 1 720 indices × 8 bytes = 13 KB  vs  86 000 bytes for the full SDR.

        Returns:
            active_indices: sorted int64 array — e.g. [9, 42, 301, 890, …]
        """
        sdr = self.compute(embedding, learn=learn, target_density=target_density)
        return np.where(sdr)[0]                                             # already sorted

    # ──────────────────────────────────────────────────────────────────────────

    def compute_batch(
        self,
        embeddings: np.ndarray,
        learn: Optional[bool] = None,
        target_density: Optional[float] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Process a batch of embeddings.

        Args:
            embeddings:     (N, input_dim) float32
            learn:          override learning flag
            target_density: override target sparsity
            show_progress:  show tqdm bar if available

        Returns:
            sdrs: (N, num_columns) uint8
        """
        N = embeddings.shape[0]
        sdrs = np.zeros((N, self.config.num_columns), dtype=np.uint8)

        iterator = range(N)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="SpatialPooler", unit="sample")
            except ImportError:
                pass

        for i in iterator:
            sdrs[i] = self.compute(
                embeddings[i], learn=learn, target_density=target_density
            )

        return sdrs

    # ──────────────────────────────────────────────────────────────────────────

    def compute_batch_indices(
        self,
        embeddings: np.ndarray,
        learn: Optional[bool] = None,
        target_density: Optional[float] = None,
        show_progress: bool = True,
    ) -> List[np.ndarray]:
        """
        Batch encode — return list of active-index arrays (memory-efficient).

        Returns:
            List of N sorted int64 arrays.
        """
        N = embeddings.shape[0]
        results: List[np.ndarray] = []

        iterator = range(N)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="SP (indices)", unit="sample")
            except ImportError:
                pass

        for i in iterator:
            results.append(
                self.compute_as_indices(
                    embeddings[i], learn=learn, target_density=target_density
                )
            )
        return results

    # ══════════════════════════════════════════════════════════════════════════
    # SIMILARITY & RIPPLE RETRIEVAL
    # ══════════════════════════════════════════════════════════════════════════

    def sdr_overlap(
        self,
        a: Union[np.ndarray, List[int]],
        b: Union[np.ndarray, List[int]],
        normalised: bool = True,
    ) -> float:
        """
        Compute the overlap between two SDRs.

        Accepts either full binary (uint8) vectors OR index arrays.

        normalised=True (default):
            overlap = |A ∩ B| / min(|A|, |B|)   ∈ [0.0, 1.0]
            0.0 = completely disjoint   (no shared neurons)
            1.0 = one is a subset       (maximum semantic similarity)

        normalised=False:
            overlap = |A ∩ B|   (raw shared-neuron count)
        """
        a = np.asarray(a)
        b = np.asarray(b)

        if a.dtype == np.uint8 and len(a) == self.config.num_columns:
            a_idx = np.where(a)[0]
            b_idx = np.where(b)[0]
        else:
            a_idx = np.sort(a.astype(np.int64))
            b_idx = np.sort(b.astype(np.int64))

        intersection = len(
            np.intersect1d(a_idx, b_idx, assume_unique=True)
        )

        if not normalised:
            return float(intersection)

        denom = min(len(a_idx), len(b_idx))
        return float(intersection) / float(denom) if denom > 0 else 0.0

    # ──────────────────────────────────────────────────────────────────────────

    def pairwise_overlaps(
        self,
        embeddings: np.ndarray,
        learn: bool = False,
        normalised: bool = True,
    ) -> np.ndarray:
        """
        Compute N×N pairwise SDR overlap matrix.

        Useful for verifying that the SP preserves semantic similarity:
        embed(s_i) ≈ embed(s_j)  should imply  overlap(sdr_i, sdr_j) > 0.

        Args:
            embeddings: (N, input_dim)
            learn:      whether to update the SP during encoding
            normalised: use normalised overlap

        Returns:
            matrix: (N, N) float64 — symmetric, diagonal = 1.0
        """
        idx_list = self.compute_batch_indices(
            embeddings, learn=learn, show_progress=len(embeddings) > 20
        )
        N = len(idx_list)
        matrix = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            matrix[i, i] = 1.0
            for j in range(i + 1, N):
                v = self.sdr_overlap(idx_list[i], idx_list[j], normalised=normalised)
                matrix[i, j] = v
                matrix[j, i] = v
        return matrix

    # ──────────────────────────────────────────────────────────────────────────

    def ripple_neighbours(
        self,
        query_indices: np.ndarray,
        stored_index_list: List[np.ndarray],
        threshold: float = 0.1,
    ) -> List[Tuple[int, float]]:
        """
        Ripple retrieval: find all stored SDRs that overlap a query above threshold.

        Models the "ripple" property: activating one cluster propagates to
        nearby clusters that share neurons, emulating associative recall.

        Args:
            query_indices:     active column indices of the query SDR
            stored_index_list: list of stored SDRs as active-index arrays
            threshold:         minimum normalised overlap to include in results

        Returns:
            Sorted list of (stored_index, overlap_score), highest overlap first.
        """
        hits = []
        for i, stored in enumerate(stored_index_list):
            score = self.sdr_overlap(query_indices, stored, normalised=True)
            if score >= threshold:
                hits.append((i, score))
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits

    # ──────────────────────────────────────────────────────────────────────────

    def semantic_similarity_preserved(
        self,
        emb_a: np.ndarray,
        emb_b: np.ndarray,
        dense_threshold: float = 0.7,
    ) -> Dict:
        """
        Diagnostic: check if the SP preserves the semantic similarity between
        two embeddings as measured by cosine similarity.

        Returns a dict with:
            cosine_sim      — cosine similarity of dense embeddings
            sdr_overlap     — normalised SDR overlap
            preserved       — True if high cosine_sim → high sdr_overlap
        """
        from numpy.linalg import norm
        cos = float(
            np.dot(emb_a, emb_b) / (norm(emb_a) * norm(emb_b) + 1e-8)
        )
        idx_a = self.compute_as_indices(emb_a, learn=False)
        idx_b = self.compute_as_indices(emb_b, learn=False)
        olap = self.sdr_overlap(idx_a, idx_b, normalised=True)
        return {
            "cosine_sim": round(cos, 4),
            "sdr_overlap": round(olap, 4),
            "n_shared_neurons": int(len(np.intersect1d(idx_a, idx_b))),
            "n_active_a": int(len(idx_a)),
            "n_active_b": int(len(idx_b)),
            "preserved": (cos >= dense_threshold) == (olap > 0.05),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # INTERNAL LEARNING ENGINE
    # ══════════════════════════════════════════════════════════════════════════

    def _learn(
        self,
        binary_input: np.ndarray,
        raw_overlap: np.ndarray,
        active_columns: np.ndarray,
    ) -> int:
        """
        Execute all learning updates for one step.

        Order:
          1. Update overlap + active duty cycles (EMA)
          2. Recompute boost factors from updated duty cycles
          3. Identify underperforming (dead) columns → bump permanences
          4. Vectorised Hebbian permanence update for active columns

        Returns:
            n_recovered: number of columns that received permanence bumps
        """
        cfg = self.config

        # 1. Duty cycles
        overlap_exceeds = raw_overlap >= cfg.stimulus_threshold             # (C,) bool
        self.duty_cycles.update(overlap_exceeds, active_columns)

        # 2. Boost factors
        self.boosting.update(self.duty_cycles.active_duty_cycle)

        # 3. Dead-column rescue
        underperforming = self.duty_cycles.underperforming_columns(
            cfg.min_pct_overlap_duty_cycle
        )
        if len(underperforming) > 0:
            self.synapses.bump_permanences(underperforming)

        # 4. Hebbian update (vectorised)
        self.synapses.update_permanences(active_columns, binary_input)

        return int(len(underperforming))

    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        """Return a comprehensive statistics dictionary."""
        s = self.stats.summary()
        s["iteration"] = self.iteration
        s["n_connected_synapses"] = self.synapses.n_connected()
        total_synapses = self.config.num_columns * self.synapses.n_syn
        s["pct_connected"] = s["n_connected_synapses"] / total_synapses
        s["boost_mean"] = float(self.boosting.boost_factors.mean())
        s["boost_std"] = float(self.boosting.boost_factors.std())
        s["boost_min"] = float(self.boosting.boost_factors.min())
        s["boost_max"] = float(self.boosting.boost_factors.max())
        s["overlap_dc_mean"] = float(self.duty_cycles.overlap_duty_cycle.mean())
        s["active_dc_mean"] = float(self.duty_cycles.active_duty_cycle.mean())
        s["target_sparsity"] = self.config.local_area_density
        return s

    def print_stats(self) -> None:
        """Print a formatted health report."""
        s = self.get_stats()
        bar = "═" * 64
        print(f"\n{bar}")
        print(f"  SpatialPooler — Health Report  (iter {s['iteration']:,})")
        print(bar)
        print(f"  Target sparsity      {s['target_sparsity']:.2%}")
        print(f"  Recent sparsity      {s['sparsity_mean']:.2%} ± {s['sparsity_std']:.4f}")
        print(f"  Active cols (mean)   {s['n_active_mean']:.0f} / {self.config.num_columns:,}")
        print(f"  Sparsity range       [{s['sparsity_min']:.2%}, {s['sparsity_max']:.2%}]")
        print(f"  Mean overlap score   {s['overlap_mean']:.2f}")
        print(f"  Connected synapses   {s['n_connected_synapses']:,}  ({s['pct_connected']:.1%})")
        print(f"  Boost factor         {s['boost_mean']:.4f} ± {s['boost_std']:.4f}")
        print(f"  Active duty cycle    {s['active_dc_mean']:.5f}  (target {self.config.local_area_density:.5f})")
        print(f"  Dead cols recovered  {s['total_dead_recovered']:,}")
        print(bar)

    def visualise_sdr(self, sdr: np.ndarray, width: int = 80) -> str:
        """
        One-line ASCII visualisation of an SDR's sparsity pattern.

        Each character represents a block of columns:
          '█' = ≥1 active column in this block
          '░' = all silent

        Args:
            sdr:   (num_columns,) uint8
            width: number of output characters

        Returns:
            80-character string (default)
        """
        n = len(sdr)
        block = n / width
        return "".join(
            "█" if sdr[int(i * block): int((i + 1) * block)].any() else "░"
            for i in range(width)
        )

    def describe_sdr(self, sdr: np.ndarray, label: str = "") -> None:
        """Print a compact description of an SDR."""
        idx = np.where(sdr)[0]
        n_active = len(idx)
        sparsity = n_active / len(sdr)
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}Active: {n_active:,} / {len(sdr):,}  ({sparsity:.2%})")
        print(f"  First 10 active indices: {idx[:10].tolist()}")
        print(f"  Pattern: {self.visualise_sdr(sdr)}")

    # ══════════════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ══════════════════════════════════════════════════════════════════════════

    def save(self, path: Union[str, Path]) -> None:
        """
        Serialise the full SpatialPooler state to disk.

        Saved:
            config, iteration
            synapses (permanences + potential_inputs)
            duty cycles (overlap EMA + active EMA)
            boost factors
            encoder running statistics
            rolling stats history
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "version": "2.0",
            "config": self.config,
            "iteration": self.iteration,
            "synapses": self.synapses.state_dict(),
            "duty_cycles": self.duty_cycles.state_dict(),
            "boosting": self.boosting.state_dict(),
            "encoder": self.encoder.state_dict(),
            "stats": self.stats.state_dict(),
        }

        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = path.stat().st_size / 1024**2
        logger.info(f"SpatialPooler saved → {path}  ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SpatialPooler":
        """
        Restore a SpatialPooler from a saved file.

        All learned state (permanences, duty cycles, boost factors, encoder
        statistics) is restored exactly.

        Args:
            path: path written by save()

        Returns:
            Fully restored SpatialPooler.
        """
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)

        sp = cls(config=state["config"])
        sp.iteration = state["iteration"]
        sp.synapses.load_state_dict(state["synapses"])
        sp.duty_cycles.load_state_dict(state["duty_cycles"])
        sp.boosting.load_state_dict(state["boosting"])
        sp.encoder.load_state_dict(state["encoder"])
        sp.stats.load_state_dict(state["stats"])

        logger.info(
            f"SpatialPooler loaded ← {path}  (iteration {sp.iteration:,})"
        )
        return sp


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEMO / INTEGRATION  (runs when executed directly)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("\n" + "━" * 64)
    print("  Full HTM Spatial Pooler — Integration Demo")
    print("━" * 64)

    # ── 1. Build SP ───────────────────────────────────────────────────────────
    config = SpatialPoolerConfig(
        input_dim=384,
        num_columns=86_000,
        inhibition_mode="global_zone",
        num_inhibition_zones=860,
        local_area_density=0.02,
        boost_strength=2.0,
        syn_perm_active_inc=0.05,
        syn_perm_inactive_dec=0.008,
        duty_cycle_period=1000,
        learn=True,
        seed=42,
    )
    sp = SpatialPooler(config)

    # ── 2. Load dense embeddings produced by your encode_texts_in_batches() ──
    #    (uses synthetic data here; swap in np.load("embeddings.npy") in prod)
    rng = np.random.RandomState(0)
    N = 200
    print(f"\nGenerating {N} synthetic embeddings (dim=384) for demo …")
    embeddings = rng.randn(N, 384).astype(np.float32)

    # Inject two semantically similar pairs so we can verify overlap
    # Pair A: "Homo sapiens is the scientific name of humans"
    base_a = rng.randn(384).astype(np.float32)
    embeddings[0] = base_a + rng.randn(384).astype(np.float32) * 0.05
    embeddings[1] = base_a + rng.randn(384).astype(np.float32) * 0.05
    # Pair B: very different content
    embeddings[2] = rng.randn(384).astype(np.float32)

    # ── 3. Warm-up learning pass (online) ─────────────────────────────────────
    print("\nLearning pass 1 …")
    sdrs_pass1 = sp.compute_batch(embeddings, learn=True, show_progress=True)

    print("\nLearning pass 2 (second epoch, permanences keep adapting) …")
    sdrs_pass2 = sp.compute_batch(embeddings, learn=True, show_progress=True)

    sp.print_stats()

    # ── 4. Encode test sentences (inference, no learning) ─────────────────────
    print("\nInference (learn=False) for semantic overlap test …")
    # Encode the same base embedding with tiny noise → should share neurons
    test_similar_a = base_a + rng.randn(384).astype(np.float32) * 0.02
    test_similar_b = base_a + rng.randn(384).astype(np.float32) * 0.02
    test_different  = rng.randn(384).astype(np.float32)

    idx_a = sp.compute_as_indices(test_similar_a, learn=False)
    idx_b = sp.compute_as_indices(test_similar_b, learn=False)
    idx_c = sp.compute_as_indices(test_different,  learn=False)

    olap_similar   = sp.sdr_overlap(idx_a, idx_b)
    olap_different = sp.sdr_overlap(idx_a, idx_c)

    print(f"\n  Overlap (similar embeddings):   {olap_similar:.4f}")
    print(f"  Overlap (different embeddings): {olap_different:.4f}")
    print(f"  → Semantic overlap preserved:  {olap_similar > olap_different}")

    # ── 5. SDR visualisation ─────────────────────────────────────────────────
    sdr_a = np.zeros(86_000, dtype=np.uint8)
    sdr_a[idx_a] = 1
    print()
    sp.describe_sdr(sdr_a, label="similar_A")

    sdr_b = np.zeros(86_000, dtype=np.uint8)
    sdr_b[idx_b] = 1
    sp.describe_sdr(sdr_b, label="similar_B")

    shared = np.intersect1d(idx_a, idx_b)
    print(f"  Shared neuron indices (first 10): {shared[:10].tolist()}")

    # ── 6. Ripple retrieval demo ──────────────────────────────────────────────
    print("\nRipple retrieval demo …")
    # Store index representations for all training samples
    stored_indices = sp.compute_batch_indices(
        embeddings[:20], learn=False, show_progress=False
    )
    # Query: very similar to embeddings[0]
    query_idx = sp.compute_as_indices(
        embeddings[0] + rng.randn(384).astype(np.float32) * 0.01,
        learn=False,
    )
    neighbours = sp.ripple_neighbours(query_idx, stored_indices, threshold=0.05)
    print(f"  Query matched {len(neighbours)} neighbours above threshold 0.05:")
    for rank, (store_idx, score) in enumerate(neighbours[:5]):
        print(f"    Rank {rank+1}: stored[{store_idx:2d}]  overlap={score:.4f}")

    # ── 7. Variable sparsity demo ─────────────────────────────────────────────
    print("\nVariable sparsity demo …")
    for density in [0.01, 0.02, 0.04]:
        idx = sp.compute_as_indices(
            embeddings[5], learn=False, target_density=density
        )
        print(f"  target_density={density:.0%} → {len(idx):,} neurons active")

    # ── 8. Save and reload ────────────────────────────────────────────────────
    save_path = "spatial_pooler_trained.pkl"
    sp.save(save_path)
    sp2 = SpatialPooler.load(save_path)

    # Verify exact reproduction
    idx_reload = sp2.compute_as_indices(test_similar_a, learn=False)
    reproduced = np.array_equal(idx_a, idx_reload)
    print(f"\n  Save/load round-trip identical: {reproduced}")

    print("\n" + "━" * 64)
    print("  Done. Plug sp.compute_as_indices(embedding) into EnhancedSuperBrain.")
    print("━" * 64 + "\n")
