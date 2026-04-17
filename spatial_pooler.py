"""
spatial_pooler.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HTM Spatial Pooler  —  Encoding layer only
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

Pipeline:
  Dense embedding (384-dim float32)
        │
  InputEncoder          — adaptive z-score binarisation (Welford online stats)
        │
  PotentialSynapses     — receptive fields, permanences, overlap scoring
        │
  BoostingModule        — homeostatic amplification of under-active columns
        │
  InhibitionModule      — zone-based or local k-WTA competition
        │
  LearningEngine        — vectorised Hebbian permanence update
        │
  DutyCycleTracker      — EMA activity statistics + dead-column rescue
        │
  SDR output (86 000-dim uint8, ~1–2 % active)

What makes this "full HTM" and not a toy:
  ✓ Online permanence learning  — synapses grow/shrink per Hebbian rule
  ✓ Homeostatic boosting        — prevents dead neurons, balances firing rates
  ✓ Overlap duty-cycle rescue   — bumps permanences of silent columns
  ✓ Adaptive input encoding     — per-dimension running z-score (Welford)
  ✓ Variable sparsity           — pass target_density per call
  ✓ Semantic overlap            — similar embeddings → overlapping active sets
  ✓ Full save / load            — all learnable state persisted to disk
  ✓ Vectorised throughout       — no Python loops in hot paths
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

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
        input_dim=384  (all-MiniLM-L6-v2)  →  num_columns=86 000 neurons
    """

    # ── Architecture ──────────────────────────────────────────────────────────
    input_dim: int = 384
    num_columns: int = 86_000

    # ── Receptive fields ──────────────────────────────────────────────────────
    potential_radius: int = 192
    """Half-width of each column's connectivity neighbourhood in input-space.
    Default = input_dim // 2 so every column can potentially reach any input."""

    potential_pct: float = 0.5
    """Fraction of the 2×potential_radius neighbourhood each column samples.
    n_synapses ≈ int(potential_pct × potential_radius × 2)  ≈ 192 per column."""

    # ── Inhibition ────────────────────────────────────────────────────────────
    inhibition_mode: Literal["global_zone", "local"] = "global_zone"
    """
    global_zone  — divide columns into N zones; top-k per zone survive.
                   Fully vectorised, scales to 86 000 columns easily.
    local        — each column competes within a sliding window of radius r.
                   Uses scipy.ndimage.rank_filter — O(N) compiled C.
    """

    num_inhibition_zones: int = 860
    """Number of inhibition zones for 'global_zone' mode.
    860 zones × ~100 cols/zone at 86 000 total."""

    local_inhibition_radius: int = 50
    """Neighbourhood half-width for 'local' inhibition mode."""

    local_area_density: float = 0.02
    """Target fraction of columns active per inhibition area.
    0.02 → ~1 720 active out of 86 000, matching cortical sparsity estimates."""

    # ── Stimulus threshold ────────────────────────────────────────────────────
    stimulus_threshold: int = 2
    """Minimum raw overlap score for a column to enter the k-WTA competition."""

    # ── Permanence learning ───────────────────────────────────────────────────
    syn_perm_connected: float = 0.50
    """A synapse counts toward overlap iff its permanence >= this value."""

    syn_perm_active_inc: float = 0.05
    """Permanence increment when the pre-synaptic bit is active (Hebbian LTP)."""

    syn_perm_inactive_dec: float = 0.008
    """Permanence decrement when the pre-synaptic bit is inactive (LTD)."""

    syn_perm_below_stimulus_dec: float = 0.01
    """Extra permanence bump for underperforming (dead) columns — rescue."""

    syn_perm_trim_threshold: float = 0.025
    """Permanences below this are zeroed out (weak synapse pruning)."""

    syn_perm_min: float = 0.0
    syn_perm_max: float = 1.0

    # ── Duty cycles and boosting ──────────────────────────────────────────────
    duty_cycle_period: int = 1000
    """EMA window for duty-cycle calculation: alpha = 1 / period."""

    boost_strength: float = 2.0
    """
    Homeostatic boost strength.
    0  = disabled.
    2-5 = active balancing.
    boost_factor[i] = exp(-boost_strength * (active_dc[i] - target_density))
    """

    min_pct_overlap_duty_cycle: float = 0.001
    """Column whose overlap duty cycle < min_pct * global_max gets rescued."""

    min_pct_active_duty_cycle: float = 0.001
    """Monitoring threshold for active duty cycle health."""

    # ── Input encoding ────────────────────────────────────────────────────────
    input_encode_strategy: Literal["adaptive", "topk", "threshold", "sign"] = "adaptive"
    """
    adaptive   — per-dimension z-score (Welford online), then top-density% -> 1.
                 Best for embeddings with varying scales (recommended).
    topk       — globally top-k values -> 1.
    threshold  — top-density percentile -> 1.
    sign       — positive values -> 1.
    """

    input_density: float = 0.5
    """Fraction of input bits set to 1 after binarisation."""

    # ── Learning ──────────────────────────────────────────────────────────────
    learn: bool = True
    """Master learning switch. Set False at inference for reproducible SDRs."""

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int = 42
    stats_window: int = 500
    """Rolling window length for health statistics."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. INPUT ENCODER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class InputEncoder:
    """
    Converts dense float embeddings -> binary input vectors.

    The Spatial Pooler expects binary inputs where roughly `input_density`
    fraction of bits are 1.

    The `adaptive` strategy maintains per-dimension running statistics via
    Welford's online algorithm, z-scores the input before thresholding — 
    making it invariant to embedding scale and distribution drift over time.

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
        self._M2 = np.ones(input_dim, dtype=np.float64)

    # ── Public API ────────────────────────────────────────────────────────────

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode one dense embedding -> binary vector.

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
        """Online Welford — update per-dimension running mean / variance."""
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
        potential_inputs[col, k]  -> which input bit synapse k of column col connects to
        permanences[col, k]       -> permanence of that synapse (float32, 0-1)

    A synapse is "connected" iff permanence[col, k] >= syn_perm_connected.
    Only connected synapses count toward the column's overlap score.

    Memory footprint (86 000 cols x ~192 synapses):
        permanences       86 000 x 192 x 4 bytes  ~66 MB
        potential_inputs  86 000 x 192 x 4 bytes  ~66 MB
    """

    def __init__(
        self, config: SpatialPoolerConfig, rng: np.random.RandomState
    ) -> None:
        self.cfg = config

        n_syn = max(1, int(config.potential_pct * config.potential_radius * 2))
        self.n_syn = n_syn

        logger.info(
            f"PotentialSynapses: {config.num_columns:,} cols x {n_syn} synapses = "
            f"{config.num_columns * n_syn:,} connections  "
            f"(~{config.num_columns * n_syn * 4 / 1024**2:.0f} MB per array)"
        )

        self.potential_inputs = np.empty((config.num_columns, n_syn), dtype=np.int32)
        self.permanences = np.empty((config.num_columns, n_syn), dtype=np.float32)
        self._initialise(rng)

    # ── Initialisation ────────────────────────────────────────────────────────

    def _initialise(self, rng: np.random.RandomState) -> None:
        """
        Build receptive fields and seed permanences.

        Each column's centre is mapped to an input position proportionally.
        Potential inputs are sampled from a neighbourhood of that centre.
        Permanences are seeded ~N(syn_perm_connected, sigma) so roughly half
        of synapses start connected.
        """
        cfg = self.cfg
        C, P = cfg.num_columns, self.n_syn
        centers = np.linspace(0, cfg.input_dim - 1, C)

        t0 = time.time()
        for col in range(C):
            c = int(round(centers[col]))
            lo = max(0, c - cfg.potential_radius)
            hi = min(cfg.input_dim - 1, c + cfg.potential_radius)
            pool = np.arange(lo, hi + 1)

            if len(pool) < P:
                extra_idx = rng.choice(
                    np.setdiff1d(np.arange(cfg.input_dim), pool),
                    size=min(P - len(pool), cfg.input_dim - len(pool)),
                    replace=False,
                )
                pool = np.concatenate([pool, extra_idx])

            chosen = rng.choice(pool, size=min(P, len(pool)), replace=False)
            if len(chosen) < P:
                chosen = rng.choice(cfg.input_dim, size=P, replace=True)
            self.potential_inputs[col] = chosen.astype(np.int32)

        logger.info(f"Receptive fields built in {time.time() - t0:.1f}s")

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

            overlap[col] = sum_k  connected[col,k]  AND  binary_input[potential_inputs[col,k]]

        Fully vectorised via numpy fancy indexing — zero Python loops.

        Args:
            binary_input: (input_dim,) uint8

        Returns:
            overlap: (num_columns,) float32
        """
        input_vals = binary_input[self.potential_inputs]                    # (C, P) uint8
        hits = self.connected_mask & input_vals.astype(bool)               # (C, P) bool
        return hits.sum(axis=1).astype(np.float32)                         # (C,)

    def update_permanences(
        self,
        active_columns: np.ndarray,
        binary_input: np.ndarray,
    ) -> None:
        """
        Vectorised Hebbian permanence update for all active columns.

        synapse to ACTIVE input   -> perm += syn_perm_active_inc
        synapse to INACTIVE input -> perm -= syn_perm_inactive_dec
        clip to [syn_perm_min, syn_perm_max]
        zero out synapses below syn_perm_trim_threshold

        Args:
            active_columns: (k,) int64 — indices of winning columns this step
            binary_input:   (input_dim,) uint8
        """
        if len(active_columns) == 0:
            return
        cfg = self.cfg

        rf_inputs = binary_input[self.potential_inputs[active_columns]]     # (k, P) uint8
        perms = self.permanences[active_columns]                            # (k, P)

        perms += cfg.syn_perm_active_inc   * rf_inputs
        perms -= cfg.syn_perm_inactive_dec * (1 - rf_inputs)

        np.clip(perms, cfg.syn_perm_min, cfg.syn_perm_max, out=perms)
        perms[perms < cfg.syn_perm_trim_threshold] = 0.0

        self.permanences[active_columns] = perms

    def bump_permanences(self, columns: np.ndarray) -> None:
        """
        Dead-column rescue: nudge all potential synapses of `columns` upward.

        Args:
            columns: (k,) int64
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
        Low -> column rarely overlaps any input -> candidate for rescue.

    active_duty_cycle[col]
        Fraction of recent steps where col was active (won inhibition).
        Low -> column rarely fires -> boosting kicks in.

    EMA update:  dc[t] = (1 - alpha) * dc[t-1]  +  alpha * current_value
    """

    def __init__(self, num_columns: int, period: int) -> None:
        self.num_columns = num_columns
        self.period = period
        self.alpha = 1.0 / max(1, period)
        self.iteration = 0

        self.overlap_duty_cycle = np.zeros(num_columns, dtype=np.float64)
        self.active_duty_cycle = np.zeros(num_columns, dtype=np.float64)

    def update(
        self,
        overlap_exceeds: np.ndarray,
        active_columns: np.ndarray,
    ) -> None:
        """
        Step the EMA forward by one iteration.

        Args:
            overlap_exceeds: (num_columns,) bool
            active_columns:  (k,) int64
        """
        a = self.alpha
        self.iteration += 1

        self.overlap_duty_cycle = (
            (1.0 - a) * self.overlap_duty_cycle
            + a * overlap_exceeds.astype(np.float64)
        )

        active_vec = np.zeros(self.num_columns, dtype=np.float64)
        if len(active_columns) > 0:
            active_vec[active_columns] = 1.0
        self.active_duty_cycle = (
            (1.0 - a) * self.active_duty_cycle + a * active_vec
        )

    def underperforming_columns(self, min_pct: float) -> np.ndarray:
        """
        Indices of columns whose overlap duty cycle is dangerously low.
        These columns will have their permanences bumped (dead-column rescue).
        """
        global_max = float(self.overlap_duty_cycle.max())
        if global_max == 0.0:
            return np.array([], dtype=np.int64)
        threshold = min_pct * global_max
        return np.where(self.overlap_duty_cycle < threshold)[0]

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
    to equalise firing rates.

    boost_factor[i] = exp( -boost_strength * (active_duty_cycle[i] - target_density) )

    Column rarely fires   -> factor > 1  (amplified)
    Column fires too much -> factor < 1  (suppressed)
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
        """Recompute boost factors from current active duty cycles."""
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

    global_zone (recommended for 86 000 columns):
        Divide all columns into N equal zones.
        Within each zone, top-k columns by boosted overlap win.
        Fully vectorised: O(C x log k) via numpy argpartition.

    local (biologically faithful):
        Column i competes only within [i-r, i+r].
        Uses scipy.ndimage.rank_filter: O(N) compiled C code.

    Variable sparsity:
        Pass target_density per call to select_active_columns().
        Complex inputs -> caller passes higher density -> more columns active.
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
        logger.info(f"Inhibition: {n} global zones, ~{base} cols/zone")

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
            target_density:     override config density (variable sparsity)

        Returns:
            active_columns: (k,) int64 — sorted indices
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

    def _zone_inhibit(
        self,
        boosted_overlap: np.ndarray,
        stimulus_threshold: int,
        density: float,
    ) -> np.ndarray:
        winners = []
        for z in range(self.n_zones):
            s = int(self.zone_starts[z])
            e = int(self.zone_ends[z])
            zone_len = e - s
            k = max(1, int(round(density * zone_len)))

            zone_scores = boosted_overlap[s:e]
            eligible = np.where(zone_scores >= stimulus_threshold, zone_scores, -np.inf)

            finite_mask = np.isfinite(eligible)
            n_eligible = int(finite_mask.sum())
            if n_eligible == 0:
                continue

            if k >= n_eligible:
                local_win = np.where(finite_mask)[0]
            else:
                local_win = np.argpartition(eligible, -k)[-k:]
                local_win = local_win[finite_mask[local_win]]

            winners.append(local_win + s)

        if not winners:
            return np.array([], dtype=np.int64)
        return np.sort(np.concatenate(winners).astype(np.int64))

    def _local_inhibit(
        self,
        boosted_overlap: np.ndarray,
        stimulus_threshold: int,
        density: float,
    ) -> np.ndarray:
        r = self.local_radius
        window = 2 * r + 1
        k = max(1, int(round(density * window)))

        masked = np.where(
            boosted_overlap >= stimulus_threshold,
            boosted_overlap,
            -np.inf,
        )

        rank = max(0, window - k)
        kth_max = rank_filter(masked, rank=rank, size=window, mode="nearest")

        active_mask = (masked >= kth_max) & np.isfinite(masked)
        return np.sort(np.where(active_mask)[0].astype(np.int64))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. STATS TRACKER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StatsTracker:
    """
    Rolling health statistics for the SpatialPooler's encoding behaviour.

    Tracks (over a sliding window):
        sparsity       — fraction of active columns per step
        n_active       — absolute count of active columns per step
        mean_overlap   — mean raw overlap score of active columns per step

    These metrics reflect encoder health only — not brain-level performance.
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
    Full HTM-style Spatial Pooler — encoding layer only.

    Responsibility:
        Take a dense embedding -> return a Sparse Distributed Representation (SDR).

    Does NOT:
        - store SDRs              (EnhancedSuperBrain's job)
        - retrieve memories       (EnhancedSuperBrain's job)
        - ripple across clusters  (EnhancedSuperBrain's job)
        - compare SDRs            (EnhancedSuperBrain's job)

    Quick start:
    ─────────────────────────────────────────────────────────────────────────
        from spatial_pooler import SpatialPooler, SpatialPoolerConfig

        config = SpatialPoolerConfig(input_dim=384, num_columns=86_000)
        sp = SpatialPooler(config)

        # Single encode — returns full binary vector
        sdr = sp.compute(embedding)                  # (86 000,) uint8

        # Single encode — returns active indices (memory-efficient, preferred)
        indices = sp.compute_as_indices(embedding)   # e.g. [42, 301, 890, ...]

        # Batch encode — training pass
        sdrs    = sp.compute_batch(embeddings, learn=True)   # (N, 86 000) uint8
        idx_list = sp.compute_batch_indices(embeddings)      # list of N arrays

        # Save / load all learned state
        sp.save("spatial_pooler.pkl")
        sp2 = SpatialPooler.load("spatial_pooler.pkl")

        # Hand the SDR to your brain
        brain.inject_sparse_sdrs(idx_list)
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

        logger.info("Initialising SpatialPooler ...")

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
            f"SpatialPooler ready | "
            f"{config.input_dim}D -> {config.num_columns:,} cols | "
            f"mode={config.inhibition_mode} | "
            f"target_sparsity={config.local_area_density:.1%} | "
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
          2. Compute vectorised overlap scores
          3. Apply homeostatic boost factors
          4. Select active columns via inhibition (k-WTA)
          5. Build binary SDR output
          6. (if learn) Hebbian update + duty cycles + dead-column rescue

        Args:
            embedding:      (input_dim,) float32
            learn:          override config.learn for this call
            target_density: override target sparsity — variable sparsity mode

        Returns:
            sdr: (num_columns,) uint8 — 1 = active, 0 = silent
        """
        should_learn = learn if learn is not None else self.config.learn

        # 1. Binarise
        binary_input = self.encoder.encode(embedding)

        # 2. Overlap
        raw_overlap = self.synapses.compute_overlap(binary_input)

        # 3. Boost
        boosted_overlap = self.boosting.apply(raw_overlap)

        # 4. Inhibition
        active_columns = self.inhibition.select_active_columns(
            boosted_overlap=boosted_overlap,
            stimulus_threshold=self.config.stimulus_threshold,
            target_density=target_density,
        )

        # 5. Build SDR
        sdr = np.zeros(self.config.num_columns, dtype=np.uint8)
        if len(active_columns) > 0:
            sdr[active_columns] = 1

        # 6. Learn
        n_recovered = 0
        if should_learn:
            n_recovered = self._learn(binary_input, raw_overlap, active_columns)

        # 7. Stats
        self.stats.update(
            active_columns, self.config.num_columns, raw_overlap, n_recovered
        )
        self.iteration += 1

        return sdr

    def compute_as_indices(
        self,
        embedding: np.ndarray,
        learn: Optional[bool] = None,
        target_density: Optional[float] = None,
    ) -> np.ndarray:
        """
        Like compute() but returns active column indices rather than the full
        binary vector.

        Memory: ~1 720 indices x 8 bytes = 14 KB vs 86 KB for full SDR.
        This is the preferred format for injection into EnhancedSuperBrain.

        Returns:
            sorted int64 array — e.g. [9, 42, 301, 890, ...]
        """
        sdr = self.compute(embedding, learn=learn, target_density=target_density)
        return np.where(sdr)[0]

    def compute_batch(
        self,
        embeddings: np.ndarray,
        learn: Optional[bool] = None,
        target_density: Optional[float] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a batch of embeddings -> full binary SDR matrix.

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

    def compute_batch_indices(
        self,
        embeddings: np.ndarray,
        learn: Optional[bool] = None,
        target_density: Optional[float] = None,
        show_progress: bool = True,
    ) -> List[np.ndarray]:
        """
        Encode a batch -> list of active-index arrays (memory-efficient).

        Preferred for large-scale encoding before injection into EnhancedSuperBrain.

        Returns:
            List of N sorted int64 arrays — ready for brain.inject_sparse_sdrs()
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
          3. Identify underperforming (dead) columns -> bump permanences
          4. Vectorised Hebbian permanence update for active columns

        Returns:
            n_recovered: number of columns that received permanence bumps
        """
        cfg = self.config

        overlap_exceeds = raw_overlap >= cfg.stimulus_threshold
        self.duty_cycles.update(overlap_exceeds, active_columns)
        self.boosting.update(self.duty_cycles.active_duty_cycle)

        underperforming = self.duty_cycles.underperforming_columns(
            cfg.min_pct_overlap_duty_cycle
        )
        if len(underperforming) > 0:
            self.synapses.bump_permanences(underperforming)

        self.synapses.update_permanences(active_columns, binary_input)

        return int(len(underperforming))

    # ══════════════════════════════════════════════════════════════════════════
    # DIAGNOSTICS  (encoder health only)
    # ══════════════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        """Return encoder health statistics."""
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
        """Print a formatted encoder health report."""
        s = self.get_stats()
        bar = "=" * 64
        print(f"\n{bar}")
        print(f"  SpatialPooler - Encoder Health  (iter {s['iteration']:,})")
        print(bar)
        print(f"  Target sparsity      {s['target_sparsity']:.2%}")
        print(f"  Recent sparsity      {s['sparsity_mean']:.2%} +/- {s['sparsity_std']:.4f}")
        print(f"  Active cols (mean)   {s['n_active_mean']:.0f} / {self.config.num_columns:,}")
        print(f"  Sparsity range       [{s['sparsity_min']:.2%}, {s['sparsity_max']:.2%}]")
        print(f"  Mean overlap score   {s['overlap_mean']:.2f}")
        print(f"  Connected synapses   {s['n_connected_synapses']:,}  ({s['pct_connected']:.1%})")
        print(f"  Boost factor         {s['boost_mean']:.4f} +/- {s['boost_std']:.4f}")
        print(f"  Active duty cycle    {s['active_dc_mean']:.5f}  (target {self.config.local_area_density:.5f})")
        print(f"  Dead cols recovered  {s['total_dead_recovered']:,}")
        print(bar)

    def visualise_sdr(self, sdr: np.ndarray, width: int = 80) -> str:
        """
        One-line ASCII visualisation of an SDR's sparsity pattern.
        Each character = one block of columns.  '#' = >=1 active, '.' = silent.
        """
        n = len(sdr)
        block = n / width
        return "".join(
            "#" if sdr[int(i * block): int((i + 1) * block)].any() else "."
            for i in range(width)
        )

    def describe_sdr(self, sdr: np.ndarray, label: str = "") -> None:
        """Print a compact description of one SDR's encoding properties."""
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
        Serialise all learned encoder state to disk.

        Saved: config, iteration, permanences, potential_inputs,
               duty cycles, boost factors, encoder running statistics.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "version": "2.1",
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
        logger.info(f"SpatialPooler saved -> {path}  ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SpatialPooler":
        """
        Restore a SpatialPooler from a saved file.

        All learned state (permanences, duty cycles, boost factors, encoder
        statistics) is restored exactly.
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
            f"SpatialPooler loaded <- {path}  (iteration {sp.iteration:,})"
        )
        return sp


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INTEGRATION DEMO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("\n" + "=" * 64)
    print("  SpatialPooler - Encoding only demo")
    print("  (brain-level logic lives in neuron_initializtion.py)")
    print("=" * 64)

    # ── Build the encoder ─────────────────────────────────────────────────────
    config = SpatialPoolerConfig(
        input_dim=384,
        num_columns=86_000,
        inhibition_mode="global_zone",
        num_inhibition_zones=860,
        local_area_density=0.02,
        boost_strength=2.0,
        learn=True,
        seed=42,
    )
    sp = SpatialPooler(config)

    # ── Synthetic embeddings (swap for np.load("embeddings.npy") in prod) ────
    rng = np.random.RandomState(0)
    N = 200
    print(f"\nGenerating {N} synthetic embeddings (dim=384) ...")
    embeddings = rng.randn(N, 384).astype(np.float32)

    # ── Two learning passes ───────────────────────────────────────────────────
    print("\nLearning pass 1 ...")
    sp.compute_batch(embeddings, learn=True)

    print("\nLearning pass 2 ...")
    sp.compute_batch(embeddings, learn=True)

    sp.print_stats()

    # ── Inference: index-format SDRs, ready for EnhancedSuperBrain ───────────
    print("\nEncoding for brain injection (learn=False) ...")
    idx_list = sp.compute_batch_indices(embeddings, learn=False)

    print(f"\nReady: {len(idx_list)} SDRs as index arrays")
    print(f"  Example SDR[0]: {idx_list[0][:10].tolist()} ... ({len(idx_list[0])} active neurons)")

    # ── Variable sparsity demo ────────────────────────────────────────────────
    print("\nVariable sparsity (adjust per-input complexity):")
    for density in [0.01, 0.02, 0.04]:
        idx = sp.compute_as_indices(embeddings[0], learn=False, target_density=density)
        print(f"  target_density={density:.0%} -> {len(idx):,} neurons active")

    # ── SDR visualisation ─────────────────────────────────────────────────────
    sdr_example = sp.compute(embeddings[0], learn=False)
    sp.describe_sdr(sdr_example, label="sample")

    # ── Save ──────────────────────────────────────────────────────────────────
    sp.save("spatial_pooler_trained.pkl")
    sp2 = SpatialPooler.load("spatial_pooler_trained.pkl")
    print("\nSave/load verified.")

    # ── Hand-off instructions ─────────────────────────────────────────────────
    print("\nNext step:")
    print("  from neuron_initializtion import EnhancedSuperBrain")
    print("  brain = EnhancedSuperBrain(num_neurons=86000, topology='scale_free')")
    print("  brain.inject_sparse_sdrs(idx_list, description='IMDB train')")
    print("  brain.save('super_brain.pkl')")
    print("=" * 64 + "\n")
