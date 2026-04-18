"""
visualize_super_brain_3d.py  (fixed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Interactive 3D visualisation of EnhancedSuperBrain neuron circuit.

Fixes vs original:
  1. Uniform sphere distribution (no more white central blob).
     Original used np.random.randn for radius → Gaussian → 68% of all nodes
     landed in a thin shell near r=0 and composited into solid white.
     Fix: r = R * cbrt(U[0,1])  → nodes fill the sphere volume evenly.

  2. Semantic radial layout.
     Highly-activated neurons pushed toward the outer shell (visible).
     Silent background neurons placed at the inner core.
     This makes the clustering and overlap structure immediately readable.

  3. Log-scale colour and size mapping.
     Original: size = 3 + count * 0.9, clipped to 14.
     Any neuron active in >12 SDRs already hit the cap → all looked the same.
     Fix: log1p transform so differences across 3 orders of magnitude are
     still visible in both colour and size.

  4. Fast edge build using numpy instead of a Python dict comprehension
     over potentially millions of synapses.

  5. Edge opacity reduced to 0.05 (was 0.12) so synapse lines don't
     contribute to the white-out even when many overlap.

  6. Active and inactive neuron layers drawn as separate traces so the
     legend and hover work cleanly.
"""

import pickle
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
BRAIN_FILE      = "super_brain_injected.pkl"
MAX_ACTIVE_SHOW = 20_000    # max active neurons to plot (all if fewer)
N_BACKGROUND    = 5_000     # silent neurons shown as faint backdrop
MAX_EDGES       = 8_000     # synapse edges (keep low for interactivity)
SPHERE_RADIUS   = 30        # outer radius of the neuron sphere
SEED            = 42


# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD BRAIN
# ──────────────────────────────────────────────────────────────────────────────
print("Loading brain ...")
with open(BRAIN_FILE, "rb") as f:
    brain = pickle.load(f)

print(f"  Neurons    : {brain.num_neurons:,}")
print(f"  Synapses   : {len(brain.synapses):,}")
print(f"  Stored SDRs: {len(brain.stored_sdrs):,}\n")


# ──────────────────────────────────────────────────────────────────────────────
# 2. COMPUTE PER-NEURON STATS
# ──────────────────────────────────────────────────────────────────────────────
print("Computing activation counts and degrees ...")

active_count = np.zeros(brain.num_neurons, dtype=np.int32)
degree       = np.zeros(brain.num_neurons, dtype=np.int32)

# Degree — use numpy for speed
if brain.synapses:
    syn_arr = np.array([(pre, post) for pre, post, _ in brain.synapses], dtype=np.int32)
    np.add.at(degree, syn_arr[:, 0], 1)
    np.add.at(degree, syn_arr[:, 1], 1)

# Activation count
for sdr_indices in tqdm(brain.stored_sdrs, desc="Activation count"):
    np.add.at(active_count, sdr_indices, 1)

print(f"  Active neurons (count > 0): {(active_count > 0).sum():,}")
print(f"  Max activation: {active_count.max():,}  |  Mean (active only): "
      f"{active_count[active_count>0].mean():.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. SELECT NODES TO DISPLAY
# ──────────────────────────────────────────────────────────────────────────────
np.random.seed(SEED)

active_idx   = np.where(active_count > 0)[0]
inactive_idx = np.where(active_count == 0)[0]

# Subsample active neurons if too many
if len(active_idx) > MAX_ACTIVE_SHOW:
    # Keep the most-activated ones (they carry the semantic information)
    top_k = np.argsort(active_count[active_idx])[-MAX_ACTIVE_SHOW:]
    active_idx = active_idx[top_k]

# Background neurons
n_bg = min(N_BACKGROUND, len(inactive_idx))
bg_idx = np.random.choice(inactive_idx, n_bg, replace=False)

print(f"\nPlotting: {len(active_idx):,} active + {n_bg:,} background neurons")


# ──────────────────────────────────────────────────────────────────────────────
# 4. POSITION ASSIGNMENT
# ──────────────────────────────────────────────────────────────────────────────
# Fix: uniform sphere sampling  →  r = R * cbrt(U)
# This ensures equal number of nodes per unit volume at any radius.
# No more Gaussian pile-up at the centre.
#
# Semantic layout:
#   Active neurons: radius in [0.5*R, R]  — outer shell, visible, colour-coded
#   Background:     radius in [0.0,   0.5*R] — inner core, faint grey

def uniform_sphere_points(n: int, r_min: float, r_max: float, rng) -> np.ndarray:
    """
    Sample n points uniformly in a spherical shell [r_min, r_max].
    Uses inverse CDF of the volume CDF:  r = (r_min^3 + U*(r_max^3-r_min^3))^(1/3)
    """
    u     = rng.uniform(0, 1, n)
    r     = (r_min**3 + u * (r_max**3 - r_min**3)) ** (1.0 / 3.0)
    theta = np.arccos(rng.uniform(-1, 1, n))   # uniform on sphere surface
    phi   = rng.uniform(0, 2 * np.pi, n)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=1)

rng = np.random.RandomState(SEED)
R = SPHERE_RADIUS

pos_active = uniform_sphere_points(len(active_idx), 0.5 * R, R, rng)
pos_bg     = uniform_sphere_points(n_bg,            0.0,     0.45 * R, rng)


# ──────────────────────────────────────────────────────────────────────────────
# 5. COLOUR AND SIZE  (log scale)
# ──────────────────────────────────────────────────────────────────────────────
# Log1p so a neuron active in 1 SDR and one active in 5000 are both visible.
counts_active = active_count[active_idx]
log_counts    = np.log1p(counts_active)

# Size: 3px (rare) to 10px (very frequent)
log_max   = max(log_counts.max(), 1.0)
norm      = log_counts / log_max                         # 0..1
node_size = 3 + norm * 7                                 # 3..10 px


# ──────────────────────────────────────────────────────────────────────────────
# 6. EDGE BUILD  (numpy-accelerated)
# ──────────────────────────────────────────────────────────────────────────────
print("Building synapse edges ...")

# Build a lookup: global neuron id → position in pos_active (or -1)
node_to_pos_active = np.full(brain.num_neurons, -1, dtype=np.int32)
node_to_pos_active[active_idx] = np.arange(len(active_idx), dtype=np.int32)

# Filter synapses to those where BOTH endpoints are in active set
if brain.synapses:
    src = syn_arr[:, 0]
    dst = syn_arr[:, 1]
    src_pos = node_to_pos_active[src]
    dst_pos = node_to_pos_active[dst]
    both_active = (src_pos >= 0) & (dst_pos >= 0)
    valid_src = src_pos[both_active]
    valid_dst = dst_pos[both_active]

    n_valid = len(valid_src)
    if n_valid > MAX_EDGES:
        sel = np.random.choice(n_valid, MAX_EDGES, replace=False)
        valid_src = valid_src[sel]
        valid_dst = valid_dst[sel]

    # Build Plotly edge arrays (interleave None to break line segments)
    p0 = pos_active[valid_src]    # (E, 3)
    p1 = pos_active[valid_dst]    # (E, 3)
    nans = np.full((len(p0), 3), np.nan)
    edge_pts = np.empty((len(p0) * 3, 3))
    edge_pts[0::3] = p0
    edge_pts[1::3] = p1
    edge_pts[2::3] = nans
    edge_x, edge_y, edge_z = edge_pts[:, 0], edge_pts[:, 1], edge_pts[:, 2]
    print(f"  Edges plotted: {len(valid_src):,} (from {n_valid:,} active↔active synapses)")
else:
    edge_x = edge_y = edge_z = []
    print("  No synapses found.")


# ──────────────────────────────────────────────────────────────────────────────
# 7. BUILD FIGURE
# ──────────────────────────────────────────────────────────────────────────────
fig = go.Figure()

# ── Synapse edges ─────────────────────────────────────────────────────────────
fig.add_trace(go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    mode="lines",
    line=dict(color="rgba(180,180,255,0.05)", width=0.5),
    hoverinfo="none",
    name="Synapses",
))

# ── Silent background neurons ──────────────────────────────────────────────────
fig.add_trace(go.Scatter3d(
    x=pos_bg[:, 0], y=pos_bg[:, 1], z=pos_bg[:, 2],
    mode="markers",
    marker=dict(
        size=1.5,
        color="rgba(80,80,120,0.25)",
        opacity=0.25,
    ),
    hoverinfo="skip",
    name="Silent neurons",
))

# ── Active neurons (colour = log activation count) ────────────────────────────
hover_text = [
    f"Neuron {active_idx[i]}<br>"
    f"Activations: {counts_active[i]:,}<br>"
    f"Degree: {degree[active_idx[i]]:,}"
    for i in range(len(active_idx))
]

fig.add_trace(go.Scatter3d(
    x=pos_active[:, 0], y=pos_active[:, 1], z=pos_active[:, 2],
    mode="markers",
    marker=dict(
        size=node_size,
        color=log_counts,
        colorscale="Plasma",
        cmin=0,
        cmax=float(log_max),
        colorbar=dict(
            title="log(1 + activation count)",
            tickvals=[0, log_max * 0.25, log_max * 0.5, log_max * 0.75, log_max],
            ticktext=[
                "0",
                f"{int(np.expm1(log_max*0.25))}",
                f"{int(np.expm1(log_max*0.50))}",
                f"{int(np.expm1(log_max*0.75))}",
                f"{int(np.expm1(log_max))}",
            ],
            thickness=15,
        ),
        opacity=0.90,
        line=dict(width=0),     # no white outline (was causing extra brightness)
    ),
    text=hover_text,
    hoverinfo="text",
    name="Active neurons",
))

# ── Layout ────────────────────────────────────────────────────────────────────
n_sdrs = len(brain.stored_sdrs)
fig.update_layout(
    title=dict(
        text=(
            "EnhancedSuperBrain 3D Neuron Circuit<br>"
            f"<sub>{brain.num_neurons:,} neurons · "
            f"{len(brain.synapses):,} synapses · "
            f"{n_sdrs:,} stored SDRs · "
            f"colour = semantic activation frequency (log scale)</sub>"
        ),
        x=0.5, xanchor="center",
    ),
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor="black",
        aspectmode="cube",
    ),
    paper_bgcolor="black",
    plot_bgcolor="black",
    margin=dict(l=0, r=0, b=0, t=80),
    height=900,
    legend=dict(
        x=0.01, y=0.99,
        font=dict(color="white"),
        bgcolor="rgba(0,0,0,0.4)",
    ),
)

fig.show()

print("\nInteractive 3D visualisation ready!")
print("  Rotate / zoom with mouse")
print("  Hover over nodes to see neuron ID, activation count, degree")
print("  Yellow/bright = neurons shared across many SDRs (semantic hubs)")
print("  Purple/dark   = neurons active in few SDRs (rare features)")
print("  Grey inner core = silent neurons (never activated by stored SDRs)")