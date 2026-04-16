import pickle
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ====================== LOAD THE BRAIN ======================
print("Loading super_brain_injected.pkl ...")
with open("super_brain_injected.pkl", "rb") as f:
    brain = pickle.load(f)

print(f"✅ Loaded EnhancedSuperBrain")
print(f"   Neurons : {brain.num_neurons:,}")
print(f"   Synapses: {len(brain.synapses):,}")
print(f"   Stored SDRs: {len(brain.stored_sdrs):,}\n")

# ====================== COMPUTE ACTIVATION (CLUSTERING) ======================
print("Calculating activation count and degree for every neuron...")
active_count = np.zeros(brain.num_neurons, dtype=np.int32)
degree = np.zeros(brain.num_neurons, dtype=np.int32)

# Degree (how connected each neuron is)
for pre, post, _ in tqdm(brain.synapses, desc="Degree"):
    degree[pre] += 1
    degree[post] += 1

# Activation count (how many SDRs use this neuron) → this shows clustering & overlap
for sdr_idx in tqdm(brain.stored_sdrs, desc="Activation"):
    active_count[sdr_idx] += 1

# ====================== SMART SUBSAMPLING (makes it fast & beautiful) ======================
active_mask = active_count > 0
active_nodes = np.where(active_mask)[0]
inactive_nodes = np.where(~active_mask)[0]

# Keep ALL active neurons + sample some background neurons
n_inactive = min(8000, len(inactive_nodes))
inactive_sample = np.random.choice(inactive_nodes, n_inactive, replace=False)

sampled_nodes = np.unique(np.concatenate([active_nodes, inactive_sample]))
node_map = {old: new for new, old in enumerate(sampled_nodes)}
n_nodes = len(sampled_nodes)

print(f"Visualizing {n_nodes:,} nodes ({len(active_nodes):,} active + {n_inactive:,} background)")

# ====================== 3D POSITIONS ======================
np.random.seed(42)
# Brain-like spherical distribution
r = np.random.randn(n_nodes) * 18
theta = np.random.uniform(0, np.pi, n_nodes)
phi = np.random.uniform(0, 2 * np.pi, n_nodes)
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)
positions = np.stack([x, y, z], axis=1)

# ====================== BUILD EDGES (network circuit) ======================
print("Building synapse edges...")
edges = []
max_edges = 12000   # keeps it interactive and beautiful

valid_edges = [(node_map[pre], node_map[post]) 
               for pre, post, _ in brain.synapses 
               if pre in node_map and post in node_map]

if len(valid_edges) > max_edges:
    valid_edges = [valid_edges[i] for i in np.random.choice(len(valid_edges), max_edges, replace=False)]

edge_x, edge_y, edge_z = [], [], []
for i, j in valid_edges:
    x0, y0, z0 = positions[i]
    x1, y1, z1 = positions[j]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_z.extend([z0, z1, None])

# ====================== 3D PLOT ======================
fig = go.Figure()

# 1. Synapses (the actual circuit)
fig.add_trace(
    go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(200,200,200,0.12)', width=0.6),
        hoverinfo='none',
        name='Synapses'
    )
)

# 2. Neurons
node_colors = active_count[sampled_nodes]
node_sizes = np.clip(3 + active_count[sampled_nodes] * 0.9, 3, 14)

fig.add_trace(
    go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale='Plasma',          # hot = highly shared neurons
            colorbar=dict(title="Activation Count<br>(# of SDRs using this neuron)"),
            opacity=0.92,
            line=dict(width=0.4, color='white')
        ),
        text=[f"Neuron {sampled_nodes[i]}<br>"
              f"Activations: {active_count[sampled_nodes[i]]}<br>"
              f"Degree: {degree[sampled_nodes[i]]}<br>"
              f"Shared by {active_count[sampled_nodes[i]]} reviews" 
              for i in range(n_nodes)],
        hoverinfo='text',
        name='Neurons'
    )
)

fig.update_layout(
    title="🧠 EnhancedSuperBrain 3D Neuron Circuit<br>"
          "<sub>Color = Semantic Overlap • Bright clusters = similar IMDB reviews sharing neurons</sub>",
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='cube'
    ),
    margin=dict(l=0, r=0, b=0, t=60),
    height=900,
    template="plotly_dark",
    showlegend=False
)

fig.show()

print("\n✅ Interactive 3D visualization ready!")
print("   • Rotate / zoom with mouse")
print("   • Hover over nodes to see details")
print("   • Bright/hot areas = strong semantic clustering & overlap")