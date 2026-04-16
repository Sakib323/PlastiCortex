#!/usr/bin/env python3
"""
Enhanced SuperBrain Network Visualization Tool
Loads the pickle file and provides multiple visualisation options.
"""

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# ============================================================
# 1. Data Loading (from your pickle file)
# ============================================================
def load_brain_pickle(filepath):
    """Load the EnhancedSuperBrain from a pickle file."""
    with open(filepath, 'rb') as f:
        brain = pickle.load(f)
    return brain

def build_networkx_graph(brain, max_nodes=None, max_edges=None):
    """Convert the brain's synapses into a NetworkX graph (sampled if needed)."""
    G = nx.DiGraph()
    nodes = brain.num_neurons if max_nodes is None else max_nodes
    G.add_nodes_from(range(nodes))
    
    edges_added = 0
    for pre, post, w in brain.synapses:
        if pre >= nodes or post >= nodes:
            continue
        G.add_edge(pre, post, weight=w)
        edges_added += 1
        if max_edges and edges_added >= max_edges:
            break
    return G

# ============================================================
# 2. Basic Statistics & Summaries
# ============================================================
def print_summary(brain):
    """Print key network statistics."""
    print("\n" + "="*50)
    print("ENHANCED SUPERBRAIN SUMMARY")
    print("="*50)
    print(f"  Neurons:          {brain.num_neurons:,}")
    print(f"  Synapses:         {brain.total_synapses:,}")
    print(f"  Avg degree:       {brain.total_synapses / brain.num_neurons:.2f}")
    print(f"  Topology:         {brain.topology}")
    print(f"  Weight range:     {brain.weight_min:.2f} – {brain.weight_max:.2f}")
    print(f"  Avg weight:       {np.mean([w for _,_,w in brain.synapses]):.3f}")
    print(f"  Created:          {brain.created}")
    print("="*50)

def plot_degree_distribution(brain, figsize=(8,5)):
    """Plot out-degree histogram with log scale."""
    out_degree = np.zeros(brain.num_neurons, dtype=int)
    for pre, _, _ in brain.synapses:
        out_degree[pre] += 1
    
    plt.figure(figsize=figsize)
    plt.hist(out_degree, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Out-degree')
    plt.ylabel('Number of neurons')
    plt.title('Degree Distribution (Log Scale)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_weight_distribution(brain, figsize=(8,5)):
    """Plot histogram of synaptic weights."""
    weights = [w for _, _, w in brain.synapses]
    plt.figure(figsize=figsize)
    plt.hist(weights, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Synaptic Weight')
    plt.ylabel('Frequency')
    plt.title('Weight Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================
# 3. Adjacency Matrix Heatmap (Sampled)
# ============================================================
def plot_adjacency_heatmap(brain, max_neurons=500, figsize=(10,8)):
    """Heatmap of the adjacency matrix for a subset of neurons."""
    n = min(brain.num_neurons, max_neurons)
    adj = np.zeros((n, n), dtype=float)
    for pre, post, w in brain.synapses:
        if pre < n and post < n:
            adj[pre, post] = w
    
    plt.figure(figsize=figsize)
    plt.imshow(adj, cmap='hot', interpolation='nearest', aspect='auto')
    plt.colorbar(label='Synaptic weight')
    plt.xlabel('Post-synaptic neuron')
    plt.ylabel('Pre-synaptic neuron')
    plt.title(f'Adjacency Matrix (first {n} neurons)')
    plt.tight_layout()
    plt.show()

# ============================================================
# 4. Graph Visualisations
# ============================================================
def plot_small_subgraph(brain, num_nodes=100, figsize=(12,10)):
    """Extract a subgraph and visualise with NetworkX."""
    n = min(brain.num_neurons, num_nodes)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    
    for pre, post, w in brain.synapses:
        if pre < n and post < n:
            G.add_edge(pre, post, weight=w)
    
    # Remove isolated nodes
    isolated = [node for node in G.nodes if G.degree(node) == 0]
    G.remove_nodes_from(isolated)
    
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42, k=0.3)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=[w*2 for w in weights], alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=6)
    
    plt.title(f'Subgraph of first {n} neurons (showing {len(G.nodes)} connected nodes)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def export_to_graphml(brain, filepath, max_nodes=None):
    """Export the network to GraphML format for external tools (Gephi, Cytoscape)."""
    G = build_networkx_graph(brain, max_nodes=max_nodes)
    nx.write_graphml(G, filepath)
    print(f"Exported to {filepath}")

# ============================================================
# 5. Main CLI Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Enhanced SuperBrain Network Visualizer')
    parser.add_argument('--file', type=str, default='super_brain.pkl', help='Path to pickle file')
    parser.add_argument('--max_nodes_heatmap', type=int, default=500, help='Max neurons for heatmap')
    parser.add_argument('--subgraph_nodes', type=int, default=100, help='Neurons for subgraph visualisation')
    parser.add_argument('--export_graphml', type=str, help='Export to GraphML file')
    parser.add_argument('--no_degree', action='store_true', help='Skip degree distribution')
    parser.add_argument('--no_weights', action='store_true', help='Skip weight distribution')
    parser.add_argument('--no_heatmap', action='store_true', help='Skip adjacency heatmap')
    parser.add_argument('--no_subgraph', action='store_true', help='Skip subgraph visualisation')
    
    args = parser.parse_args()
    
    # Load the brain
    brain = load_brain_pickle(args.file)
    
    # Print summary
    print_summary(brain)
    
    # Generate visualisations
    if not args.no_degree:
        plot_degree_distribution(brain)
    if not args.no_weights:
        plot_weight_distribution(brain)
    if not args.no_heatmap:
        plot_adjacency_heatmap(brain, max_neurons=args.max_nodes_heatmap)
    if not args.no_subgraph:
        plot_small_subgraph(brain, num_nodes=args.subgraph_nodes)
    
    # Export if requested
    if args.export_graphml:
        export_to_graphml(brain, args.export_graphml)

if __name__ == "__main__":
    main()