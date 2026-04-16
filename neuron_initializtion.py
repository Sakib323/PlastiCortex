import numpy as np
import pickle
from datetime import datetime
from typing import List, Optional

class EnhancedSuperBrain:
    """
    A brain architecture designed to be stronger and more advanced than the human brain.
    - No LTD (no forgetting)
    - Strong initial weights
    - No synaptic pruning (only addition)
    - Scale‑free / small‑world connectivity
    """
    def __init__(self, num_neurons=86000, target_degree=10, 
                 weight_range=(0.5, 1.0), topology='scale_free'):
        self.num_neurons = num_neurons
        self.target_degree = target_degree
        self.weight_min, self.weight_max = weight_range
        self.topology = topology

        # Build connectivity (your original code unchanged)
        self.synapses = []  # list of (pre, post, weight)
        if topology == 'random':
            self._build_random()
        elif topology == 'scale_free':
            self._build_scale_free()
        elif topology == 'small_world':
            self._build_small_world()
        else:
            raise ValueError("Unknown topology")

        self.total_synapses = len(self.synapses)
        self.created = datetime.now().isoformat()

        # NEW: Storage for injected sparse SDRs
        self.stored_sdrs: List[np.ndarray] = []   # list of active column indices

        print(f"Enhanced SuperBrain created: {num_neurons} neurons, {self.total_synapses} synapses")
        print(f"Average degree: {self.total_synapses/num_neurons:.2f} (target {target_degree})")

    # ==================== YOUR ORIGINAL BUILD METHODS (unchanged) ====================
    def _build_random(self):
        p = self.target_degree / (self.num_neurons - 1)
        for pre in range(self.num_neurons):
            n_out = np.random.binomial(self.num_neurons - 1, p)
            if n_out == 0: continue
            posts = np.random.choice(self.num_neurons, size=n_out, replace=False)
            posts = posts[posts != pre]
            for post in posts:
                weight = np.random.uniform(self.weight_min, self.weight_max)
                self.synapses.append((pre, post, weight))

    def _build_scale_free(self):
        m0 = 5
        edges = set()
        nodes = list(range(m0))
        for i in range(m0):
            for j in range(i+1, m0):
                edges.add((i, j)); edges.add((j, i))
        for new_node in range(m0, self.num_neurons):
            degrees = {n: sum(1 for (u,v) in edges if u==n or v==n) for n in range(new_node)}
            total_deg = sum(degrees.values())
            if total_deg == 0: continue
            probs = [degrees[n]/total_deg for n in range(new_node)]
            targets = np.random.choice(range(new_node), size=m0, replace=False, p=probs)
            for t in targets:
                edges.add((new_node, t))
                edges.add((t, new_node))
        for (pre, post) in edges:
            weight = np.random.uniform(self.weight_min, self.weight_max)
            self.synapses.append((pre, post, weight))

    def _build_small_world(self):
        k = self.target_degree if self.target_degree % 2 == 0 else self.target_degree + 1
        beta = 0.1
        edges = set()
        for i in range(self.num_neurons):
            for j in range(1, k//2 + 1):
                neighbor = (i + j) % self.num_neurons
                edges.add((i, neighbor))
        edges_list = list(edges)
        for idx, (u, v) in enumerate(edges_list):
            if np.random.rand() < beta:
                new_v = np.random.choice(self.num_neurons)
                while new_v == u or (u, new_v) in edges:
                    new_v = np.random.choice(self.num_neurons)
                edges.remove((u, v))
                edges.add((u, new_v))
        for (pre, post) in edges:
            weight = np.random.uniform(self.weight_min, self.weight_max)
            self.synapses.append((pre, post, weight))

    # ====================== NEW: INJECT SPARSE SDRs ======================
    def inject_sparse_sdrs(self, sdr_list: List[np.ndarray], description: str = "IMDB train"):
        """
        Inject binary SDRs (or index arrays) into the brain.
        We store only the active indices (much smaller memory).
        """
        print(f"Injecting {len(sdr_list):,} sparse SDRs into EnhancedSuperBrain...")
        for sdr in sdr_list:
            # Convert full binary SDR → active indices (memory efficient)
            if sdr.ndim == 1 and sdr.dtype in (np.uint8, bool):
                indices = np.where(sdr)[0].astype(np.int32)
            else:
                indices = sdr.astype(np.int32)
            self.stored_sdrs.append(indices)

        self.metadata = getattr(self, 'metadata', {})
        self.metadata.update({
            "injected_dataset": description,
            "total_sdrs": len(self.stored_sdrs),
            "injected_at": datetime.now().isoformat()
        })
        print(f"✅ Injection complete. Brain now holds {len(self.stored_sdrs):,} sparse representations.")

    # ====================== SAVE / LOAD ======================
    def save(self, filepath: str = "super_brain.pkl"):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 EnhancedSuperBrain saved → {filepath} ({len(self.stored_sdrs)} SDRs stored)")

    @staticmethod
    def load(filepath: str):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def summary(self):
        print(f"\n=== EnhancedSuperBrain Summary ===")
        print(f"Neurons          : {self.num_neurons:,}")
        print(f"Synapses         : {self.total_synapses:,}")
        print(f"Topology         : {self.topology}")
        print(f"Stored SDRs      : {len(self.stored_sdrs):,}")
        if hasattr(self, 'metadata'):
            print(f"Last injection   : {self.metadata.get('injected_dataset', 'N/A')}")
        print(f"Created          : {self.created}")