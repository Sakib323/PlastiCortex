import numpy as np
import pickle
from datetime import datetime
from typing import List

class EnhancedSuperBrain:
    """
    Enhanced SuperBrain - Fixed version with proper SDR injection
    """
    def __init__(self, num_neurons=86000, target_degree=10, 
                 weight_range=(0.5, 1.0), topology='scale_free'):
        self.num_neurons = num_neurons
        self.target_degree = target_degree
        self.weight_min, self.weight_max = weight_range
        self.topology = topology

        # Build connectivity
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

        # IMPORTANT: Initialize storage for SDRs here
        self.stored_sdrs: List[np.ndarray] = []
        self.metadata = {}

        print(f"Enhanced SuperBrain created: {num_neurons} neurons, {self.total_synapses} synapses")
        print(f"Average degree: {self.total_synapses/num_neurons:.2f} (target {target_degree})")

    # ==================== BUILD METHODS (unchanged) ====================
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
        for i in range(m0):
            for j in range(i+1, m0):
                edges.add((i, j))
                edges.add((j, i))
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

    # ====================== INJECT SDRs ======================
    def inject_sparse_sdrs(self, sdr_list, description: str = "IMDB train"):
        """
        Robust injection: accepts list of numpy arrays OR plain Python lists.
        Works with np.load(..., allow_pickle=True).tolist()
        """
        print(f"Injecting {len(sdr_list):,} sparse SDRs into EnhancedSuperBrain...")

        for sdr in sdr_list:
            # Force conversion to numpy array (handles both list and ndarray)
            if isinstance(sdr, list):
                sdr = np.array(sdr, dtype=np.int32)
            else:
                sdr = np.asarray(sdr, dtype=np.int32)

            # Now safe to check ndim
            if sdr.ndim == 1:
                if sdr.max() <= 1 and sdr.min() >= 0:          # binary vector
                    indices = np.where(sdr > 0)[0].astype(np.int32)
                else:
                    indices = sdr.astype(np.int32)              # already indices
            else:
                indices = sdr.astype(np.int32)

            self.stored_sdrs.append(indices)

        # Update metadata
        self.metadata.update({
            "injected_dataset": description,
            "total_sdrs": len(self.stored_sdrs),
            "injected_at": datetime.now().isoformat(),
            "avg_active_neurons": float(np.mean([len(idx) for idx in self.stored_sdrs]))
        })

        print(f"✅ Injection complete! Brain now holds {len(self.stored_sdrs):,} SDRs")
        print(f"   Average active neurons per SDR: ~{int(self.metadata['avg_active_neurons'])}")

    # ====================== SAVE / LOAD ======================
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 EnhancedSuperBrain saved to {filepath} ({len(self.stored_sdrs)} SDRs stored)")

    @staticmethod
    def load(filepath: str):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def summary(self):
        print(f"\n=== EnhancedSuperBrain Summary ===")
        print(f"Neurons          : {self.num_neurons:,}")
        print(f"Synapses         : {self.total_synapses:,}")
        print(f"Topology         : {self.topology}")
        print(f"Stored SDRs      : {len(getattr(self, 'stored_sdrs', [])):,}")
        if hasattr(self, 'metadata') and self.metadata:
            print(f"Dataset          : {self.metadata.get('injected_dataset', 'N/A')}")
            print(f"Avg active neurons: {self.metadata.get('avg_active_neurons', 0):.0f}")
        print(f"Created          : {self.created}")