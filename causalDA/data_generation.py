# causalDA/data_generation.py
"""casusalDA data generation functions."""

# Authors: Mai-Boi Quach <quachmaiboi@gmail.com>
# License: GNU General Public License v3.0

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, Tuple, Optional, List
from pymc_marketing.mmm.transformers import geometric_adstock, michaelis_menten

class DataGenerationError(Exception):
    """Custom exception for errors in data generation."""
    pass


class DataGenerator:
    """
    Class to generate synthetic time series data from a causal DAG structure.

    This class is designed for experiments in causal-driven attribution. It allows you to:
      - Generate a random directed acyclic graph (DAG) with a designated sink node (e.g., "conversion").
      - Simulate synthetic time series data for each node, propagating influence along DAG edges.

    Workflow
    --------
    1. Call `generate_random_dag()` to create a DAG adjacency matrix.
    2. Call `generate_data()` with explicit parameters to simulate time series.

    Attributes
    ----------
    node_lookup : dict[int, str]
        Index → name mapping for nodes.
    target_node : str
        Sink node name.
    seed : int or None
        Random seed (if provided).
    graph : np.ndarray or None
        Adjacency matrix (n_nodes x n_nodes). Created after `generate_random_dag()`.

    Returns
    -------
    generate_random_dag : np.ndarray
        Adjacency matrix with directed edges encoded as 1.0.
    generate_data : pd.DataFrame
        Synthetic dataset with one column per node and one row per time step.

    Raises
    ------
    DataGenerationError
        If:
          - `node_lookup` is empty or not a dict.
          - `target_node` is not found in `node_lookup`.
          - `edge_prob`, `time_periods`, or ranges are invalid.
          - `generate_data()` is called before `generate_random_dag()`.
          - `conversion_dict` references unknown channels.
    """

    def __init__(self, node_lookup: Dict[int, str], name_activity: str = 'impression', target_node: str = "conversion", seed: Optional[int] = None):
        self.node_lookup = node_lookup
        self.name_activity = name_activity
        self.target_node = target_node
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        if not isinstance(node_lookup, dict) or len(node_lookup) == 0:
            raise DataGenerationError(
                "`node_lookup` must be a non-empty dict[int, str].\n"
                "Example:\n"
                "    node_lookup = {0: 'Facebook', 1: 'Google Ads', 2: 'conversion'}"
            )
        if len(set(node_lookup.values())) != len(node_lookup):
            raise DataGenerationError("Values in `node_lookup` must be unique.")
        if target_node not in node_lookup.values():
            raise DataGenerationError(
                f"Target node '{target_node}' not found in node_lookup.\n"
                "Example:\n"
                "    node_lookup = {0: 'A', 1: 'B', 2: 'conversion'}\n"
                "    gen = DataGenerator(node_lookup, target_node='conversion')"
            )

        self.graph: Optional[np.ndarray] = None

    def generate_random_dag(self, edge_prob: Optional[float] = None) -> np.ndarray:
        np.random.seed(self.seed)
        if edge_prob is None:
            raise DataGenerationError(
                "`edge_prob` must be provided explicitly (float in [0,1]).\n"
                "Example:\n"
                "    gen.generate_random_dag(edge_prob=0.4)"
            )
        if not (0.0 <= edge_prob <= 1.0):
            raise DataGenerationError(f"`edge_prob` must be in [0,1], got {edge_prob}")

        n = len(self.node_lookup)
        graph = np.zeros((n, n), dtype=float)

        target_idx = [k for k, v in self.node_lookup.items() if v == self.target_node][0]
        ordering = np.random.permutation([i for i in range(n) if i != target_idx]).tolist() + [target_idx]

        for i, src in enumerate(ordering[:-1]):
            for tgt in ordering[i + 1 : -1]:
                if np.random.rand() < edge_prob:
                    graph[src, tgt] = 1.0

        for src in range(n):
            if src != target_idx:
                graph[src, target_idx] = 1.0

        self.graph = graph
        return graph

    def _topological_sort(self, adj: np.ndarray) -> List[int]:
        indegree = np.sum(adj, axis=0)
        queue = deque([i for i in range(len(indegree)) if indegree[i] == 0])
        order = []

        while queue:
            u = queue.popleft()
            order.append(u)
            for v in range(len(adj)):
                if adj[u, v] != 0:
                    indegree[v] -= 1
                    if indegree[v] == 0:
                        queue.append(v)

        if len(order) != len(adj):
            raise DataGenerationError("Graph contains a cycle; cannot sort.")
        return order

    def generate_data(
        self,
        influence_from_parents: Optional[Tuple[float, float]] = None,
        conversion_dict: Optional[Dict[str, float]] = None,
        time_periods: Optional[int] = None,
        base_range: Optional[Tuple[float, float]] = None,
        carryover: bool = False,
    ) -> pd.DataFrame:
        
        np.random.seed(self.seed)

        if self.graph is None:
            raise DataGenerationError(
                "Graph is not defined. Call `generate_random_dag()` before `generate_data()`.\n"
                "Example:\n"
                "    graph = gen.generate_random_dag(edge_prob=0.4)\n"
                "    df = gen.generate_data(...)"
            )

        if influence_from_parents is None:
            raise DataGenerationError(
                "`influence_from_parents` must be provided (tuple of floats).\n"
                "Example:\n"
                "    gen.generate_data(influence_from_parents=(0.15, 0.30), ...)"
            )
        if time_periods is None:
            raise DataGenerationError(
                "`time_periods` must be provided (positive int).\n"
                "Example:\n"
                "    gen.generate_data(time_periods=365, ...)"
            )
        if base_range is None:
            raise DataGenerationError(
                "`base_range` must be provided (tuple of floats).\n"
                "Example:\n"
                "    gen.generate_data(base_range=(1000, 5000), ...)"
            )
        if conversion_dict is None:
            raise DataGenerationError(
                "`conversion_dict` must be provided (dict[str,float]).\n"
                "Example:\n"
                "    gen.generate_data(conversion_dict={'Facebook': 0.02, 'Google Ads': 0.03}, ...)"
            )

        min_base, max_base = base_range
        if min_base < 0 or max_base < min_base:
            raise DataGenerationError(f"Invalid base_range {base_range}.")

        inv = {v: k for k, v in self.node_lookup.items()}
        target_idx = inv[self.target_node]

        topo = self._topological_sort(self.graph)

        df = pd.DataFrame()
        values = {}
        transformed_values = {}

        dict_contributions = {}
        effect_dict = {}

        for node in topo:
            name = self.node_lookup[node]

            if name == self.target_node:
                continue  # We'll handle conversion separately later

            parents = np.where(self.graph[:, node] == 1)[0]

            # Generate base + noise
            base = np.random.uniform(min_base, max_base, size=time_periods)
            noise_scale = 0.05 * min_base if len(parents) > 0 else 0.1 * min_base
            noise = np.random.normal(loc=0, scale=noise_scale, size=time_periods)
            values[node] = base + noise

            # Add influence from parents
            if len(parents) > 0:
                influence = np.zeros(time_periods)
                for p in parents:
                    coeff = np.random.uniform(*influence_from_parents)
                    influence += coeff * transformed_values[p]

                    # Track parent → child with effect strength
                    child_name = name
                    parent_name = self.node_lookup[p]
                    if child_name not in effect_dict:
                        effect_dict[child_name] = {}
                    effect_dict[child_name][parent_name] = coeff

                values[node] += influence

            # Transform + spike
            impressions = values[node].copy().flatten()
            spike_time = np.random.randint(low=10, high=time_periods - 10)
            spike_magnitude = np.random.uniform(0.25, 0.50) * np.mean(impressions)
            impressions[spike_time:spike_time + 2] += spike_magnitude

            # Carryover or not
            if carryover:
                adstocked = geometric_adstock(impressions, alpha=0.6, l_max=28).eval()
                scale = np.mean(adstocked) / 2.5
                lam_scaled = 20 * scale
                alpha_scaled = 25 * scale
                channel_effect = michaelis_menten(adstocked, alpha_scaled, lam_scaled)
            else:
                channel_effect = impressions.copy()

            transformed_values[node] = channel_effect
            dict_contributions[name] = channel_effect
            df[name + '_' + self.name_activity] = channel_effect

        conv_base = 0.1 * min_base + np.random.normal(
            loc=0, scale=0.005 * min_base, size=time_periods
        )
        conv_val = conv_base.copy()
        # conv_val = 0.0
        for ch, w in conversion_dict.items():
            ch_name = ch + '_' + self.name_activity
            if ch_name not in df.columns:
                raise DataGenerationError(
                    f"Unknown channel '{ch}' in conversion_dict.\n"
                    f"Available channels: {list(df.columns)}"
                )
            channel_conv = w * df[ch_name]
            df[ch + '_' + self.target_node] = channel_conv
            conv_val += channel_conv
        
        df[self.target_node] = conv_val
        df["baseline"] = conv_base

        effect_dict[self.target_node] = conversion_dict.copy()

        return df, dict_contributions, effect_dict
    
    def get_causal_graph(self) -> dict:
        """
        Extract causal graph in child → parent dictionary format from the internal adjacency matrix.

        Returns
        -------
        causal_graph : dict
            Example: {"conversion": {"facebook": {}, "youtube": {}}, "facebook": {"tiktok": {}}}
        """
        if self.graph is None:
            raise DataGenerationError("Graph is not defined. Call `generate_random_dag()` first.")

        causal_graph = {}
        num_nodes = self.graph.shape[0]

        for child_idx in range(num_nodes):
            child_name = self.node_lookup[child_idx]
            parent_idxs = np.where(self.graph[:, child_idx] == 1)[0]
            if len(parent_idxs) > 0:
                causal_graph[child_name] = {
                    self.node_lookup[parent_idx]: {} for parent_idx in parent_idxs
                }

        return causal_graph
