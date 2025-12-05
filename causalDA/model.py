import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.pcmci import PCMCI

from dowhy import gcm
import networkx as nx


class CausalModel:
    """
    Causal discovery model wrapper using Tigramite's PCMCI.

    This class provides a structured workflow for running causal discovery
    on time series data using Tigramite's PCMCI algorithm. It converts a
    pandas DataFrame into a Tigramite-compatible format, runs PCMCI with
    configurable independence tests, extracts lagged causal links, and
    prunes bidirectional links to enforce DAG structure.

    Workflow
    --------
    1. Build Tigramite dataframe from pandas DataFrame.
    2. Run PCMCI with conditional independence tests.
    3. Extract lagged links (tau > 0).
    4. Prune bidirectional edges to enforce DAG.

    Attributes
    ----------
    df : pandas.DataFrame
        Original input dataframe.
    selected_columns : list of str
        Subset of dataframe columns used in analysis.
    dataframe : tigramite.data_processing.DataFrame
        Tigramite-compatible dataframe built from the input.
    var_names : list of str
        Variable names used in PCMCI analysis.
    results : dict or None
        Results from the last PCMCI run. Contains p_matrix, val_matrix, etc.
    verbose : int
        Verbosity level passed to Tigramite (0 = silent).

    Returns
    -------
    run_pcmci : dict
        PCMCI results dictionary with keys:
        - "p_matrix": array of p-values
        - "val_matrix": array of effect sizes/statistics
        - "conf_matrix": array of confidence intervals (if computed)
    get_lagged_links : dict
        Mapping of target → source → {"value": effect_size, "lag": tau}.
    prune_bidirectional_links : dict
        Same format as get_lagged_links, but with cycles removed.

    Raises
    ------
    ImportError
        If Tigramite is not installed in the environment.
    ValueError
        If selected columns are not found in the dataframe.
    RuntimeError
        If methods requiring PCMCI results are called before `run_pcmci()`.

    Example
    -------
    >>> model = CausalModel(df, selected_columns=df.columns.tolist())
    >>> model.run_pcmci(tau_max=14, pc_alpha=0.2, alpha_level=0.01)
    >>> links = model.get_lagged_links(alpha_level=0.05, drop_source="conversion")
    >>> dag_links = model.prune_bidirectional_links(links)
    """

    def __init__(self, df: pd.DataFrame, selected_columns: List[str], seed: Optional[int] = None, verbose: int = 0):
        
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        self.df = df
        self.selected_columns = selected_columns
        self.dataframe = self.build_tigramite_dataframe(df, selected_columns)
        self.var_names = self.dataframe.var_names
        self.results: Optional[Dict[str, Any]] = None
        self.verbose = verbose

    @staticmethod
    def build_tigramite_dataframe(df: pd.DataFrame, selected_columns: List[str]) -> Any:
        """
        Build Tigramite dataframe for causal analysis.

        Args
        ----
        df : pd.DataFrame
            Input dataframe.
        selected_columns : list of str
            Columns to include in analysis.

        Returns
        -------
        pp.DataFrame
            Tigramite DataFrame.

        Raises
        ------
        ValueError
            If selected columns are not in the dataframe.
        """
        missing_cols = [col for col in selected_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in dataframe: {missing_cols}")

        data_array = df[selected_columns].values
        datatime = {0: np.arange(data_array.shape[0])}
        return pp.DataFrame(data_array, datatime=datatime, var_names=selected_columns)

    def run_pcmci(self, tau_max: int, pc_alpha: float, alpha_level: float = 0.05, use_robust: bool = True):
        """
        Run PCMCI on the dataframe.

        Args
        ----
        tau_max : int
            Maximum time lag.
        pc_alpha : float
            Significance level for PC algorithm.
        alpha_level : float, default=0.05
            Significance level for conditional independence tests.
        use_robust : bool, default=True
            Whether to use RobustParCorr instead of ParCorr.
        """

        np.random.seed(self.seed)
        cond_ind_test = RobustParCorr(significance="analytic") if use_robust else ParCorr(significance="analytic")

        pcmci = PCMCI(
            dataframe=self.dataframe,
            cond_ind_test=cond_ind_test,
            verbosity=self.verbose,
        )

        self.results = pcmci.run_pcmci(
            tau_max=tau_max, 
            pc_alpha=pc_alpha, 
            alpha_level=alpha_level)
        
        return self.results

    def get_lagged_links(
        self,
        alpha_level: float = 0.05,
        drop_source: Optional[str] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Extract only lagged significant links (tau > 0).

        Parameters
        ----------
        alpha_level : float, default=0.05
            Significance threshold.
        drop_source : str, optional
            If provided, remove all links coming FROM this source variable.

        Returns
        -------
        dict
            {target_var: {source_var: {"value": effect_size, "lag": tau}}}
        """
        if self.results is None:
            raise RuntimeError("PCMCI must be run before extracting lagged links.")

        p_matrix = self.results["p_matrix"]
        val_matrix = self.results["val_matrix"]

        N = len(self.var_names)
        result = {}

        for j in range(N):  # target
            target = self.var_names[j]
            result[target] = {}

            for i in range(N):  # source
                for tau in range(1, p_matrix.shape[2]):  # tau > 0
                    if p_matrix[i, j, tau] <= alpha_level:
                        source = self.var_names[i]
                        if drop_source is not None and source == drop_source:
                            continue
                        if source == target:  # prevent self loops
                            continue

                        result[target][source] = {
                            "value": float(val_matrix[i, j, tau]),
                            "lag": tau,
                        }

        return result

    @staticmethod
    def prune_bidirectional_links(
        links_dict: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Remove cycles by pruning bidirectional links, keeping one edge only.

        Rule:
        1. Keep the edge with smaller lag.
        2. If lags are equal, keep the edge with larger |value|.

        Parameters
        ----------
        links_dict : dict
            {target: {source: {"value": effect_size, "lag": tau}}}

        Returns
        -------
        dict
            Pruned dictionary with no reciprocal edges.
        """
        pruned = {k: dict(v) for k, v in links_dict.items()}

        for target, parents in list(links_dict.items()):
            for source, data in list(parents.items()):
                if source in links_dict and target in links_dict[source]:
                    val1, lag1 = data["value"], data["lag"]
                    val2, lag2 = links_dict[source][target]["value"], links_dict[source][target]["lag"]

                    # Decide which edge to drop
                    if lag1 == lag2:
                        drop = (source, target) if abs(val1) < abs(val2) else (target, source)
                    else:
                        drop = (source, target) if lag1 > lag2 else (target, source)

                    if drop[1] in pruned and drop[0] in pruned[drop[1]]:
                        del pruned[drop[1]][drop[0]]

        return pruned
    
    @staticmethod
    def prune_cycles(
        links_dict: Dict[str, Dict[str, Dict[str, float]]],
        priority_flag: bool = False,
        priority_node: str = "conversion",
        verbose: bool = False,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Remove all cycles in the graph, prioritizing keeping incoming edges to priority_node.

        Parameters
        ----------
        links_dict : dict
            {target: {source: {"value": effect_size, "lag": tau}}}
        priority_node : str, default='conversion'
            Node whose incoming edges should be preserved as much as possible.
        verbose : bool, default=True
            Print removed edges.

        Returns
        -------
        dict
            Acyclic DAG links dict.
        """
        # Convert to DiGraph
        G = nx.DiGraph()
        for target, parents in links_dict.items():
            for source, data in parents.items():
                G.add_edge(source, target, **data)

        # Add this to ensure isolated nodes are present
        for node in links_dict:
            G.add_node(node)

        while not nx.is_directed_acyclic_graph(G):
            try:
                cycle = next(nx.simple_cycles(G))
            except StopIteration:
                break

            # List edges in the cycle
            cycle_edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

            # Prefer to drop edges NOT going into priority_node
            non_priority_edges = [(u, v) for u, v in cycle_edges if v != priority_node]

            if non_priority_edges:
                u, v = min(
                    non_priority_edges,
                    key=lambda edge: abs(G[edge[0]][edge[1]]["value"])
                )
            else:
                # All edges go to priority_node; reluctantly remove weakest one
                u, v = min(
                    cycle_edges,
                    key=lambda edge: abs(G[edge[0]][edge[1]]["value"])
                )

            if verbose:
                print(f"Removing edge: {u} -> {v} to break cycle: {cycle}")
            G.remove_edge(u, v)

        # Convert back to links_dict format
        pruned = {}
        for u, v, data in G.edges(data=True):
            pruned.setdefault(v, {})[u] = {"value": data["value"], "lag": data["lag"]}

        # Ensure every node eventually influences priority_node (added at the end)
        all_nodes = G.nodes()

        def has_path_to_priority(node, visited=None):
            """DFS to check if node eventually leads to priority_node."""
            if node == priority_node:
                return True
            if visited is None:
                visited = set()
            if node in visited or node not in pruned:
                return False
            visited.add(node)
            return any(has_path_to_priority(parent, visited) for parent in pruned[node])

        if priority_flag:
            for node in all_nodes:
                if node == priority_node:
                    continue

                # Skip if there's already a direct edge to priority_node
                if priority_node in pruned and node in pruned[priority_node]:
                    continue

                # Skip if node already leads to priority_node through other nodes
                if has_path_to_priority(node):
                    continue

                # Otherwise, add a dummy edge node -> priority_node
                pruned.setdefault(priority_node, {})[node] = {"value": 0.0, "lag": 0}
                if verbose:
                    print(f"Added dummy edge at the end: {node} -> {priority_node} (value=0, lag=0)")



        return pruned
    
    @staticmethod
    def estimate_conversion_drop_via_sampling_repeated(
        scm,
        data: pd.DataFrame,
        channel: str,
        target: str = "conversion",
        n_runs: int = 5000
    ) -> float:
        """
        Estimate conversion drop by performing repeated interventional sampling.
        Parameters
        ----------
        scm : dowhy.gcm.CausalModel
            The structural causal model.
        data : pd.DataFrame
            Observed data for conditioning.
        channel : str
            The marketing channel to intervene on.
        target : str, default='conversion'
            The outcome variable.
        n_runs : int, default=5000
            Number of repeated sampling runs.
        Returns
        -------
        float
            Estimated average conversion drop.
        """
        drops = []
        for _ in range(n_runs):
            baseline = gcm.interventional_samples(scm, {}, observed_data=data)
            after_do = gcm.interventional_samples(scm, {channel: lambda x: 0.0}, observed_data=data)
            drop = baseline[target].mean() - after_do[target].mean()
            drops.append(drop)
        return float(np.mean(drops))

