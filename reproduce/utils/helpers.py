# reproduce/utils/helpers.py
"""
Helper functions for causalDA project.
Author: Boi Mai Quach <quachmaiboi.com>
License: GNU General Public License v3.0
"""

from collections import defaultdict, deque

def snake_case(s):
    """
    Convert string to snake_case.
    Example:
        "Target Node" -> "target_node"
    """
    return s.strip().lower().replace(" ", "_")

def snake_case_dict(obj):
    """
    Recursively convert all string keys/values in a dict to snake_case.
    """
    if isinstance(obj, dict):
        return {
            snake_case(k): snake_case_dict(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [snake_case_dict(i) for i in obj]
    elif isinstance(obj, str):
        return snake_case(obj)
    else:
        return obj  # int, float, etc.

def convert_effect_dict_to_links_dict(effect_dict, default_lag=0):
    """
    Convert flat effect_dict (child → {parent: value}) to links_dict format
    expected by plot_links_graph_svg.
    """
    return {
        child: {
            parent: {"value": value, "lag": default_lag}
            for parent, value in parents.items()
        }
        for child, parents in effect_dict.items()
    }

def build_causal_digraph(dag_result, extra_nodes=None, extra_edges=None):
    """
    Build a Graphviz digraph string from a DAG result.
    Parameters
    ----------
    dag_result : dict
        Dictionary representing the DAG in the format {effect: [causes]}.
    extra_nodes : list, optional
        Additional nodes to include in the graph.
    extra_edges : list of tuples, optional
        Additional edges to include in the graph, as (source, target) tuples.
    Returns
    -------
    str
        Graphviz digraph string.
    """
    edges = []
    nodes = set()

    for effect, causes in dag_result.items():
        nodes.add(effect)
        for cause in causes:
            nodes.add(cause)
            edges.append(f'    {cause} -> {effect}')

    if extra_nodes:
        for node in extra_nodes:
            node = snake_case(node)
            nodes.add(node)

    if extra_edges:
        for src, dst in extra_edges:
            src = snake_case(src)
            dst = snake_case(dst)
            nodes.add(src)
            nodes.add(dst)
            edges.append(f'    {src} -> {dst}')

    # Build Graphviz digraph
    node_lines = [f'    {node}' for node in sorted(nodes)]
    graph = 'digraph {\n' + '\n'.join(node_lines + edges) + '\n}'
    return graph

def str2bool(v):
    """
    Convert string to boolean.
    Accepts various string representations of true/false.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "yes", "y", "1"):
        return True
    elif v.lower() in ("false", "f", "no", "n", "0"):
        return False
    else:
        raise ValueError(f"Invalid boolean value: '{v}'. Use True/False.")


def compute_causal_layers(dag):
    """
    Compute the number of layers and nodes in each layer of a DAG.
    Layers are defined such that edges point from higher layers to lower layers,
    with sink nodes in layer 0.
    Parameters
    ----------
    dag : dict
        Dictionary representing the DAG in the format {child: [parents]}.
    Returns
    -------
    int
        Number of layers in the DAG.
    list of list
        List of layers, each containing the nodes in that layer.
    """
    # Build inverse DAG: parent → children
    children = defaultdict(list)
    in_degree = defaultdict(int)
    nodes = set()

    for child, parents in dag.items():
        nodes.add(child)
        for parent in parents:
            children[parent].append(child)
            in_degree[child] += 1
            nodes.add(parent)

    # Find root nodes (no incoming edges)
    root_nodes = [node for node in nodes if in_degree[node] == 0]

    # Standard Kahn's algorithm to get topological layers
    layers = []
    queue = deque(root_nodes)
    layer_map = {node: 0 for node in root_nodes}

    while queue:
        current_layer_nodes = list(queue)
        layers.append(current_layer_nodes)
        next_queue = deque()
        for node in current_layer_nodes:
            for child in children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    layer_map[child] = layer_map[node] + 1
                    next_queue.append(child)
        queue = next_queue

    # Reverse the layer order: sinks become S^0
    max_layer = max(layer_map.values())
    reversed_layer_map = {
        node: max_layer - layer for node, layer in layer_map.items()
    }

    # Group nodes by reversed layer
    reversed_layers = defaultdict(list)
    for node, layer in reversed_layer_map.items():
        reversed_layers[layer].append(node)

    # Sort layers by index
    final_layers = [reversed_layers[i] for i in sorted(reversed_layers)]

    return len(final_layers), final_layers