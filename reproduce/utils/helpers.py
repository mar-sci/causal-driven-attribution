def snake_case(s):
    return s.strip().lower().replace(" ", "_")

def snake_case_dict(obj):
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
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "yes", "y", "1"):
        return True
    elif v.lower() in ("false", "f", "no", "n", "0"):
        return False
    else:
        raise ValueError(f"Invalid boolean value: '{v}'. Use True/False.")
