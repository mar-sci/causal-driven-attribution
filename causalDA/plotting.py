"""causalDA plotting package."""

# Author: Boi Mai Quach <quachmaiboi.com>
#
# License: GNU General Public License v3.0

import matplotlib.pyplot as plt
import networkx as nx

from graphviz import Digraph
from IPython.display import SVG, display

# Define some colors (first color will be used for nodes)
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


COLORS = ['#1f77b4']  # fallback color list if not already defined

def plot_graph(input_graph, node_lookup, figsize=(8, 8), ax=None):
    """
    Visualize a directed graph from its adjacency matrix.

    Args
    ----
    input_graph : np.ndarray
        Adjacency matrix (n_nodes x n_nodes).
    node_lookup : dict[int, str]
        Mapping of node index → node label.
    figsize : tuple[int, int], optional
        Size of the plot (width, height in inches). Default is (8, 8).
    ax : matplotlib.axes.Axes, optional
        Matplotlib axis to draw on. If None, a new figure is created.

    Example
    -------
    >>> plot_graph(graph, node_lookup)
    """
    graph = nx.DiGraph(input_graph)

    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Compute layout
    pos = nx.circular_layout(graph)

    # Draw graph
    nx.draw(
        G=graph,
        pos=pos,
        ax=ax,
        node_color=COLORS[0],
        node_size=figsize[0] * 500,
        arrowsize=figsize[0] * 1.0,
        with_labels=True,
        labels=node_lookup,
        font_color="white",
        font_size=figsize[0] * 1.25,
    )

    ax.set_axis_off()  # optional: turn off axis lines

    if ax is None:
        plt.show()



def plot_links_graph_svg(
    links_dict,
    show_value: bool = False,
    show_legend: bool = True,
    rounding: int = 3,
    max_lag: int = None,
):
    """
    Visualize causal links as an SVG using Graphviz.

    Parameters
    ----------
    links_dict : dict
        Dictionary of causal links, typically from
        CausalModel.get_lagged_links() or prune_bidirectional_links().
        Format:
        {target: {source: {"value": effect_size, "lag": tau}}}
    show_value : bool, default=False
        If True, show edge effect size values as labels.
    rounding : int, default=3
        Decimal places for rounding effect sizes in labels.
    max_lag : int, optional
        If provided, only show edges with lag <= max_lag.

    Example
    -------
    >>> from plotting import plot_links_graph_svg
    >>> plot_links_graph_svg(dag_links, show_value=True, max_lag=7)
    """
    dot = Digraph(comment="Causal Graph")
    dot.attr("node", shape="ellipse")

    # Add nodes and edges
    for target, parents in links_dict.items():
        dot.node(target)
        for source, data in parents.items():
            lag = data["lag"]
            value = data["value"]

            if max_lag is not None and lag > max_lag:
                continue

            dot.node(source)
            label = f"{round(value, rounding)}" if show_value else ""
            color = "green" if value >= 0 else "red"
            dot.edge(source, target, label=label, color=color)

    # --- Legend as HTML-like node ---
    if show_legend:
        legend_html = """<
        <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="4">
            <TR>
                <TD><FONT COLOR="green">&#8594;</FONT></TD>
                <TD>Positive Weight</TD>
            </TR>
            <TR>
                <TD><FONT COLOR="red">&#8594;</FONT></TD>
                <TD>Negative Weight</TD>
            </TR>"""

        if max_lag is not None:
            legend_html += f"""
            <TR><TD COLSPAN="2" CELLPADDING="2">Max lag: {max_lag}</TD></TR>"""

        legend_html += "</TABLE>>"

        dot.node("legend", legend_html, shape="none")
    # -------------------------------
    # Render as SVG (inline in Jupyter)
    svg_str = dot.pipe(format="svg")
    display(SVG(svg_str))

