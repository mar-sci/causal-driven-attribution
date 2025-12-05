from collections import defaultdict
import os
import numpy as np
import json
import pandas as pd
from dowhy import gcm
from typing import Dict, Any
from sklearn.metrics import roc_auc_score, confusion_matrix, fbeta_score

from reproduce.utils.config import ACTIVITY_NAME, TARGET_NODE
from reproduce.utils.helpers import snake_case

from gadjid import shd, sid


def dict_to_binary_matrix(pred_dict: Dict[str, Dict[str, Any]], node_lookup: Dict[int, str]) -> np.ndarray:
    """
    Convert predicted dictionary to binary adjacency matrix (0/1).

    Parameters
    ----------
    pred_dict : dict
        Dictionary of predicted edges, typically from causal discovery.
        Format: {target: {source: {...}}}
    node_lookup : dict[int, str]
        Mapping of node index → node name.

    Returns
    -------
    np.ndarray
        Binary adjacency matrix of shape (N, N).
    """
    N = len(node_lookup)
    binary_matrix = np.zeros((N, N), dtype=int)
    
    name_to_idx = {
        (snake_case(v) if v == TARGET_NODE else f"{snake_case(v)}_{ACTIVITY_NAME}"): k
        for k, v in node_lookup.items()
        }
    
    # name_to_idx = {v: k for k, v in node_lookup.items()}

    for target, parents in pred_dict.items():
        tgt_idx = name_to_idx[target]
        for source in parents.keys():
            src_idx = name_to_idx[source]
            binary_matrix[src_idx, tgt_idx] = 1

    return binary_matrix


def evaluate_binary(truth_matrix: np.ndarray, pred_matrix: np.ndarray, beta: float = 0.5) -> Dict[str, float]:
    """
    Evaluate predicted adjacency matrix against ground truth.

    Parameters
    ----------
    truth_matrix : np.ndarray
        Ground truth adjacency matrix (N x N).
    pred_matrix : np.ndarray
        Predicted adjacency matrix (N x N).
    beta : float, default=0.5
        Beta parameter for F-beta score.

    Returns
    -------
    dict
        Dictionary with evaluation metrics:
        {"sid", "shd", "AUC", "FPR", "TPR", "F{beta}"}.
    """
    N = truth_matrix.shape[0]
    select_off_diagonal = (np.identity(N) == 0)

    G_true = np.asarray(truth_matrix, dtype=np.int8).copy(order="C")
    G_pred = np.asarray(pred_matrix, dtype=np.int8).copy(order="C")

    y_true = truth_matrix.astype(int)[select_off_diagonal]
    y_pred = pred_matrix.astype(int)[select_off_diagonal]

    # Distance-based metrics
    sid_score,_ = sid(G_true, G_pred, edge_direction="from row to column")
    shd_score,_ = shd(G_true, G_pred)

    # Classification-based metrics
    auc = roc_auc_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / float(fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / float(tp + fn) if (tp + fn) > 0 else 0
    fscore = fbeta_score(y_true, y_pred, beta=beta)

    return {
        "sid": float(sid_score),
        "shd": float(shd_score),
        "AUC": float(auc),
        "FPR": float(fpr),
        "TPR": float(tpr),
        f"F{beta}": float(fscore),
    }


def save_evaluation_results(
    filepath: str,
    seed: int,
    tau: int,
    predicted_dag: Dict[str, Dict[str, Dict[str, Any]]],
    metrics: Dict[str, float],
) -> None:
    """
    Save evaluation results to a JSON file.

    Parameters
    ----------
    filepath : str
        Path to save results.
    seed : int
        Random seed used in the experiment.
    tau : int
        Maximum time lag used in PCMCI.
    predicted_dag : dict
        Predicted DAG structure. Format:
        {target: {source: {"value": float, "lag": int}}}
    metrics : dict
        Evaluation metrics (AUC, FPR, TPR, F-beta).
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert predicted_dag to JSON-safe format
    safe_dag = {}
    for target, parents in predicted_dag.items():
        safe_dag[target] = {
            source: {"value": float(data["value"]), "lag": int(data["lag"])}
            for source, data in parents.items()
        }

    result = {
        "seed": seed,
        "tau": tau,
        "predicted_dag": safe_dag,
    }
    result.update(metrics)

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)


def compute_causal_effects(links_dict, outcome='conversion'):
    """
    Computes direct and total causal effects from a DAG links_dict to a specified outcome variable.

    Args:
        links_dict (dict): DAG in the format {target: {source: {'value': weight, 'lag': int}}}
        outcome (str): Name of the outcome node (default: 'conversion')

    Returns:
        dict: direct_effects
        dict: total_effects
    """

    # Direct effects: only edges pointing to the outcome node
    direct_effect = {
        src: data['value']
        for src, data in links_dict.get(outcome, {}).items()
    }

    # Build adjacency list: source -> list of (target, weight)
    edges = defaultdict(list)
    for target, parents in links_dict.items():
        for parent, data in parents.items():
            edges[parent].append((target, data['value']))

    # Recursive function to find all weighted paths from node to outcome
    def get_paths_to_outcome(node, visited=None, cumulative=1.0):
        if visited is None:
            visited = set()
        if node in visited:
            return []
        visited.add(node)
        paths = []
        for child, weight in edges[node]:
            new_cumulative = cumulative * weight
            if child == outcome:
                paths.append(new_cumulative)
            else:
                paths += get_paths_to_outcome(child, visited.copy(), new_cumulative)
        return paths

    # Total effects: sum of all path weights to the outcome
    total_effect = {}
    for node in edges.keys():
        paths = get_paths_to_outcome(node)
        if paths:
            total_effect[node] = sum(paths)

    return direct_effect, total_effect

def calculate_causal_effect_per_unit(conversion_drop: float, channel_series: pd.Series) -> float:
    """
    Calculate the causal effect per unit change in a marketing channel.
    Parameters
    ----------
    conversion_drop : float
        The observed drop in conversions.
    channel_series : pd.Series
        Time series data for the marketing channel.
    Returns
    -------
    float
        Causal effect per unit change in the marketing channel.
    """
    mean_input = channel_series.mean()
    if mean_input == 0:
        raise ValueError("Mean of channel is zero; can't divide by zero.")
    return conversion_drop / mean_input

