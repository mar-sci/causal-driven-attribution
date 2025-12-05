import argparse
import random
import re
from tqdm import tqdm
import networkx as nx
import pandas as pd
from causalDA.evaluation import calculate_causal_effect_per_unit, compute_causal_effects
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

from causalDA.data_generation import DataGenerator
from causalDA.model import CausalModel
from reproduce.utils.config import ACTIVITY_NAME, BASE_RANGE, EDGE_PROB, INFLUENCE_FROM_PARENTS, NODE_LOOKUP, TARGET_NODE, TIME_PERIODS, sample_conversion_dict
from reproduce.utils.helpers import build_causal_digraph, compute_causal_layers, convert_effect_dict_to_links_dict, snake_case, snake_case_dict, str2bool
from dowhy import gcm
from datetime import datetime
import pickle

random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

try:
    import torch
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except ImportError:
    pass


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def estimate_conversion_drop_via_sampling_repeated(
    scm,
    data: pd.DataFrame,
    channel: str,
    target: str = "conversion",
    n_runs: int = 5000
) -> float:
    """
    Estimate conversion drop by repeatedly sampling from the SCM
    with and without intervention on the specified channel.
    Parameters
    ----------
    scm : gcm.StructuralCausalModel
        The structural causal model.
    data : pd.DataFrame
        The observed data.
    channel : str
        The channel to intervene on (set to zero).
    target : str, default: conversion
        The target variable to measure drop in.
    n_runs : int, default: 5000
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

def transform_links_dict(d, activity, target_node):
    """
    Transform links_dict to append activity suffix to each node,
    except for the target node.
    
    Parameters
    ----------
    d : dict
        Original links_dict in the format {child: {parent: value}}.
    activity : str
        Activity suffix to append.
    target_node : str
        The target node which should not be suffixed.
    Returns
    -------
    dict
        Transformed links_dict with suffixed node names.
    """

    new_dict = {}
    for parent, children in d.items():
        new_parent = snake_case(parent) + f'_{activity}' if parent != target_node else target_node
        new_children = {}
        for child, value in children.items():
            new_child = snake_case(child) + f'_{activity}'
            new_children[new_child] = value
        new_dict[new_parent] = new_children
    return new_dict

def run_evaluation_for_seed(seed, tau_max, priority_flag, gen, df, links_dict):
    """
    Run causal discovery and ACE evaluation for a given seed.
    Parameters
    ----------
    seed : int
        Random seed index used to select the synthetic dataset.
    tau_max : int
        Maximum time lag (tau_max) for PCMCI.
    priority_flag : bool, default=False
        Whether to prioritize pruning cycles based on the target node.
    gen : DataGenerator
        The data generator instance used to create the synthetic data.
    df : pd.DataFrame
        The synthetic dataset.
    links_dict : dict
        The ground truth causal links dictionary.
    Returns
    -------
    dict
        Dictionary containing evaluation metrics, e.g.:
        {
            "true_relative_rmse": float,
            "true_mape": float,
            "true_spearman_corr": float,
            "model_relative_rmse": float,
            "model_mape": float,
            "model_spearman_corr": float,
        }
    """

    # 1. snake_case columns
    df.columns = [snake_case(col) for col in df.columns]
    # 2. run PCMCI
    # Get causal DAG with edge weights
    dag_result = gen.get_causal_graph()

    # Causal Discovery
    all_channels = [snake_case(ch) + '_' + ACTIVITY_NAME for ch in NODE_LOOKUP.values() if ch != TARGET_NODE]
    target_node = snake_case(TARGET_NODE)
    selected_columns = all_channels + [target_node]
    data = df[[target_node] + all_channels]

    model = CausalModel(data, selected_columns, verbose=0)
    results = model.run_pcmci(tau_max=tau_max, pc_alpha=0.2, alpha_level=0.05)
    # 3. build graphs
    lagged_links = model.get_lagged_links(alpha_level=0.05, drop_source=target_node)
    dag_links = model.prune_bidirectional_links(lagged_links)


    pruned_dag = model.prune_cycles(dag_links, priority_node=target_node, priority_flag=priority_flag, verbose=False)
    model_causal_graph = build_causal_digraph(pruned_dag)
    true_dag = transform_links_dict(links_dict, ACTIVITY_NAME, target_node)
    true_causal_graph = build_causal_digraph(true_dag)
    # 4. estimate ACE
    # Extract edges using regex
    true_edges = re.findall(r'(\w+)\s*->\s*(\w+)', true_causal_graph)
    model_edges = re.findall(r'(\w+)\s*->\s*(\w+)', model_causal_graph)

    # Create the DiGraph
    true_converted_causal_graph = nx.DiGraph(true_edges)
    model_converted_causal_graph = nx.DiGraph(model_edges)

    # Create the structural causal model object
    scm_true = gcm.StructuralCausalModel(true_converted_causal_graph)
    scm_model = gcm.StructuralCausalModel(model_converted_causal_graph)

    # Automatically assign generative models to each node based on the given data
    random.seed(42); np.random.seed(42)
    gcm.auto.assign_causal_mechanisms(scm_true, data)
    random.seed(42); np.random.seed(42)
    gcm.auto.assign_causal_mechanisms(scm_model, data)

    # Fit the SCMs
    gcm.fit(scm_true, data)
    gcm.fit(scm_model, data)

    true_ACE_result = {} # Average Causal Effect
    model_ACE_result = {} # Average Causal Effect

    for source in all_channels:
        # True DAG ACE
        channel_drop = estimate_conversion_drop_via_sampling_repeated(scm_true, data, source, target = target_node, n_runs = 5000)
        channel_mean = data[source].mean()
        causal_effect = calculate_causal_effect_per_unit(channel_drop, data[source])
        channel_name = source[:-(len(ACTIVITY_NAME)+1)]
        true_ACE_result[channel_name] = {
            "conversion_drop": channel_drop,
            "cate": causal_effect
        }

        # Model DAG ACE
        channel_drop = estimate_conversion_drop_via_sampling_repeated(scm_model, data, source, target = target_node, n_runs = 5000)
        channel_mean = data[source].mean()
        causal_effect = calculate_causal_effect_per_unit(channel_drop, data[source])
        channel_name = source[:-(len(ACTIVITY_NAME)+1)]
        model_ACE_result[channel_name] = {
            "conversion_drop": channel_drop,
            "cate": causal_effect
        }
    # Ground Truth
    _, total_effect = compute_causal_effects(links_dict)

    # 5. compute RMSE, MAPE, Spearman
    # ======================================================
    # Ground Truth ACE Evaluation
    # ======================================================

    # Normalize direct_effect keys
    normalized_total = {snake_case(k): v for k, v in total_effect.items()}
    true_normalized_ace = {k: v['cate'] for k, v in true_ACE_result.items()}
    model_normalized_ace = {k: v['cate'] for k, v in model_ACE_result.items()}

    # Align keys - Ground Truth
    true_common_keys = list(set(normalized_total.keys()) & set(true_normalized_ace.keys()))
    true_total_vals = [normalized_total[k] for k in true_common_keys]
    true_ace_vals = [true_normalized_ace[k] for k in true_common_keys]
    

    # Convert to numpy arrays
    true_total_vals = np.array(true_total_vals, dtype=float)
    true_ace_vals   = np.array(true_ace_vals, dtype=float)

    # ---------- 1. Relative RMSE ----------
    true_rmse = np.sqrt(np.mean((true_ace_vals - true_total_vals) ** 2))
    true_relative_rmse = (true_rmse / np.mean(np.abs(true_total_vals))) * 100
    

    # ---------- 2. MAPE ----------
    true_mape = np.mean(np.abs((true_ace_vals - true_total_vals) / true_total_vals)) * 100
    

    # ---------- 3. Spearman rank correlation ----------
    true_spear_corr, _ = spearmanr(true_ace_vals, true_total_vals)

    # ======================================================
    # Model ACE Evaluation
    # ======================================================

    model_common_keys = list(set(normalized_total.keys()) & set(model_normalized_ace.keys()))
    model_total_vals = [normalized_total[k] for k in model_common_keys]
    model_ace_vals = [model_normalized_ace[k] for k in model_common_keys]

    # Convert to numpy arrays
    model_total_vals = np.array(model_total_vals, dtype=float)
    model_ace_vals   = np.array(model_ace_vals, dtype=float)

    # ---------- 1. Relative RMSE ----------
    model_rmse = np.sqrt(np.mean((model_ace_vals - model_total_vals) ** 2))
    model_relative_rmse = (model_rmse / np.mean(np.abs(model_total_vals))) * 100
    

    # ---------- 2. MAPE ----------
    model_mape = np.mean(np.abs((model_ace_vals - model_total_vals) / model_total_vals)) * 100

    # ---------- 3. Spearman rank correlation ----------
    model_spear_corr, _ = spearmanr(model_ace_vals, model_total_vals)

    return {
        "true_relative_rmse": true_relative_rmse,
        "true_mape": true_mape,
        "true_spearman_corr": true_spear_corr,
        "model_relative_rmse": model_relative_rmse,
        "model_mape": model_mape,
        "model_spearman_corr": model_spear_corr,
    }


def run_evaluation_uniform_layers(num_per_layers=200, tau_max=45, priority_flag=True):
    """
    Run multiple causal discovery and ACE evaluations,
    ensuring uniform distribution across different numbers of causal layers.
    Parameters
    ----------
    num_per_layers : int, default=200
        Number of samples to generate for each distinct number of causal layers (from 2 to 6).
    tau_max : int, default=45
        Maximum time lag (tau_max) for PCMCI.
    priority_flag : bool, default=True
        Whether to prioritize pruning cycles based on the target node.
    Returns
    -------
    dict
        Dictionary containing evaluation results for each seed, e.g.:
        {
            seed: {
                "n_layers": int,
                "layers": list of list,
                "true_relative_rmse": float,
                "true_mape": float,
                "true_spearman_corr": float,
                "model_relative_rmse": float,
                "model_mape": float,
                "model_spearman_corr": float,
            },
            ...
        }
    """
    results = {}
    counts = {n: 0 for n in range(2, 7)}  # Track how many times each n_layer has appeared
    tried_seeds = set()
    seed = 0

    with tqdm(total=num_per_layers * len(counts)) as pbar:
        while any(v < num_per_layers for v in counts.values()):
            if seed in tried_seeds:
                seed += 1
                continue
            tried_seeds.add(seed)

            try:
                # Init generator
                gen = DataGenerator(
                    node_lookup=NODE_LOOKUP,
                    name_activity=ACTIVITY_NAME,
                    target_node=TARGET_NODE,
                    seed=seed,
                )

                # Generate DAG
                graph = gen.generate_random_dag(edge_prob=EDGE_PROB)

                # Sample conversion effectiveness
                conversion_dict = sample_conversion_dict()

                # Generate synthetic data
                df, dict_contributions, effect_dict = gen.generate_data(
                    influence_from_parents=INFLUENCE_FROM_PARENTS,
                    conversion_dict=conversion_dict,
                    time_periods=TIME_PERIODS,
                    base_range=BASE_RANGE,
                    carryover=False,
                )

                links_dict = convert_effect_dict_to_links_dict(effect_dict)

                # Compute number of layers
                n_layers, layers = compute_causal_layers(links_dict)

                if n_layers not in counts or counts[n_layers] >= num_per_layers:
                    seed += 1
                    continue  # Skip if we already have enough for this group
                
                print(f"\nSeed {seed} | Causal Layers: {n_layers} | Layers: {layers}")
                # Run evaluation
                evaluation = run_evaluation_for_seed(seed, tau_max, priority_flag, gen, df, links_dict)  # You can modularize this

                # Store result
                results[seed] = {
                    "n_layers": n_layers,
                    "layers": layers,
                    **evaluation
                }

                counts[n_layers] += 1
                pbar.update(1)
            except Exception as e:
                print(f"Skipping seed {seed} due to error: {e}")
            finally:
                seed += 1

    return results


def main():
    parser = argparse.ArgumentParser(description="Run multiple samples CDA.")
    parser.add_argument("--num_per_layers", type=int, required=True, help="Number of samples per layer count")
    parser.add_argument(
        "--priority",
        type=str2bool,
        default=False,
        help="Whether to use priority pruning to the target node (True/False)"
    )
    parser.add_argument("--tau_max", type=int, default=45, help="Maximum tau")

    args = parser.parse_args()

    num_per_layers = args.num_per_layers
    priority_flag = args.priority
    tau_max = args.tau_max
    
    # ---------------------------------------------------
    # Run evaluation
    # ---------------------------------------------------
    results = run_evaluation_uniform_layers(
        num_per_layers=num_per_layers,
        tau_max=tau_max,
        priority_flag=priority_flag,
    )

    # print("Final Results:", results)

    # ---------------------------------------------------
    # Save results to results/multi_simulation/
    # ---------------------------------------------------
    output_dir = "reproduce/results/multi_simulation"
    os.makedirs(output_dir, exist_ok=True)

    # timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"results_{num_per_layers}_tau_{tau_max}_{timestamp}.pkl")

    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\nPickle saved to: {output_path}\n")


if __name__ == "__main__":
    main()
