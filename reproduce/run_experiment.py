import os
import argparse
import pickle
import pandas as pd
from tqdm import tqdm

from causalDA.model import CausalModel
from causalDA.evaluation import (
    dict_to_binary_matrix,
    evaluate_binary,
    save_evaluation_results,
)
from reproduce.utils.config import ACTIVITY_NAME, BETA, NODE_LOOKUP, PC_ALPHA, ALPHA_LEVEL, TARGET_NODE
from reproduce.utils.helpers import snake_case, str2bool


def run_experiment(seed: int, tau: int, priority_flag=False) -> dict:
    """
    Run causal discovery and evaluation for a given seed and tau.

    This function loads synthetic data generated for a specific random seed,
    performs causal discovery using Tigramite's PCMCI algorithm, evaluates
    the discovered graph against the ground-truth adjacency matrix, and saves
    the evaluation results (including metrics and discovered DAG) to disk.

    Workflow
    --------
    1. Load synthetic dataset for the given seed (from pickle file).
    2. Run PCMCI causal discovery with the specified tau (maximum lag).
    3. Extract lagged significant links and prune bidirectional edges.
    4. Convert the discovered DAG into an adjacency matrix.
    5. Evaluate the predicted matrix against the ground-truth matrix
       using metrics such as AUC, FPR, TPR, and F-beta.
    6. Save results as a JSON file inside ``results/eval/``.

    Parameters
    ----------
    seed : int
        Random seed index used to select the synthetic dataset.
    tau : int
        Maximum time lag (tau_max) for PCMCI.
    priority_flag : bool, default=False
        Whether to prioritize pruning cycles based on the target node.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics, e.g.:
        {
            "shd": float,
            "sid": float,
            "AUC": float,
            "FPR": float,
            "TPR": float,
            "F0.5": float
        }

    Raises
    ------
    FileNotFoundError
        If the synthetic dataset file for the given seed does not exist.
    RuntimeError
        If causal discovery or evaluation fails unexpectedly.

    Saves
    -----
    JSON file at:
    ``results/eval/eval_seed_{seed:04d}_tau{tau}.json``

    The JSON contains:
    - seed: The random seed used.
    - tau: The tau_max used in PCMCI.
    - predicted_dag: The discovered DAG in dictionary format.
    - metrics: Evaluation metrics (AUC, FPR, TPR, F-beta).
    """

    # --- Load synthetic data for this seed ---
    results_dir = "reproduce/results"
    run_path = os.path.join(results_dir, "data", f"graph_{seed:04d}.pkl")

    if not os.path.exists(run_path):
        raise FileNotFoundError(f"Missing synthetic run file: {run_path}")

    with open(run_path, "rb") as f:
        run_data = pickle.load(f)

    df = run_data["data"]
    graph = run_data["graph"]
    

    # --- Causal discovery ---
    # Assume df is your synthetic dataset
    selected_columns = df.columns.tolist()
    model = CausalModel(df, selected_columns, verbose=0)
    
    model.run_pcmci(
        tau_max=tau, 
        pc_alpha=PC_ALPHA, 
        alpha_level=ALPHA_LEVEL
        )

    lagged_links = model.get_lagged_links(alpha_level=ALPHA_LEVEL, drop_source=TARGET_NODE)
    dag_links = model.prune_bidirectional_links(lagged_links)
    pruned_dag = model.prune_cycles(dag_links, priority_flag=priority_flag ,priority_node=TARGET_NODE)
    

    # --- Evaluation ---
    pred_matrix = dict_to_binary_matrix(pruned_dag, NODE_LOOKUP)
    results = evaluate_binary(graph, pred_matrix, beta=BETA)

    # --- Save evaluation ---
    eval_dir = os.path.join(results_dir, "eval")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    if priority_flag:
        out_dir = os.path.join(eval_dir, f"priority_pruned/seed_{seed:04d}")
    else:
        out_dir = os.path.join(eval_dir, f"no_priority/seed_{seed:04d}")
        
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"tau{tau}.json")
    save_evaluation_results(
        out_path,
        seed=seed,
        tau=tau,
        predicted_dag=pruned_dag,
        metrics=results,
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Run causal discovery experiments.")
    parser.add_argument("--seed_min", type=int, required=True, help="Minimum random seed index")
    parser.add_argument("--seed_max", type=int, required=True, help="Maximum random seed index")
    parser.add_argument("--tau_min", type=int, default=1, help="Minimum tau")
    parser.add_argument("--tau_max", type=int, default=30, help="Maximum tau")
    parser.add_argument(
        "--priority",
        type=str2bool,
        default=False,
        help="Whether to use priority pruning to the target node (True/False)"
    )
    args = parser.parse_args()

    print(f"Running experiments for seeds {args.seed_min}..{args.seed_max}, "
          f"tau={args.tau_min}..{args.tau_max}")

    for seed in tqdm(range(args.seed_min, args.seed_max + 1), desc="Seed sweep"):
        for tau in tqdm(range(args.tau_min, args.tau_max + 1), desc=f"Tau sweep (seed={seed})", leave=False):
            results = run_experiment(seed=seed, tau=tau, priority_flag=args.priority)
            tqdm.write(f"Seed {seed}, tau={tau} → {results}")


if __name__ == "__main__":
    main()