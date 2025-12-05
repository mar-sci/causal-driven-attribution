# reproduce/generate_data.py

import os
import pickle
import numpy as np
from tqdm import tqdm
from causalDA.data_generation import DataGenerator, DataGenerationError
from reproduce.utils import config
from reproduce.utils.helpers import convert_effect_dict_to_links_dict, snake_case

OUTPUT_DIR = "reproduce/results/data/"


def main():
    """
    Generate synthetic datasets and save to disk.
    For each random seed in the specified range, this function:
    1. Initializes a DataGenerator with the given configuration.
    2. Generates a random DAG structure.
    3. Samples a random conversion dictionary.
    4. Generates synthetic time series data based on the DAG and conversion dict.
    5. Saves the generated graph, data, links_dict, and conversion_dict to a
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for seed in tqdm(range(config.N_SEEDS), desc="Generating datasets", unit="run"):
        np.random.seed(seed)
        try:
            gen = DataGenerator(
                node_lookup=config.NODE_LOOKUP,
                name_activity=config.ACTIVITY_NAME,
                target_node=config.TARGET_NODE,
                seed=seed,
            )

            graph = gen.generate_random_dag(edge_prob=config.EDGE_PROB)

            conversion_dict = config.sample_conversion_dict()

            df, _, effect_dict = gen.generate_data(
                influence_from_parents=config.INFLUENCE_FROM_PARENTS,
                conversion_dict=conversion_dict,
                time_periods=config.TIME_PERIODS,
                base_range=config.BASE_RANGE,
                carryover=False,
            )

            df.columns = [snake_case(col) for col in df.columns]
            all_channels = [snake_case(ch) + '_' + config.ACTIVITY_NAME for ch in config.NODE_LOOKUP.values() if ch != config.TARGET_NODE]
            target_node = snake_case(config.TARGET_NODE)
            data = df[[target_node] + all_channels]
            

            links_dict = convert_effect_dict_to_links_dict(effect_dict)

            result = {
                "seed": seed,
                "graph": graph,
                "data": data,
                "links_dict": links_dict,
                "conversion_dict": conversion_dict,
            }

            out_file = os.path.join(OUTPUT_DIR, f"graph_{seed:04d}.pkl")
            with open(out_file, "wb") as f:
                pickle.dump(result, f)

        except DataGenerationError as e:
            tqdm.write(f"Error at seed {seed}: {e}")


if __name__ == "__main__":
    main()
