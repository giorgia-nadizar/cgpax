import os

import pandas as pd

import jax.numpy as jnp
from jax import random

import cgpax
from analysis.genome_analysis import __load_last_genome__
from cgpax.jax_evaluation import evaluate_cgp_genome, evaluate_lgp_genome
from cgpax.run_utils import __init_environment_from_config__, __update_config_with_env_data__


def compute_rewards_df(genome: jnp.ndarray, rnd_key: random.PRNGKey, config: dict) -> pd.DataFrame:
    env = __init_environment_from_config__(config)
    __update_config_with_env_data__(config, env)
    if config["solver"] == "cgp":
        result = evaluate_cgp_genome(genome, rnd_key, config, env)
    elif config["solver"] == "lgp":
        result = evaluate_lgp_genome(genome, rnd_key, config, env)
    else:
        raise ValueError
    healthy = result["healthy_rewards"].tolist()
    ctrl = result["ctrl_rewards"].tolist()
    forward = result["forward_rewards"].tolist()
    cum = result["cum_reward"]

    df = pd.DataFrame(list(zip(healthy, ctrl, forward)), columns=["healthy", "ctrl", "forward"])
    df["cumulative"] = cum
    return df


def write_rewards_df(base_path: str, seed: int, generation: int, target_file: str, rnd_key: random.PRNGKey) -> None:
    genome = jnp.load(f"{base_path}/{seed}_{generation}_best_genome.npy", allow_pickle=True).astype(int)
    config = cgpax.get_config(f"{base_path}/config.yaml")
    dataframe = compute_rewards_df(genome, rnd_key, config)
    dataframe.to_csv(target_file, index=False)


if __name__ == '__main__':
    main_target_dir = "data/rewards"

    for outer_seed in range(10):
        for folder in os.listdir("genomes"):
            if not folder.endswith(f"_{outer_seed}"):
                continue

            base_p = f"genomes/{folder}"
            rand_key = random.PRNGKey(outer_seed)
            cfg = cgpax.get_config(f"{base_p}/config.yaml")
            for inner_seed in range(10):
                try:
                    genes, gen = __load_last_genome__(base_p, inner_seed)
                except:
                    continue

                target_f = f"{folder.replace(f'_{outer_seed}', f'_{inner_seed}')}_{gen}.csv"
                if target_f not in os.listdir("data/rewards"):
                    print(f"{folder} -> {inner_seed}")
                    write_rewards_df(base_p, outer_seed, gen, f"data/rewards/{target_f}", rand_key)
