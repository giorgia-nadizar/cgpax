import os
import sys

import pandas as pd
import wandb

from cgpax.jax_functions import available_functions
from cgpax.run_utils import update_config_with_env_data, init_environment_from_config, config_to_run_name
from cgpax.utils import compute_active_size, graph_from_genome, interpretability_from_genome

import jax.numpy as jnp

if __name__ == '__main__':
    flag = "gp" if len(sys.argv) <= 1 else sys.argv[1].lower()
    assert flag in ["all", "rl", "gp"], "Argument not in the allowed list."

    name_patterns = [] if len(sys.argv) <= 2 else sys.argv[2:]

    os.system("wandb sync")

    api = wandb.Api(timeout=40)
    entity, project = "giorgianadizar", "cgpax"
    runs = api.runs(entity + "/" + project)

    if flag in ["all", "gp"]:
        counter = 0
        for wandb_run in runs:
            if wandb_run.state == "finished" and not wandb_run.name.startswith("RL"):
                process = len(name_patterns) == 0
                for name_pattern in name_patterns:
                    if name_pattern in wandb_run.name:
                        process = True
                        break
                if process:
                    run_name, env_name, solver, ea, fitness, seed = config_to_run_name(wandb_run.config,
                                                                                       wandb_run.created_at)
                    wandb_run.name = run_name
                    wandb_run.update()

                    print(f"{counter} -> {run_name}")
                    counter += 1

                    # download history
                    if not os.path.exists(f"data/fitness/{run_name}.csv"):
                        dct = wandb_run.scan_history()
                        df = pd.DataFrame.from_dict(dct)
                        df["solver"] = solver
                        df["ea"] = ea
                        df["fitness"] = fitness
                        df["environment"] = env_name
                        df["seed"] = df["training.run_id"] if wandb_run.config.get("n_parallel_runs", 0) > 1 else \
                            wandb_run.config[
                                "seed"]
                        df["training.evaluation"] = df["training.generation"] * wandb_run.config["n_individuals"]
                        for i in range(3):
                            if f"training.top_k_reward.top_{i}_reward" not in df.columns:
                                df[f"training.top_k_reward.top_{i}_reward"] = df[f"training.top_k_fit.top_{i}_fit"]

                        # top_k_reward
                        df.to_csv(f"data/fitness/{run_name}.csv")

                    # download genomes
                    target_dir = f"../analysis/genomes/{wandb_run.name}"
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                        files = wandb_run.files()
                        for file in files:
                            if file.name.startswith("genomes/") or file.name == "config.yaml":
                                file.name = file.name.replace("genomes/", "")
                                file.download(root=target_dir)

                        # compute graph sizes and intepretability measures
                        graph_sizes = []
                        interpretability_measures = []
                        environment = init_environment_from_config(wandb_run.config)
                        update_config_with_env_data(wandb_run.config, environment)

                        file_names = [f for f in os.listdir(f"{target_dir}/") if f != "config.yaml"]
                        for file_name in file_names:
                            genome = jnp.load(f"{target_dir}/{file_name}", allow_pickle=True).astype(int)

                            # info on graph size
                            graph_size, max_size = compute_active_size(genome, wandb_run.config)
                            info = file_name.split("_")
                            seed = info[0] if wandb_run.config.get("n_parallel_runs", 0) > 1 else wandb_run.config[
                                "seed"]
                            generation = info[1]
                            graph_sizes.append({
                                "seed": str(seed),
                                "generation": generation,
                                "evaluation": generation * wandb_run.config["n_individuals"],
                                "graph_size": graph_size,
                                "max_size": max_size
                            })
                            # info on interpretability
                            graph = graph_from_genome(genome, wandb_run.config)
                            single_arity = 0
                            double_arity = 0
                            n_inputs = 0
                            for node in graph.iternodes():
                                if "_" in node:
                                    if node.startswith("i_"):
                                        n_inputs += 1
                                    continue
                                node_symbol = node.split(" ")[0]
                                if node_symbol in [x.symbol for x in available_functions.values() if x.arity == 1]:
                                    single_arity += 1
                                else:
                                    double_arity += 1
                            phi = interpretability_from_genome(genome, wandb_run.config)
                            interpretability_measures.append({
                                "seed": str(seed),
                                "generation": generation,
                                "evaluation": generation * wandb_run.config["n_individuals"],
                                "n_edges": graph.number_of_edges(),
                                "phi": phi,
                                "n_single_arity": single_arity,
                                "n_double_arity": double_arity,
                                "n_inputs": n_inputs
                            })

                        graph_df = pd.DataFrame.from_dict(graph_sizes)
                        graph_df["solver"] = solver
                        graph_df["ea"] = ea
                        graph_df["fitness"] = fitness
                        graph_df["environment"] = env_name
                        graph_df.to_csv(f"data/graph_size/{run_name}.csv")

                        interpretability_df = pd.DataFrame.from_dict(interpretability_measures)
                        interpretability_df["solver"] = solver
                        interpretability_df["ea"] = ea
                        interpretability_df["fitness"] = fitness
                        interpretability_df["environment"] = env_name
                        interpretability_df.to_csv(f"data/interpretability/{run_name}.csv")

    if flag in ["all", "rl"]:
        counter = 0
        for wandb_run in runs:
            if wandb_run.state == "finished" and wandb_run.name.startswith("RL") and not os.path.exists(
                    f"data/rl/{wandb_run.name.replace('RL_', '')}.csv"):
                process = len(name_patterns) == 0
                for name_pattern in name_patterns:
                    if name_pattern in wandb_run.name:
                        process = True
                        break
                if process:
                    print(f"{counter} -> {wandb_run.name}")
                    counter += 1

                    # download history
                    split_name = wandb_run.name.split("_")
                    ddf = wandb_run.history(pandas=True)
                    ddf["seed"] = split_name[-1]
                    ddf["rl_algorithm"] = split_name[-2]
                    ddf["environment"] = "_".join(split_name[1:-2])
                    ddf.to_csv(f"data/rl/{wandb_run.name.replace('RL_', '')}.csv")
