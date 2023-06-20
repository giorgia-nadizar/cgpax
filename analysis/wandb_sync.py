import os

import pandas as pd
import wandb

from cgpax.run_utils import __update_config_with_env_data__, __init_environment_from_config__
from cgpax.utils import compute_active_size

import jax.numpy as jnp

if __name__ == '__main__':
    os.system("wandb sync")

    api = wandb.Api(timeout=40)
    entity, project = "giorgianadizar", "cgpax"
    runs = api.runs(entity + "/" + project)
    for wandb_run in runs:
        if wandb_run.state == "finished":
            solver = wandb_run.config["solver"]
            if solver == "cgp" and wandb_run.config.get("levels_back") is not None:
                solver += "-local"
            env_name = wandb_run.config["problem"]["environment"]
            ea = "1+lambda" if wandb_run.config["n_parallel_runs"] > 1 else "mu+lambda"
            seed = wandb_run.config["seed"]
            run_name = f"{env_name}_{solver}_{ea}_{seed}"
            wandb_run.name = run_name
            wandb_run.update()

            print(run_name)

            # download history
            if not os.path.exists(f"data/fitness/{run_name}.csv"):
                dct = wandb_run.scan_history()
                df = pd.DataFrame.from_dict(dct)
                df["solver"] = solver
                df["ea"] = ea
                df["environment"] = env_name
                df["seed"] = df["training.run_id"] if wandb_run.config.get("n_parallel_runs", 0) > 1 else \
                    wandb_run.config[
                        "seed"]
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

                # compute graph sizes
                graph_sizes = []
                environment = __init_environment_from_config__(wandb_run.config)
                __update_config_with_env_data__(wandb_run.config, environment)

                file_names = [f for f in os.listdir(f"{target_dir}/") if f != "config.yaml"]
                for file_name in file_names:
                    genome = jnp.load(f"{target_dir}/{file_name}", allow_pickle=True).astype(int)
                    graph_size, max_size = compute_active_size(genome, wandb_run.config)
                    info = file_name.split("_")
                    seed = info[0] if wandb_run.config.get("n_parallel_runs", 0) > 1 else wandb_run.config["seed"]
                    generation = info[1]
                    graph_sizes.append({
                        "seed": str(seed),
                        "generation": generation,
                        "graph_size": graph_size,
                        "max_size": max_size
                    })
                graph_df = pd.DataFrame.from_dict(graph_sizes)
                graph_df["solver"] = solver
                graph_df["ea"] = ea
                graph_df["environment"] = env_name
                graph_df.to_csv(f"data/graph_size/{run_name}.csv")
