import time
from functools import partial
from typing import Dict

import jax.numpy as jnp
import numpy as np
from jax import default_backend, vmap
from jax import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from wandb.sdk.wandb_run import Run

import cgpax
from cgpax.evaluation.classification_evaluation import evaluate_cgp_genome, evaluate_lgp_genome
from cgpax.functions import function_set_numeric
from cgpax.run_utils import compile_parents_selection, compile_mutation, \
    compute_masks, compute_genome_transformation_function, compile_survival_selection, compile_crossover, \
    process_dictionary, update_config_with_data, \
    load_dataset
from cgpax.standard import individual


def run(config: Dict, wandb_run: Run) -> None:
    rnd_key = random.PRNGKey(config["seed"])

    x_values, y_values = load_dataset(config["problem"])
    n_classes = len(set(y_values))
    train_size = config.get("train_size", 0.8)

    # split train and test
    x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=(1. - train_size))

    # standardize
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    update_config_with_data(config, x_train.shape[1], n_classes, function_set=function_set_numeric)
    # wandb.config.update(config, allow_val_change=True)

    # preliminary evo steps
    genome_mask, mutation_mask = compute_masks(config)
    genome_transformation_function = compute_genome_transformation_function(config)
    genome_evaluation_function = evaluate_cgp_genome if config["solver"] == "cgp" else evaluate_lgp_genome
    genome_to_fitness = partial(genome_evaluation_function, config=config, x_values=x_train, y_values=y_train)

    def genomes_to_fitnesses(genotypes: jnp.ndarray, fake_rnd_keys: jnp.ndarray = None) -> float:
        return vmap(genome_to_fitness)(genotypes)["accuracy"]

    # compilation of functions
    select_parents = compile_parents_selection(config)
    crossover_genomes = compile_crossover(config)
    mutate_genomes = compile_mutation(config, genome_mask, mutation_mask, genome_transformation_function)
    select_survivals = compile_survival_selection(config)

    rnd_key, genome_key = random.split(rnd_key, 2)
    genomes = individual.generate_population(pop_size=config["n_individuals"],
                                             genome_mask=genome_mask, rnd_key=genome_key,
                                             genome_transformation_function=genome_transformation_function)

    with open(f"results/{cfg['run_name']}.csv", "w") as csv_file:
        csv_file.write("iteration,fitness,test_accuracy,time\n")

    times = {}
    # evolutionary loop
    for _generation in range(config["n_generations"]):

        # evaluate population
        start_eval_time = time.time()
        fitnesses = genomes_to_fitnesses(genomes)
        eval_time = time.time() - start_eval_time

        # compute test fitness
        best_individual = genomes[np.argmax(fitnesses)]
        best_test_accuracy = genome_evaluation_function(best_individual, config=config, x_values=x_test,
                                                        y_values=y_test)["accuracy"]

        with open(f"results/{cfg['run_name']}.csv", "a") as csv_file:
            csv_file.write(f"{_generation},{jnp.max(fitnesses)},{best_test_accuracy},{eval_time:.2f}\n")

        print(
            f"{_generation} \t"
            f"FITNESS: {jnp.max(fitnesses)} \t "
            f"E: {eval_time:.2f} \t"
        )

        if (config.get("early_stop", False) and config.get("target_fitness", None) is not None and
                jnp.max(fitnesses) >= config["target_fitness"]):
            print(f"Fitness reached target of {config['target_fitness']}")
            break

        # select parents
        rnd_key, select_key = random.split(rnd_key, 2)
        start_selection = time.process_time()
        parents = select_parents(genomes, fitnesses, select_key)
        end_selection = time.process_time()
        times["selection_time"] = end_selection - start_selection

        # compute offspring
        rnd_key, mutate_key = random.split(rnd_key, 2)
        mutate_keys = random.split(mutate_key, len(parents))
        start_offspring = time.process_time()
        if config.get("crossover", False):
            parents1, parents2 = jnp.split(parents, 2)
            rnd_key, *xover_keys = random.split(rnd_key, len(parents1) + 1)
            offspring1, offspring2 = crossover_genomes(parents1, parents2, jnp.array(xover_keys))
            new_parents = jnp.concatenate((offspring1, offspring2))
        else:
            new_parents = parents
        offspring_matrix = mutate_genomes(new_parents, mutate_keys)
        offspring = jnp.reshape(offspring_matrix, (-1, offspring_matrix.shape[-1]))
        end_offspring = time.process_time()
        times["mutation_time"] = end_offspring - start_offspring

        # select survivals
        rnd_key, survival_key = random.split(rnd_key, 2)
        survivals = parents if select_survivals is None else select_survivals(genomes, fitnesses, survival_key)

        # update population
        assert len(genomes) == len(survivals) + len(offspring)
        genomes = jnp.concatenate((survivals, offspring))


if __name__ == '__main__':

    print(f"Starting the run with {default_backend()} as backend...")

    # telegram_config = cgpax.get_config("telegram/token.yaml")
    # telegram_bot = telegram.Bot(telegram_config["token"])

    # api = wandb.Api(timeout=40)
    entity, project = "giorgianadizar", "cgpax"
    # existing_run_names = [r.name for r in api.runs(entity + "/" + project) if r.state == "finished"]

    config_files = ["configs/graph_gp_classification.yaml"]
    unpacked_configs = []

    for config_file in config_files:
        unpacked_configs += process_dictionary(cgpax.get_config(config_file))

    # notify_update(f"Total configs found: {len(unpacked_configs)}", telegram_bot, telegram_config["chat_id"])
    print(f"Total configs found: {len(unpacked_configs)}")
    for count, cfg in enumerate(unpacked_configs):
        # run_name, _, _, _, _, _ = config_to_run_name(cfg)
        cfg["run_name"] = f"ga_{cfg['solver']}_{cfg['problem']}_{cfg['seed']}"
        print(cfg["run_name"])
        # if run_name in existing_run_names:
        #     notify_update(f"{count + 1}/{len(unpacked_configs)} - {run_name} already exists")
        # continue
        # notify_update(f"{count + 1}/{len(unpacked_configs)} - {run_name} starting\n{cfg}", telegram_bot,
        # telegram_config["chat_id"])
        # wb_run = wandb.init(config=cfg, project=project, name=run_name)
        run(cfg, None)
        # wb_run.finish()
        print()
