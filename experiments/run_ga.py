import sys
import time
from pathlib import Path
from typing import Dict

import jax.numpy as jnp
import yaml
from jax import default_backend, vmap
from jax import random

import cgpax
from cgpax.evaluation.evaluation_utils import prepare_evaluation_functions
from cgpax.run_utils import compute_masks, compute_genome_transformation_function, process_dictionary, \
    compile_parents_selection, compile_crossover, compile_mutation, compile_survival_selection, parse_args
from cgpax.standard import individual


def run_ga(config: Dict) -> None:
    if "n_evaluations" not in config:
        config["n_evaluations"] = config["n_generations"] * config["n_individuals"]

    rnd_key = random.PRNGKey(config["seed"])

    # compose genome eval
    genomes_to_fitnesses, genome_to_test_accuracy = prepare_evaluation_functions(config)
    genome_mask, mutation_mask = compute_masks(config)
    genome_transformation_function = compute_genome_transformation_function(config)

    # compilation of functions
    select_parents = compile_parents_selection(config)
    crossover_genomes = compile_crossover(config)
    mutate_genomes = compile_mutation(config, genome_mask, mutation_mask, genome_transformation_function)
    select_survivals = compile_survival_selection(config)

    rnd_key, genome_key = random.split(rnd_key, 2)
    genomes = individual.generate_population(pop_size=config["n_individuals"],
                                             genome_mask=genome_mask, rnd_key=genome_key,
                                             genome_transformation_function=genome_transformation_function)

    # evaluate population
    with open(f"../results/{config['run_name']}.csv", "a") as csv_file:
        csv_file.write(f"evaluation,fitness,"
                       f"{'test_accuracy,' if genome_to_test_accuracy is not None else ''}"
                       f"uniqueness,time\n")

    times = {}
    best_test_accuracy = None
    offspring = None
    survivals_fitnesses = None
    # evolutionary loop
    n_evaluated = 0
    while n_evaluated < config["n_evaluations"]:
        # evaluate population
        start_eval_time = time.time()
        if n_evaluated == 0 or config.get("reassess", False):
            rnd_key, *eval_keys = random.split(rnd_key, len(genomes) + 1)
            fitnesses = genomes_to_fitnesses(genomes, jnp.array(eval_keys))
            n_evaluated += len(fitnesses)
        else:
            rnd_key, *eval_keys = random.split(rnd_key, len(offspring) + 1)
            new_fitnesses = genomes_to_fitnesses(offspring, jnp.array(eval_keys))
            n_evaluated += len(new_fitnesses)
            fitnesses = jnp.concatenate((survivals_fitnesses, new_fitnesses))
        eval_time = time.time() - start_eval_time

        if genome_to_test_accuracy is not None:
            # compute test fitness
            best_individual = genomes[jnp.argmax(fitnesses)]
            best_test_accuracy = genome_to_test_accuracy(best_individual)

        uniqueness = len(jnp.unique(genomes.astype(int), axis=0)) / len(genomes)

        with open(f"../results/{cfg['run_name']}.csv", "a") as csv_file:
            csv_file.write(f"{n_evaluated},{jnp.max(fitnesses)},"
                           f"{best_test_accuracy if genome_to_test_accuracy is not None else ''}"
                           f"{',' if genome_to_test_accuracy is not None else ''}"
                           f"{uniqueness},{eval_time:.2f}\n")

        print(
            f"{n_evaluated} \t"
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

        # extract fitness of survivals
        comparisons = genomes[None, :, :] == survivals[:, None, :]
        matches = jnp.all(comparisons, axis=2)

        def _get_first_match_index(row_matches):
            row_indices = jnp.arange(row_matches.shape[0])
            masked_indices = jnp.where(row_matches, row_indices, row_matches.shape[0])  # Invalid index = M
            first_idx = jnp.min(masked_indices)
            return jnp.where(first_idx == row_matches.shape[0], -1, first_idx)

        first_match_indices = vmap(_get_first_match_index)(matches)
        survivals_fitnesses = fitnesses[first_match_indices]

        # update population
        assert len(genomes) == len(survivals) + len(offspring)
        genomes = jnp.concatenate((survivals, offspring))

    Path(f"../results/{config['run_name']}").mkdir(parents=True, exist_ok=True)
    jnp.save(f"../results/{config['run_name']}/genomes.npy", genomes)
    jnp.save(f"../results/{config['run_name']}/fitnesses.npy", fitnesses)
    with open(f"../results/{config['run_name']}/config.yml", "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)


if __name__ == '__main__':

    print(f"Starting the run with {default_backend()} as backend...")

    # problem_types = ["boolean", "classification", "regression", "discrete_control", "continuous_control"]
    problem_types = ["continuous_control", "regression"]

    args = parse_args(sys.argv[1:])

    for problem_type in problem_types:
        problem_prefix = problem_type if "_" not in problem_type else problem_type.split("_")[1]

        config_files = [f"../configs/graph_gp_{problem_type}.yaml"]
        unpacked_configs = []

        for config_file in config_files:
            current_config = cgpax.get_config(config_file)
            current_config.update(args)
            unpacked_configs += process_dictionary(current_config)

        print(f"\n\nRunning {problem_type}...")
        print(f"Total configs found: {len(unpacked_configs)}")
        for cfg in unpacked_configs:
            cfg["problem_type"] = problem_type
            problem_name = cfg['problem']['environment'].lower().split("-")[0] if "control" in problem_type \
                else cfg['problem']
            cfg["run_name"] = f"{problem_prefix}/ga_{cfg['solver']}_{problem_name}_{cfg['seed']}"
            if cfg["n_individuals"] > 200:
                cfg["run_name"] += "_large_pop"
            if Path(f"../results/{cfg['run_name']}.csv").exists():
                print(f"{cfg['run_name']} already exists!")
                continue
            print(f"Running {cfg['run_name']}...")
            run_ga(cfg)
            print()
