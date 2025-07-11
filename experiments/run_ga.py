import sys
import time
from typing import Dict

import jax.numpy as jnp
from jax import default_backend, vmap
from jax import random

import cgpax
from cgpax.evaluation.evaluation_utils import prepare_evaluation_functions
from cgpax.run_utils import compute_masks, compute_genome_transformation_function, process_dictionary, \
    compile_parents_selection, compile_crossover, compile_mutation, compile_survival_selection, parse_args
from cgpax.standard import individual


def run_ga(config: Dict) -> None:
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
                       f"time\n")

    times = {}
    best_test_accuracy = None
    offspring = None
    survivals_fitnesses = None
    # evolutionary loop
    for _generation in range(config["n_generations"]):
        # evaluate population
        start_eval_time = time.time()
        if _generation ==0 or config.get("reassess", True):
            fitnesses = genomes_to_fitnesses(genomes)
        else:
            new_fitnesses = genomes_to_fitnesses(offspring)
            fitnesses = jnp.concatenate((survivals_fitnesses, new_fitnesses))
        eval_time = time.time() - start_eval_time

        if genome_to_test_accuracy is not None:
            # compute test fitness
            best_individual = genomes[jnp.argmax(fitnesses)]
            best_test_accuracy = genome_to_test_accuracy(best_individual)

        with open(f"../results/{cfg['run_name']}.csv", "a") as csv_file:
            csv_file.write(f"{_generation * len(fitnesses)},{jnp.max(fitnesses)},"
                           f"{best_test_accuracy if genome_to_test_accuracy is not None else ''}"
                           f"{',' if genome_to_test_accuracy is not None else ''}"
                           f"{eval_time:.2f}\n")

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


if __name__ == '__main__':

    print(f"Starting the run with {default_backend()} as backend...")

    # problem_types = ["boolean", "classification", "regression", "discrete_control", "continuous_control"]
    problem_types = ["regression"]
    args = parse_args(sys.argv[1:])

    for problem_type in problem_types:

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
            cfg["run_name"] = f"ga_{cfg['solver']}_{problem_name}_{cfg['seed']}"
            print(cfg["run_name"])
            run_ga(cfg)
            print()
