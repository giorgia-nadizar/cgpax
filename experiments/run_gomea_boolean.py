import time
from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import default_backend, vmap
from jax import random

import cgpax
from cgpax.evaluation.boolean_evaluation import evaluate_cgp_genome, evaluate_lgp_genome
from cgpax.functions import function_set_boolean
from cgpax.gomea.fos import compute_fos
from cgpax.gomea.gom import parallel_gom
from cgpax.run_utils import compute_masks, compute_genome_transformation_function, process_dictionary, \
    update_config_with_data, load_dataset
from cgpax.standard import individual


def run_gomea_boolean(config: Dict) -> None:
    if "n_evaluations" not in config:
        config["n_evaluations"] = config["n_generations"] * config["n_individuals"]

    rnd_key = random.PRNGKey(config["seed"])

    x_values, y_values = load_dataset(config["problem"])

    # assert this
    config["use_input_constants"] = False

    update_config_with_data(config, x_values.shape[1], y_values.shape[1], function_set=function_set_boolean)

    # preliminary evo steps
    genome_mask, mutation_mask = compute_masks(config)
    genome_transformation_function = compute_genome_transformation_function(config)
    genome_evaluation_function = evaluate_cgp_genome if config["solver"] == "cgp" else evaluate_lgp_genome
    genome_to_fitness = partial(genome_evaluation_function, config=config, x_values=x_values, y_values=y_values)

    def genomes_to_fitnesses(genotypes: jnp.ndarray, fake_rnd_keys: jnp.ndarray = None) -> jnp.ndarray:
        return vmap(genome_to_fitness)(genotypes)["accuracy"]

    rnd_key, genome_key = random.split(rnd_key, 2)
    genomes = individual.generate_population(pop_size=config["n_individuals"],
                                             genome_mask=genome_mask, rnd_key=genome_key,
                                             genome_transformation_function=genome_transformation_function)

    bias_matrix = None  # init needed for gom

    # evaluate population
    _fitness_evaluation = 0

    start_eval_time = time.time()
    fitnesses = genomes_to_fitnesses(genomes)
    eval_time = time.time() - start_eval_time
    print(
        f"{_fitness_evaluation} \t"
        f"FITNESS: {jnp.max(fitnesses)} \t "
        f"E: {eval_time:.2f} \t"
    )
    with open(f"results/{config['run_name']}.csv", "a") as csv_file:
        csv_file.write("evaluation,fitness,time\n")
        csv_file.write(f"0,{jnp.max(fitnesses)},{eval_time:.2f}\n")

    times = {}
    # evolutionary loop
    while _fitness_evaluation < config["n_evaluations"]:
        # fos computation
        fos_start_time = time.process_time()
        rnd_key, fos_key = random.split(rnd_key, 2)
        fos, bias_matrix = compute_fos(genomes, rnd_key, config, bias_matrix, ignore_full_list=True)
        times["fos_time"] = time.process_time() - fos_start_time
        print("FOS DONE")

        gom_start_time = time.process_time()
        genomes, fitnesses, fitnesses_history = parallel_gom(genomes, fitnesses, fos, genomes_to_fitnesses, rnd_key,
                                                             track_fitnesses=True, intermediate_prints=True)
        times["gom_time"] = time.process_time() - gom_start_time
        avg_gom_time = times["gom_time"] / len(fos)

        with open(f"results/{config['run_name']}.csv", "a") as csv_file:
            for details_dict in fitnesses_history:
                csv_file.write(
                    f"{_fitness_evaluation + details_dict['evaluation']},{details_dict['max_fitness']},{avg_gom_time:.2f}\n"
                )

        _fitness_evaluation += len(fos) * len(genomes)

        # print progress
        print(
            f"{_fitness_evaluation} \t"
            f"F: {times['fos_time']:.2f} \t"
            f"G: {times['gom_time']:.2f} \t"
            f"FITNESS: {jnp.max(fitnesses)}"
        )

        if (config.get("early_stop", False) and config.get("target_fitness", None) is not None and
                jnp.max(fitnesses) >= config["target_fitness"]):
            print(f"Fitness reached target of {config['target_fitness']}")
            break


if __name__ == '__main__':

    print(f"Starting the run with {default_backend()} as backend...")

    entity, project = "giorgianadizar", "cgpax"

    config_files = ["configs/graph_gp_gomea_boolean.yaml"]
    unpacked_configs = []

    for config_file in config_files:
        unpacked_configs += process_dictionary(cgpax.get_config(config_file))

    print(f"Total configs found: {len(unpacked_configs)}")
    for count, cfg in enumerate(unpacked_configs):
        cfg["run_name"] = f"gomea_{cfg['solver']}_{cfg['problem']}_{cfg['seed']}"
        print(cfg["run_name"])
        run_gomea_boolean(cfg)
        print()
