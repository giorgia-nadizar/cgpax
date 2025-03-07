import time
from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import default_backend, vmap
from jax import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import cgpax
from cgpax.classification_evaluation import evaluate_cgp_genome, evaluate_lgp_genome
from cgpax.functions import function_set_numeric
from cgpax.gomea.fos import compute_fos
from cgpax.gomea.gom import parallel_gom
from cgpax.run_utils import compute_masks, compute_genome_transformation_function, process_dictionary, \
    update_config_with_data, \
    load_dataset
from cgpax.standard import individual


def run_gomea_classification(config: Dict) -> None:
    rnd_key = random.PRNGKey(config["seed"])

    x_values, y_values = load_dataset(config["problem"])

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

    # preliminary evo steps
    genome_mask, mutation_mask = compute_masks(config)
    genome_transformation_function = compute_genome_transformation_function(config)
    genome_evaluation_function = evaluate_cgp_genome if config["solver"] == "cgp" else evaluate_lgp_genome
    genome_to_fitness = partial(genome_evaluation_function, config=config, x_values=x_values, y_values=y_values)

    def genomes_to_fitnesses(genotypes: jnp.ndarray, fake_rnd_keys: jnp.ndarray = None) -> float:
        return vmap(genome_to_fitness)(genotypes)["accuracy"]

    def genomes_to_test_accuracy(genotype: jnp.ndarray, fake_rnd_key: random.PRNGKey = None) -> float:
        return genome_evaluation_function(genotype, config=config, x_values=x_test, y_values=y_test)["accuracy"]

    rnd_key, genome_key = random.split(rnd_key, 2)
    genomes = individual.generate_population(pop_size=config["n_individuals"],
                                             genome_mask=genome_mask, rnd_key=genome_key,
                                             genome_transformation_function=genome_transformation_function)

    n_inner_iterations = 2 * len(genomes[0]) - 2
    print(f"N GOMEA ITERATIONS: {n_inner_iterations}")
    bias_matrix = None  # init needed for gom

    # evaluate population
    _generation = 0

    start_eval_time = time.time()
    fitnesses = genomes_to_fitnesses(genomes)
    eval_time = time.time() - start_eval_time
    print(
        f"{_generation} \t"
        f"FITNESS: {jnp.max(fitnesses)} \t "
        f"E: {eval_time:.2f} \t"
    )
    with open(f"results/{cfg['run_name']}.csv", "w") as csv_file:
        csv_file.write("iteration,fitness,test_accuracy,time\n")
        csv_file.write(f"0,{jnp.max(fitnesses)},{eval_time:.2f}\n")

    times = {}
    # evolutionary loop
    while _generation < config["n_generations"]:
        # fos computation
        fos_start_time = time.process_time()
        rnd_key, fos_key = random.split(rnd_key, 2)
        fos, bias_matrix = compute_fos(genomes, rnd_key, config, bias_matrix, ignore_full_list=True)
        times["fos_time"] = time.process_time() - fos_start_time
        print("FOS DONE")

        # each gomea round has this many iterations within it
        gom_start_time = time.process_time()
        genomes, fitnesses, fitnesses_history, test_accuracies_history = parallel_gom(genomes, fitnesses, fos,
                                                                                      genomes_to_fitnesses, rnd_key,
                                                                                      track_fitnesses=True,
                                                                                      intermediate_prints=True,
                                                                                      test_eval_fn=genomes_to_test_accuracy)
        times["gom_time"] = time.process_time() - gom_start_time
        avg_gom_time = times["gom_time"] / n_inner_iterations

        with open(f"results/{cfg['run_name']}.csv", "a") as csv_file:
            for fit_idx, fit_hist in enumerate(fitnesses_history):
                csv_file.write(
                    f"{_generation + fit_idx},{fit_hist},{test_accuracies_history[fit_idx]},{avg_gom_time:.2f}\n"
                )

        _generation += n_inner_iterations

        # print progress
        print(
            f"{_generation} \t"
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

    config_files = ["configs/graph_gp_gomea_classification.yaml"]
    unpacked_configs = []

    for config_file in config_files:
        unpacked_configs += process_dictionary(cgpax.get_config(config_file))

    print(f"Total configs found: {len(unpacked_configs)}")
    for count, cfg in enumerate(unpacked_configs):
        cfg["run_name"] = f"gomea_{cfg['solver']}_{cfg['problem']}_{cfg['seed']}"
        print(cfg["run_name"])
        run_gomea_classification(cfg)
        print()
