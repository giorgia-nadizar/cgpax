import time
from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import jit, default_backend
from jax import random

import cgpax
from cgpax.gomea.fos import compute_fos
from cgpax.gomea.gom import parallel_gom
from cgpax.run_utils import update_config_with_env_data, init_environment_from_config, compute_masks, \
    compile_genome_evaluation, compute_genome_transformation_function, process_dictionary
from cgpax.standard import individual


def run_gomea_discrete_control(config: Dict) -> None:
    rnd_key = random.PRNGKey(config["seed"])

    environment = init_environment_from_config(config)
    update_config_with_env_data(config, environment)

    # preliminary evo steps
    genome_mask, mutation_mask = compute_masks(config)
    genome_transformation_function = compute_genome_transformation_function(config)

    # compilation of functions
    evaluate_genomes = compile_genome_evaluation(config, environment, config["problem"]["episode_length"])
    replace_invalid_nan_reward = jit(partial(jnp.nan_to_num, nan=config["nan_replacement"]))

    # compose genome eval
    def genomes_to_fitnesses(gs: jnp.ndarray, rnd_keys: jnp.ndarray) -> jnp.ndarray:
        evaluation_outcomes = evaluate_genomes(gs, jnp.array(rnd_keys))
        cumulative_rewards = evaluation_outcomes
        return replace_invalid_nan_reward(cumulative_rewards)


    rnd_key, genome_key = random.split(rnd_key, 2)
    genomes = individual.generate_population(pop_size=config["n_individuals"],
                                             genome_mask=genome_mask, rnd_key=genome_key,
                                             genome_transformation_function=genome_transformation_function)
    bias_matrix = None  # init needed for gom

    # evaluate population
    _generation = 0
    rnd_key, *eval_keys = random.split(rnd_key, len(genomes) + 1)
    start_eval_time = time.time()
    fitnesses = genomes_to_fitnesses(genomes, jnp.array(eval_keys))
    eval_time = time.time() - start_eval_time
    print(
        f"{_generation} \t"
        f"FITNESS: {jnp.max(fitnesses)} \t "
        f"E: {eval_time:.2f} \t"
    )
    with open(f"results/{config['run_name']}.csv", "a") as csv_file:
        csv_file.write("iteration,fitness,time\n")
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
        genomes, fitnesses, fitnesses_history = parallel_gom(genomes, fitnesses, fos, genomes_to_fitnesses, rnd_key,
                                                             track_fitnesses=True, intermediate_prints=True)
        times["gom_time"] = time.process_time() - gom_start_time
        avg_gom_time = times["gom_time"] / len(fos)

        with open(f"results/{config['run_name']}.csv", "a") as csv_file:
            for fit_idx, fit_hist in enumerate(fitnesses_history):
                csv_file.write(f"{_generation + fit_idx},{fit_hist},{avg_gom_time:.2f}\n")

        _generation += len(fos)

        # print progress
        print(
            f"{_generation} \t"
            f"F: {times['fos_time']:.2f} \t"
            f"G: {times['gom_time']:.2f} \t"
            f"FITNESS: {jnp.max(fitnesses)}"
        )




if __name__ == '__main__':

    print(f"Starting the run with {default_backend()} as backend...")

    config_files = ["configs/graph_gp_gomea_discrete_control.yaml"]
    unpacked_configs = []

    for config_file in config_files:
        unpacked_configs += process_dictionary(cgpax.get_config(config_file))

    print(f"Total configs found: {len(unpacked_configs)}")
    for count, cfg in enumerate(unpacked_configs):
        env_name = cfg['problem']['environment'].lower().split("-")[0]
        cfg["run_name"] = f"gomea_{cfg['solver']}_{env_name}_{cfg['seed']}"
        print(cfg["run_name"])
        run_gomea_discrete_control(cfg)
        print()
