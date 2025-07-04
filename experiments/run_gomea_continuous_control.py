import time
from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import jit, default_backend
from jax import random

import cgpax
from cgpax.evaluation.evaluation_utils import prepare_evaluation_functions_continuous_control
from cgpax.gomea.fos import compute_fos
from cgpax.gomea.gom import parallel_gom, parallel_forced_improvement
from cgpax.run_utils import update_config_with_env_data, init_environment_from_config, compute_masks, \
    compile_genome_evaluation, compute_genome_transformation_function, process_dictionary
from cgpax.standard import individual


def run_gomea_continuous_control(config: Dict) -> None:
    if "n_evaluations" not in config:
        config["n_evaluations"] = config["n_generations"] * config["n_individuals"]
    forced_improvement = config.get("forced_improvement", False)
    forced_improvement_generations_threshold = 1 + jnp.log10(config["n_individuals"])

    rnd_key = random.PRNGKey(config["seed"])

    # compose genome eval
    genomes_to_fitnesses, _ = prepare_evaluation_functions_continuous_control(config)
    genome_mask, mutation_mask = compute_masks(config)
    genome_transformation_function = compute_genome_transformation_function(config)

    rnd_key, genome_key = random.split(rnd_key, 2)
    genomes = individual.generate_population(pop_size=config["n_individuals"],
                                             genome_mask=genome_mask, rnd_key=genome_key,
                                             genome_transformation_function=genome_transformation_function)
    bias_matrix = None  # init needed for gom

    # evaluate population
    _fitness_evaluation = 0
    rnd_key, *eval_keys = random.split(rnd_key, len(genomes) + 1)
    start_eval_time = time.time()
    fitnesses = genomes_to_fitnesses(genomes, jnp.array(eval_keys))
    eval_time = time.time() - start_eval_time
    print(
        f"{_fitness_evaluation} \t"
        f"FITNESS: {jnp.max(fitnesses)} \t "
        f"E: {eval_time:.2f} \t"
    )
    with open(f"../results/{config['run_name']}.csv", "a") as csv_file:
        csv_file.write("evaluation,fitness,time\n")
        csv_file.write(f"0,{jnp.max(fitnesses)},{eval_time:.2f}\n")

    times = {}
    # evolutionary loop
    elite_fitness = -jnp.inf
    elite_individual = None
    no_fitness_improvement_generations = 0
    while _fitness_evaluation < config["n_evaluations"]:
        # fos computation
        fos_start_time = time.process_time()
        rnd_key, fos_key = random.split(rnd_key, 2)
        fos, bias_matrix = compute_fos(genomes, rnd_key, config, bias_matrix, ignore_full_list=True)
        times["fos_time"] = time.process_time() - fos_start_time
        print("FOS DONE")

        rnd_key, gom_key = random.split(rnd_key, 2)
        gom_start_time = time.process_time()
        offspring_genomes, fitnesses, fitnesses_history = parallel_gom(genomes, fitnesses, fos, genomes_to_fitnesses,
                                                                       gom_key, track_fitnesses=True,
                                                                       intermediate_prints=True)
        times["gom_time"] = time.process_time() - gom_start_time
        avg_gom_time = times["gom_time"] / len(fos)

        with open(f"../results/{config['run_name']}.csv", "a") as csv_file:
            for details_dict in fitnesses_history:
                csv_file.write(
                    f"{_fitness_evaluation + details_dict['evaluation']},{details_dict['max_fitness']},{avg_gom_time:.2f}\n"
                )

        _fitness_evaluation += len(fos) * len(offspring_genomes)

        # print progress
        print(
            f"{_fitness_evaluation} \t"
            f"F: {times['fos_time']:.2f} \t"
            f"G: {times['gom_time']:.2f} \t"
            f"FITNESS: {jnp.max(fitnesses)}"
        )

        if jnp.max(fitnesses) == elite_fitness:
            no_fitness_improvement_generations += 1
        else:
            elite_fitness = jnp.max(fitnesses)
            elite_individual = offspring_genomes[jnp.argmax(fitnesses)]

        rnd_key, forced_impro_key = random.split(rnd_key, 2)
        if forced_improvement:
            # no improvements stretch
            if no_fitness_improvement_generations >= forced_improvement_generations_threshold:
                print("global forced improvement due to no improvements stretch")
                genomes, fitnesses, history_dicts, evals_done = parallel_forced_improvement(
                    offspring_genomes,
                    fitnesses,
                    elite_individual,
                    elite_fitness,
                    fos,
                    genomes_to_fitnesses,
                    forced_impro_key,
                    True
                )
            else:
                unchanged_genomes_ids = jnp.where(jnp.all(genomes == offspring_genomes, axis=1))[0]
                if len(unchanged_genomes_ids) == 0:
                    continue
                print("forced improvement on not changed genomes")
                unchanged_genomes = genomes[unchanged_genomes_ids]
                unchanged_fitnesses = fitnesses[unchanged_genomes_ids]
                forced_improved_genomes, forced_improved_fitnesses, history_dicts, evals_done = parallel_forced_improvement(
                    unchanged_genomes,
                    unchanged_fitnesses,
                    elite_individual,
                    elite_fitness,
                    fos,
                    genomes_to_fitnesses,
                    forced_impro_key,
                    True
                )
                genomes = offspring_genomes
                genomes = genomes.at[unchanged_genomes_ids].set(forced_improved_genomes)
                fitnesses = fitnesses.at[unchanged_genomes_ids].set(forced_improved_fitnesses)

            _fitness_evaluation += evals_done

            if jnp.max(fitnesses) > elite_fitness:
                elite_fitness = jnp.max(fitnesses)
                elite_individual = offspring_genomes[jnp.argmax(fitnesses)]
                # if forced improvement gave better results, keep track of them
                with open(f"../results/{config['run_name']}.csv", "a") as csv_file:
                    csv_file.write(
                        f"{_fitness_evaluation},{elite_fitness},{avg_gom_time:.2f}\n"
                    )


if __name__ == '__main__':

    print(f"Starting the run with {default_backend()} as backend...")

    config_files = ["../configs/graph_gp_gomea_continuous_control.yaml"]
    unpacked_configs = []

    for config_file in config_files:
        unpacked_configs += process_dictionary(cgpax.get_config(config_file))

    print(f"Total configs found: {len(unpacked_configs)}")
    for count, cfg in enumerate(unpacked_configs):
        env_name = cfg['problem']['environment'].lower().split("-")[0]
        cfg["run_name"] = f"gomea_{cfg['solver']}_{env_name}_{cfg['seed']}"
        print(cfg["run_name"])
        run_gomea_continuous_control(cfg)
        print()
