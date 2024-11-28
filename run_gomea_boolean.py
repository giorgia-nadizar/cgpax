import time
from os import write
from typing import Dict

from wandb.sdk.wandb_run import Run

import cgpax
import wandb
import telegram
from jax import jit, default_backend, vmap
import jax.numpy as jnp
from jax import random

from functools import partial

from cgpax.boolean_evaluation import evaluate_cgp_genome, evaluate_lgp_genome
from cgpax.functions import function_set_control, function_set_boolean
from cgpax.gomea.fos import compute_normalized_mutual_information_matrix, compute_fos
from cgpax.gomea.gom import parallel_gom
from cgpax.standard import individual

from cgpax.run_utils import update_config_with_env_data, compile_parents_selection, compile_mutation, \
    init_environment_from_config, compute_masks, compile_genome_evaluation, init_tracking, update_tracking, \
    compute_genome_transformation_function, compile_survival_selection, compile_crossover, \
    config_to_run_name, compute_weights_mutation_function, notify_update, process_dictionary, update_config_with_data, \
    load_dataset


def run(config: Dict, wandb_run: Run) -> None:
    rnd_key = random.PRNGKey(config["seed"])

    x_values, y_values = load_dataset(config["problem"])

    # assert this
    config["use_input_constants"] = False

    update_config_with_data(config, x_values.shape[1], y_values.shape[1], function_set=function_set_boolean)
    # wandb.config.update(config, allow_val_change=True)

    # preliminary evo steps
    genome_mask, mutation_mask = compute_masks(config)
    genome_transformation_function = compute_genome_transformation_function(config)
    genome_evaluation_function = evaluate_cgp_genome if config["solver"] == "cgp" else evaluate_lgp_genome
    genome_to_fitness = partial(genome_evaluation_function, config=config, x_values=x_values, y_values=y_values)

    def genomes_to_fitnesses(genotypes: jnp.ndarray, fake_rnd_keys: jnp.ndarray = None) -> float:
        return vmap(genome_to_fitness)(genotypes)["accuracy"]

    # compilation of functions
    replace_invalid_nan_reward = jit(partial(jnp.nan_to_num, nan=config["nan_replacement"]))

    # compose genome eval
    # def genomes_to_fitnesses(gs: jnp.ndarray, rnd_keys: jnp.ndarray) -> jnp.ndarray:
    #     evaluation_outcomes = evaluate_genomes(gs, jnp.array(rnd_keys))
    #     return replace_invalid_nan_reward(evaluation_outcomes["cum_reward"])

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
    with open(f"results/{cfg['run_name']}.csv", "a") as csv_file:
        csv_file.write("iteration,fitness,time\n")
        csv_file.write(f"0,{jnp.max(fitnesses)},{eval_time:.2f}\n")

    times = {}
    # evolutionary loop
    while _generation < config["n_generations"]:
        # fos computation
        fos_start_time = time.process_time()
        nmi_matrix, bias_matrix = compute_normalized_mutual_information_matrix(genomes, config,
                                                                               bias_matrix)  # (genotype size, genotype size)
        print("NMI DONE")
        rnd_key, fos_key = random.split(rnd_key, 2)
        fos = compute_fos(nmi_matrix, fos_key, ignore_full_list=True)  # 2 * genotype size - 2
        times["fos_time"] = time.process_time() - fos_start_time
        print("FOS DONE")

        # each gomea round has this many iterations within it
        gom_start_time = time.process_time()
        genomes, fitnesses, fitnesses_history = parallel_gom(genomes, fitnesses, fos, genomes_to_fitnesses, rnd_key,
                                                             track_fitnesses=True, intermediate_prints=True)
        times["gom_time"] = time.process_time() - gom_start_time
        avg_gom_time = times["gom_time"] / n_inner_iterations

        with open(f"results/{cfg['run_name']}.csv", "a") as csv_file:
            for fit_idx, fit_hist in enumerate(fitnesses_history):
                csv_file.write(f"{_generation + fit_idx},{fit_hist},{avg_gom_time:.2f}\n")

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

        # tracking_objects = update_tracking(
        #     config=config,
        #     tracking_objects=tracking_objects,
        #     genomes=genomes,
        #     fitness_values=fitnesses,
        #     rewards=fitnesses,
        #     detailed_rewards=None,
        #     times=times,
        #     wdb_run=wandb_run
        # )


if __name__ == '__main__':

    print(f"Starting the run with {default_backend()} as backend...")

    # telegram_config = cgpax.get_config("telegram/token.yaml")
    # telegram_bot = telegram.Bot(telegram_config["token"])

    # api = wandb.Api(timeout=40)
    entity, project = "giorgianadizar", "cgpax"
    # existing_run_names = [r.name for r in api.runs(entity + "/" + project) if r.state == "finished"]

    config_files = ["configs/graph_gp_gomea_boolean.yaml"]
    unpacked_configs = []

    for config_file in config_files:
        unpacked_configs += process_dictionary(cgpax.get_config(config_file))

    # notify_update(f"Total configs found: {len(unpacked_configs)}", telegram_bot, telegram_config["chat_id"])
    print(f"Total configs found: {len(unpacked_configs)}")
    for count, cfg in enumerate(unpacked_configs):
        # run_name, _, _, _, _, _ = config_to_run_name(cfg)
        cfg["run_name"] = f"gomea_{cfg['solver']}_{cfg['problem']}_{cfg['seed']}"
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
