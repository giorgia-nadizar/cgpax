import time
from typing import Dict

from wandb.sdk.wandb_run import Run

import cgpax
import wandb
import telegram
from jax import jit, default_backend
import jax.numpy as jnp
from jax import random

from functools import partial

from cgpax.standard import individual
from cgpax.weighted import individual_weighted

from cgpax.run_utils import update_config_with_env_data, compile_parents_selection, compile_mutation, \
    init_environment_from_config, compute_parallel_runs_indexes, init_environments, compute_masks, \
    compile_genome_evaluation, init_tracking, update_tracking, compute_genome_transformation_function, \
    compile_survival_selection, compute_novelty_scores, normalize_array, compile_crossover, \
    config_to_run_name, compute_weights_mutation_function, notify_update, process_dictionary


def run(config: Dict, wandb_run: Run) -> None:
    rnd_key = random.PRNGKey(config["seed"])

    novelty_archive, alpha = None, 0
    if config.get("novelty") is not None:
        novelty_archive = {0}
        alpha = config["novelty"].get("alpha", 0.5)

    if config.get("n_parallel_runs", 1) > 1:
        runs_indexes = compute_parallel_runs_indexes(config["n_individuals"], config["n_parallel_runs"])
        config["runs_indexes"] = runs_indexes

    # incremental episode duration
    if config["problem"]["incremental_steps"] > 1:
        environments = init_environments(config)
        start_gens = [e["start_gen"] for e in environments]
        env_dict = environments[0]
        environment = env_dict["env"]
        fitness_scaler = env_dict["fitness_scaler"]
    else:
        environments, start_gens = None, None
        environment = init_environment_from_config(config)
        fitness_scaler = 1.0
    update_config_with_env_data(config, environment)
    wandb.config.update(config, allow_val_change=True)

    # preliminary evo steps
    genome_mask, mutation_mask = compute_masks(config)
    weights_mutation_function = compute_weights_mutation_function(config)
    genome_transformation_function = compute_genome_transformation_function(config)

    # compilation of functions
    evaluate_genomes = compile_genome_evaluation(config, environment, config["problem"]["episode_length"])
    select_parents = compile_parents_selection(config)
    crossover_genomes = compile_crossover(config)
    mutate_genomes = compile_mutation(config, genome_mask, mutation_mask, genome_transformation_function)
    replace_invalid_nan_reward = jit(partial(jnp.nan_to_num, nan=config["nan_replacement"]))
    replace_invalid_nan_zero = jit(partial(jnp.nan_to_num, nan=0))
    select_survivals = compile_survival_selection(config)

    # initialize tracking
    store_fitness_details = config.get("store_fitness_details", [])
    store_fitness_details = store_fitness_details if isinstance(store_fitness_details, list) else [
        store_fitness_details]
    tracking_objects = init_tracking(config, store_fitness_details=store_fitness_details,
                                     saving_interval=config["saving_interval"])

    rnd_key, genome_key = random.split(rnd_key, 2)
    if config.get("weighted_connections"):
        genomes = individual_weighted.generate_population(
            pop_size=config["n_individuals"] * config.get("n_parallel_runs", 1),
            genome_mask=genome_mask, rnd_key=genome_key,
            weights_mutation_function=weights_mutation_function,
            genome_transformation_function=genome_transformation_function)
    else:
        genomes = individual.generate_population(pop_size=config["n_individuals"] * config.get("n_parallel_runs", 1),
                                                 genome_mask=genome_mask, rnd_key=genome_key,
                                                 genome_transformation_function=genome_transformation_function)

    times = {}
    # evolutionary loop
    for _generation in range(config["n_generations"]):
        # check if env needs update
        if config["problem"]["incremental_steps"] > 1 and _generation in start_gens:
            env_idx = start_gens.index(_generation)
            env_dict = environments[env_idx]
            environment = env_dict["env"]
            fitness_scaler = env_dict["fitness_scaler"]
            evaluate_genomes = compile_genome_evaluation(config, environment, env_dict["duration"])

        # evaluate population
        rnd_key, *eval_keys = random.split(rnd_key, len(genomes) + 1)
        start_eval = time.process_time()
        evaluation_outcomes = evaluate_genomes(genomes, jnp.array(eval_keys))
        reward_values = replace_invalid_nan_reward(evaluation_outcomes["cum_reward"]) * fitness_scaler
        detailed_rewards = {
            "healthy": evaluation_outcomes["cum_healthy_reward"],
            "ctrl": evaluation_outcomes["cum_ctrl_reward"],
            "forward": evaluation_outcomes["cum_forward_reward"]
        }

        if novelty_archive is not None:
            novelty_values = compute_novelty_scores(evaluation_outcomes["feet_contact_proportion"], novelty_archive)
            novelty_values = replace_invalid_nan_zero(novelty_values)
            normalized_reward = normalize_array(reward_values)
            normalized_novelty = normalize_array(novelty_values)
            fitness_values = alpha * normalized_reward + (1 - alpha) * normalized_novelty
        elif config.get("distance", False):
            distances = evaluation_outcomes["x_distance"]
            fitness_values = replace_invalid_nan_zero(distances)
        elif config.get("weighted_rewards", None) is not None:
            weights = config["weighted_rewards"]
            healthy_w, ctrl_w, forward_w = weights["healthy"], weights["ctrl"], weights["forward"]
            incr_weight = _generation / config["n_generations"]
            decr_weight = 1 - incr_weight
            healthy_w = incr_weight if healthy_w == "i" else decr_weight if healthy_w == "d" else healthy_w
            ctrl_w = incr_weight if ctrl_w == "i" else decr_weight if ctrl_w == "d" else ctrl_w
            forward_w = incr_weight if forward_w == "i" else decr_weight if forward_w == "d" else forward_w
            fitness_values = healthy_w * detailed_rewards["healthy"] + ctrl_w * detailed_rewards["ctrl"] \
                             + forward_w * detailed_rewards["forward"]
            fitness_values = replace_invalid_nan_reward(fitness_values)
        else:
            fitness_values = reward_values
        end_eval = time.process_time()
        times["evaluation_time"] = end_eval - start_eval

        # if multiple evals, need median
        if config["n_evals_per_individual"] > 1:
            fitness_values = jnp.median(fitness_values, axis=1)
            # TODO should extract the id of the median
            reward_values = jnp.mean(reward_values, axis=1)
            for rew in detailed_rewards:
                detailed_rewards[rew] = jnp.mean(detailed_rewards[rew], axis=1)

        # select parents
        rnd_key, select_key = random.split(rnd_key, 2)
        start_selection = time.process_time()
        parents = select_parents(genomes, fitness_values, select_key)
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

        # print progress
        print(
            f"{_generation} \t"
            f"E: {times['evaluation_time']:.2f} \t"
            f"S: {times['selection_time']:.2f} \t"
            f"M: {times['mutation_time']:.2f} \t"
            f"FITNESS: {jnp.max(fitness_values)}"
        )

        tracking_objects = update_tracking(
            config=config,
            tracking_objects=tracking_objects,
            genomes=genomes,
            fitness_values=fitness_values,
            rewards=reward_values,
            detailed_rewards=detailed_rewards,
            times=times,
            wdb_run=wandb_run
        )

        # select survivals
        rnd_key, survival_key = random.split(rnd_key, 2)
        survivals = parents if select_survivals is None else select_survivals(genomes, fitness_values, survival_key)

        # update population
        assert len(genomes) == len(survivals) + len(offspring)
        genomes = jnp.concatenate((survivals, offspring))


if __name__ == '__main__':

    print(f"Starting the run with {default_backend()} as backend...")

    telegram_config = cgpax.get_config("telegram/token.yaml")
    telegram_bot = telegram.Bot(telegram_config["token"])

    api = wandb.Api(timeout=40)
    entity, project = "giorgianadizar", "cgpax"
    existing_run_names = [r.name for r in api.runs(entity + "/" + project) if r.state == "finished"]

    config_files = ["configs/mini_graph_gp.yaml"]
    unpacked_configs = []

    for config_file in config_files:
        unpacked_configs += process_dictionary(cgpax.get_config(config_file))

    notify_update(f"Total configs found: {len(unpacked_configs)}", telegram_bot, telegram_config["chat_id"])
    for count, cfg in enumerate(unpacked_configs):
        run_name, _, _, _, _, _ = config_to_run_name(cfg)
        if run_name in existing_run_names:
            notify_update(f"{count + 1}/{len(unpacked_configs)} - {run_name} already exists")
            continue
        notify_update(f"{count + 1}/{len(unpacked_configs)} - {run_name} starting\n{cfg}", telegram_bot,
                      telegram_config["chat_id"])
        wb_run = wandb.init(config=cfg, project=project, name=run_name)
        run(cfg, wb_run)
        wb_run.finish()
        print()
