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

from cgpax.jax_individual import generate_population

from cgpax.run_utils import __update_config_with_env_data__, __compile_parents_selection__, __compile_mutation__, \
    __init_environment_from_config__, __compute_parallel_runs_indexes__, __init_environments__, __compute_masks__, \
    __compile_genome_evaluation__, __init_tracking__, __update_tracking__, __compute_genome_transformation_function__, \
    __compile_survival_selection__, __compute_novelty_scores__, __normalize_array__, __compile_crossover__, \
    __config_to_run_name__, __compute_weights_mutation_function__, __notify_update__, __process_dictionary__


def run(config: Dict, wandb_run: Run) -> None:
    rnd_key = random.PRNGKey(config["seed"])

    novelty_archive, alpha = None, 0
    if config.get("novelty") is not None:
        novelty_archive = {0}
        alpha = config["novelty"].get("alpha", 0.5)

    if config.get("n_parallel_runs", 1) > 1:
        runs_indexes = __compute_parallel_runs_indexes__(config["n_individuals"], config["n_parallel_runs"])
        config["runs_indexes"] = runs_indexes

    # incremental episode duration
    if config["problem"]["incremental_steps"] > 1:
        environments = __init_environments__(config)
        start_gens = [e["start_gen"] for e in environments]
        env_dict = environments[0]
        environment = env_dict["env"]
        fitness_scaler = env_dict["fitness_scaler"]
    else:
        environments, start_gens = None, None
        environment = __init_environment_from_config__(config)
        fitness_scaler = 1.0
    __update_config_with_env_data__(config, environment)
    wandb.config.update(config, allow_val_change=True)

    # preliminary evo steps
    genome_mask, mutation_mask = __compute_masks__(config)
    weights_mutation_function = __compute_weights_mutation_function__(config)
    genome_transformation_function = __compute_genome_transformation_function__(config)

    # compilation of functions
    evaluate_genomes = __compile_genome_evaluation__(config, environment, config["problem"]["episode_length"])
    select_parents = __compile_parents_selection__(config)
    crossover_genomes = __compile_crossover__(config)
    mutate_genomes = __compile_mutation__(config, genome_mask, mutation_mask, weights_mutation_function,
                                          genome_transformation_function)
    replace_invalid_nan_reward = jit(partial(jnp.nan_to_num, nan=config["nan_replacement"]))
    replace_invalid_nan_zero = jit(partial(jnp.nan_to_num, nan=0))
    select_survivals = __compile_survival_selection__(config)

    # initialize tracking
    store_fitness_details = config.get("store_fitness_details", [])
    store_fitness_details = store_fitness_details if isinstance(store_fitness_details, list) else [
        store_fitness_details]
    tracking_objects = __init_tracking__(config, store_fitness_details=store_fitness_details,
                                         saving_interval=config["saving_interval"])

    rnd_key, genome_key = random.split(rnd_key, 2)
    genomes = generate_population(pop_size=config["n_individuals"] * config.get("n_parallel_runs", 1),
                                  genome_mask=genome_mask, rnd_key=genome_key,
                                  weights_mutation_function=weights_mutation_function,
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
            evaluate_genomes = __compile_genome_evaluation__(config, environment, env_dict["duration"])

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
            novelty_values = __compute_novelty_scores__(evaluation_outcomes["feet_contact_proportion"], novelty_archive)
            novelty_values = replace_invalid_nan_zero(novelty_values)
            normalized_reward = __normalize_array__(reward_values)
            normalized_novelty = __normalize_array__(novelty_values)
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

        tracking_objects = __update_tracking__(
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
    assert default_backend() == "gpu"

    telegram_config = cgpax.get_config("telegram/token.yaml")
    telegram_bot = telegram.Bot(telegram_config["token"])

    api = wandb.Api(timeout=40)
    entity, project = "giorgianadizar", "cgpax"
    existing_run_names = [r.name for r in api.runs(entity + "/" + project) if r.state == "finished"]

    config_files = ["configs/miniant.yaml"]
    unpacked_configs = []

    for config_file in config_files:
        unpacked_configs += __process_dictionary__(cgpax.get_config(config_file))

    __notify_update__(f"Total configs found: {len(unpacked_configs)}", telegram_bot, telegram_config["chat_id"])
    for count, cfg in enumerate(unpacked_configs):
        run_name, _, _, _, _, _ = __config_to_run_name__(cfg)
        if run_name in existing_run_names:
            __notify_update__(f"{count + 1}/{len(unpacked_configs)} - {run_name} already exists")
            continue
        __notify_update__(f"{count + 1}/{len(unpacked_configs)} - {run_name} starting\n{cfg}", telegram_bot,
                          telegram_config["chat_id"])
        wb_run = wandb.init(config=cfg, project=project, name=run_name)
        run(cfg, wb_run)
        wb_run.finish()
        print()
