import asyncio
import copy
import functools
from datetime import datetime
from functools import partial, reduce
from typing import List, Callable, Tuple, Dict, Union, Set

import telegram
from brax import envs
from brax.envs import ant
from brax.envs.wrappers import EpisodeWrapper
from scipy.cluster.hierarchy import weighted
from wandb.apis.public import Run

from jax import vmap, jit, random
import jax.numpy as jnp

from cgpax.control_evaluation import evaluate_cgp_genome, evaluate_cgp_genome_n_times, evaluate_lgp_genome, \
    evaluate_lgp_genome_n_times
from cgpax.functions import function_set_control, constants
from cgpax.selection import truncation_selection, tournament_selection, fp_selection, composed_selection
from cgpax.tracker import Tracker
from cgpax.standard import individual
from cgpax.standard.individual import levels_back_transformation_function
from cgpax.utils import identity
from cgpax.weighted import encoding_weighted, individual_weighted


def init_environment(env_name: str, episode_length: int, terminate_when_unhealthy: bool = True) -> EpisodeWrapper:
    if env_name == "miniant":
        env = functools.partial(ant.Ant, use_contact_forces=False)(terminate_when_unhealthy=terminate_when_unhealthy)
    else:
        try:
            env = envs.get_environment(env_name=env_name, terminate_when_unhealthy=terminate_when_unhealthy)
        except TypeError:
            env = envs.get_environment(env_name=env_name)
    env = EpisodeWrapper(env, episode_length=episode_length, action_repeat=1)
    return env


def init_environment_from_config(config: Dict) -> EpisodeWrapper:
    return init_environment(config["problem"]["environment"], config["problem"]["episode_length"],
                            config.get("unhealthy_termination", True))


def init_environments(config: Dict) -> List[Dict]:
    n_steps = config["problem"]["incremental_steps"]
    min_duration = config["problem"]["min_length"]
    step_size = (config["problem"]["episode_length"] - min_duration) / (n_steps - 1)
    gen_step_size = int(config["n_generations"] / n_steps)
    return [
        {
            "start_gen": gen_step_size * n,
            "env": init_environment(env_name=config["problem"]["environment"],
                                    episode_length=(min_duration + int(step_size * n))
                                    ),
            "fitness_scaler": config["problem"]["episode_length"] / (min_duration + int(step_size * n)),
            "duration": (min_duration + int(step_size * n))
        }
        for n in range(n_steps)
    ]


def update_config_with_data(config: Dict, input_space_size: int, output_space_size: int,
                            function_set: Dict = function_set_control) -> None:
    """Updates the config dictionary based on the provided values."""
    config["n_functions"] = len(function_set)
    config["n_constants"] = len(constants) if config.get("use_input_constants", True) else 0

    config["n_in_env"] = input_space_size
    config["n_in"] = config["n_in_env"] + config["n_constants"]
    config["n_out"] = output_space_size
    weighted_connections = config.get("weighted_connections", False)

    if config["solver"] == "cgp":
        config["buffer_size"] = config["n_in"] + config["n_nodes"]
        config["genome_size"] = config["n_nodes"] * (4 if weighted_connections else 3) + config["n_out"]
        levels_back = config.get("levels_back")
        if levels_back is not None and levels_back < config["n_in"]:
            config["levels_back"] = config["n_in"]
    else:
        config["n_registers"] = config["n_in"] + config["n_extra_registers"] + config["n_out"]
        config["genome_size"] = config["n_rows"] * (5 if weighted_connections else 4)


def update_config_with_env_data(config: Dict, env) -> None:
    update_config_with_data(config, env.observation_size, env.action_size)


def load_dataset(problem_name: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x_values = jnp.load(f"datasets/{problem_name}_x.npy")
    y_values = jnp.load(f"datasets/{problem_name}_y.npy")
    return x_values, y_values


def compute_parallel_runs_indexes(n_individuals: int, n_parallel_runs: int, n_elites: int = 1) -> jnp.ndarray:
    indexes = jnp.zeros((n_parallel_runs, n_individuals))
    for run_idx in range(n_parallel_runs):
        for elite_idx in range(n_elites):
            indexes = indexes.at[run_idx, elite_idx].set(run_idx * n_elites + elite_idx)
        for ind_idx in range(n_individuals - n_elites):
            indexes = indexes.at[run_idx, ind_idx + n_elites].set(
                n_elites * n_parallel_runs + ind_idx + (n_individuals - n_elites) * run_idx)
    return indexes.astype(int)


def compile_genome_evaluation(config: Dict, env: EpisodeWrapper, episode_length: int) -> Callable:
    if config["solver"] == "cgp":
        eval_func, eval_n_times_func = evaluate_cgp_genome, evaluate_cgp_genome_n_times
        w_encoding_func = encoding_weighted.genome_to_cgp_program
    else:
        eval_func, eval_n_times_func = evaluate_lgp_genome, evaluate_lgp_genome_n_times
        w_encoding_func = encoding_weighted.genome_to_lgp_program
    if config["n_evals_per_individual"] == 1:
        partial_eval_genome = partial(eval_func, config=config, env=env, episode_length=episode_length)
    else:
        partial_eval_genome = partial(eval_n_times_func, config=config, env=env,
                                      n_times=config["n_evals_per_individual"], episode_length=episode_length)

    if config.get("weighted_connections", False):
        partial_eval_genome = partial(partial_eval_genome, genome_encoder=w_encoding_func)

    vmap_evaluate_genome = vmap(partial_eval_genome, in_axes=(0, 0))
    return jit(vmap_evaluate_genome)


def compile_crossover(config: Dict) -> Union[Callable, None]:
    if config.get("crossover", False) and config["solver"] == "lgp":
        xover_func = individual_weighted.lgp_one_point_crossover_genomes \
            if config.get("weighted_connections", False) else individual.lgp_one_point_crossover_genomes
        vmap_crossover = vmap(xover_func, in_axes=(0, 0, 0))
        return jit(vmap_crossover)
    else:
        return None


def compile_mutation(config: Dict, genome_mask: jnp.ndarray, mutation_mask: jnp.ndarray,
                     genome_transformation_function: Callable[[jnp.ndarray], jnp.ndarray],
                     n_mutations_per_individual: int = 1) -> Callable:
    if config.get("weighted_connections", False):
        weights_mutation_function = compute_weights_mutation_function(config)
        if config["mutation"] == "standard":
            partial_multiple_mutations = partial(individual_weighted.mutate_genome_n_times,
                                                 n_mutations=n_mutations_per_individual,
                                                 genome_mask=genome_mask, mutation_mask=mutation_mask,
                                                 weights_mutation_function=weights_mutation_function,
                                                 genome_transformation_function=genome_transformation_function)
        else:
            partial_multiple_mutations = partial(individual_weighted.mutate_genome_n_times_stacked,
                                                 n_mutations=n_mutations_per_individual,
                                                 genome_mask=genome_mask, mutation_mask=mutation_mask,
                                                 weights_mutation_function=weights_mutation_function,
                                                 genome_transformation_function=genome_transformation_function)
    else:
        if config["mutation"] == "standard":
            partial_multiple_mutations = partial(individual.mutate_genome_n_times,
                                                 n_mutations=n_mutations_per_individual,
                                                 genome_mask=genome_mask, mutation_mask=mutation_mask,
                                                 genome_transformation_function=genome_transformation_function)
        else:
            partial_multiple_mutations = partial(individual.mutate_genome_n_times_stacked,
                                                 n_mutations=n_mutations_per_individual,
                                                 genome_mask=genome_mask, mutation_mask=mutation_mask,
                                                 genome_transformation_function=genome_transformation_function)

    vmap_multiple_mutations = vmap(partial_multiple_mutations)
    return jit(vmap_multiple_mutations)


def compile_survival_selection(config: Dict) -> Union[Callable, None]:
    if config["survival"] == "parents":
        return None
    elif config["survival"] == "truncation":
        return jit(partial(truncation_selection, n_elites=config["selection"]["elite_size"]))
    elif config["survival"] == "tournament":
        return jit(partial(tournament_selection, n_elites=config["selection"]["elite_size"],
                           tour_size=config["selection"]["tour_size"]))
    else:
        return jit(partial(fp_selection, n_elites=config["selection"]["elite_size"]))


def compile_parents_selection(config: Dict, n_parents: int = 0) -> Callable:
    if n_parents == 0:
        n_parents = config["n_individuals"] - config["selection"]["elite_size"]
    if config["selection"]["type"] == "truncation":
        partial_selection = partial(truncation_selection, n_elites=n_parents)
    elif config["selection"]["type"] == "tournament":
        partial_selection = partial(tournament_selection, n_elites=n_parents,
                                    tour_size=config["selection"]["tour_size"])
    else:
        partial_selection = partial(fp_selection, n_elites=n_parents)
    inner_selection = jit(partial_selection)
    if config.get("n_parallel_runs", 1) == 1:
        return inner_selection
    else:
        def _composite_selection(genomes, fitness_values, select_key):
            parents_list = []
            for run_idx in config["runs_indexes"]:
                rnd_key, sel_key = random.split(select_key, 2)
                current_parents = composed_selection(genomes, fitness_values, sel_key, run_idx, inner_selection)
                parents_list.append(current_parents)
            parents_matrix = jnp.array(parents_list)
            return jnp.reshape(parents_matrix, (-1, parents_matrix.shape[-1]))

        return _composite_selection


def compute_masks(config: Dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if config.get("weighted_connections", False):
        if config["solver"] == "cgp":
            genome_mask = individual_weighted.compute_cgp_genome_mask(config, config["n_in"], config["n_out"])
            mutation_mask = individual_weighted.compute_cgp_mutation_prob_mask(config, config["n_out"])
        else:
            genome_mask = individual_weighted.compute_lgp_genome_mask(config, config["n_in"])
            mutation_mask = individual_weighted.compute_lgp_mutation_prob_mask(config)
    else:
        if config["solver"] == "cgp":
            genome_mask = individual.compute_cgp_genome_mask(config, config["n_in"], config["n_out"])
            mutation_mask = individual.compute_cgp_mutation_prob_mask(config, config["n_out"])
        else:
            genome_mask = individual.compute_lgp_genome_mask(config, config["n_in"])
            mutation_mask = individual.compute_lgp_mutation_prob_mask(config)
    return genome_mask, mutation_mask


def compute_weights_mutation_function(config: Dict) -> Union[Callable[[random.PRNGKey], jnp.ndarray], None]:
    if not config.get("weighted_connections"):
        return None

    sigma = config.get("weights_sigma", 0.0)
    length = config.get("n_rows", config.get("n_nodes"))

    def _gaussian_function(rnd_key: random.PRNGKey) -> jnp.ndarray:
        return random.normal(key=rnd_key, shape=[length]) * sigma

    return _gaussian_function


def compute_genome_transformation_function(config: Dict) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if config["solver"] == "cgp" and config.get("levels_back") is not None:
        return levels_back_transformation_function(config["n_in"], config["n_nodes"])
    else:
        return identity


def init_tracking(config: Dict, saving_interval: int = 100, store_fitness_details: List[str] = None) -> Tuple:
    if config.get("n_parallel_runs", 1) > 1:
        trackers = [Tracker(config, idx=k, saving_interval=saving_interval, store_fitness_details=store_fitness_details)
                    for k in range(config["n_parallel_runs"])]
        tracker_states = [t.init() for t in trackers]
        return trackers, tracker_states
    else:
        tracker = Tracker(config, idx=config["seed"], saving_interval=saving_interval,
                          store_fitness_details=store_fitness_details)
        tracker_state = tracker.init()
        return tracker, tracker_state


def update_tracking(config: Dict, tracking_objects: Tuple, genomes: jnp.ndarray, fitness_values: jnp.ndarray,
                    rewards: jnp.ndarray, detailed_rewards: Dict, times: Dict, wdb_run: Run) -> Tuple:
    if config.get("n_parallel_runs", 1) == 1:
        tracker, tracker_state = tracking_objects
        tracker_state = tracker.update(
            tracker_state=tracker_state,
            fitness=fitness_values,
            rewards=rewards,
            detailed_rewards=detailed_rewards,
            best_individual=genomes.at[jnp.argmax(fitness_values)].get(),
            times=times
        )
        tracker.wandb_log(tracker_state, wdb_run)
        return tracker, tracker_state
    else:
        trackers, tracker_states = tracking_objects
        for run_idx in range(config["n_parallel_runs"]):
            current_indexes = config["runs_indexes"].at[run_idx, :].get()
            sub_fitness = jnp.take(fitness_values, current_indexes, axis=0)
            sub_rewards = jnp.take(rewards, current_indexes, axis=0)
            sub_genomes = jnp.take(genomes, current_indexes, axis=0)
            tracker_states[run_idx] = trackers[run_idx].update(
                tracker_state=tracker_states[run_idx],
                fitness=sub_fitness,
                rewards=sub_rewards,
                detailed_rewards=detailed_rewards,
                best_individual=sub_genomes.at[jnp.argmax(sub_fitness)].get(),
                times=times
            )
            trackers[run_idx].wandb_log(tracker_states[run_idx], wdb_run)
        return trackers, tracker_states


def normalize_array(array: jnp.ndarray) -> jnp.ndarray:
    min_val = jnp.min(array)
    max_val = jnp.max(array)
    return (array - min_val) / (max_val - min_val)


@jit
def compute_max_distance(x_coord: float, x_pos_archive: jnp.ndarray) -> jnp.ndarray:
    @jit
    def _distance(x1: float, x2: float) -> float:
        return jnp.abs(x1 - x2)

    distances = vmap(_distance, in_axes=(None, 0))(x_coord, x_pos_archive)
    return jnp.max(distances)


def compute_novelty_scores(final_positions: jnp.ndarray, novelty_archive: Set, decimals: int = 2) -> jnp.ndarray:
    x_coordinates = final_positions.at[:, :, 0].get()
    x_coordinates_flat = x_coordinates.flatten()
    archive_array = jnp.asarray(list(novelty_archive))
    max_distances = vmap(compute_max_distance, in_axes=(0, None))(x_coordinates_flat, archive_array)
    max_distances = max_distances.reshape(x_coordinates.shape)
    rounded_positions = jnp.around(x_coordinates_flat, decimals=decimals)
    for pos in rounded_positions[~jnp.isnan(rounded_positions)]:
        novelty_archive.add(float(pos))
    return max_distances


def config_to_run_name(config: Dict, date: str = None):
    if date is None:
        date = str(datetime.today())
    solver = config["solver"]
    if config["n_generations"] > 1000:
        solver += "-long"
    if config["solver"] == "cgp" and config["n_nodes"] > 50:
        solver += "-large"
    if config["solver"] == "cgp" and config.get("levels_back") is not None:
        solver += "-local"
    if config.get("weights_sigma", 0) != 0:
        solver += "-weighted"
    env_name = config["problem"]["environment"]
    ea = "1+lambda" if config["n_parallel_runs"] > 1 else "mu+lambda"
    day, month = int(date[8:10]), int(date[5:7])
    fitness = "reward"
    if config.get("novelty") is not None:
        fitness = "novelty"
    if config.get("distance", False):
        fitness = "distance"
    if config.get("weighted_rewards", None) is not None:
        weights = config["weighted_rewards"]
        healthy_w, ctrl_w, forward_w = weights["healthy"], weights["ctrl"], weights["forward"]
        fitness = f"weighted-{healthy_w}-{ctrl_w}-{forward_w}"
    if day >= 30 or month > 6:
        ea += "-ga"
        if config.get("unhealthy_termination", True):
            fitness += "-unhealthy_termination"
        else:
            fitness += "-no_termination"
    if month >= 7 and ((solver == "lgp" and day >= 6) or day >= 7):
        ea = ea.replace("-ga", "-ga1")
    seed = config["seed"]
    run_name = f"{env_name}_{solver}_{ea}_{fitness}_{seed}"
    return run_name, env_name, solver, ea, fitness, seed


# methods for update notifications

async def send_telegram_text(bot: telegram.Bot, chat: str, text: str):
    await bot.send_message(chat, text=text)


def notify_update(text: str, bot: telegram.Bot = None, chat: str = None):
    print(text)
    if bot and chat:
        asyncio.get_event_loop().run_until_complete(send_telegram_text(bot, chat, text))


# methods for configs management

def _unnest_dictionary(config: Dict, nesting_keyword: str = "nested") -> List[Dict]:
    nested_values = config[nesting_keyword]
    del config[nesting_keyword]
    return [dict(config, **x) for x in nested_values]


def _unnest_dictionary_recursive(config: Dict, nesting_keyword: str = "nested") -> List[Dict]:
    nesting_keywords = [x for x in config.keys() if nesting_keyword in x]
    configs = [config]
    partially_unpacked_configs = []
    for keyword in nesting_keywords:
        for conf in configs:
            partially_unpacked_configs += _unnest_dictionary(conf, keyword)
        configs = partially_unpacked_configs
        partially_unpacked_configs = []
    return configs


def _unpack_dictionary(config: Dict) -> List[Dict]:
    config_list = []
    for key, value in config.items():
        if type(value) == dict:
            unpacked_value = _unpack_dictionary(value)
            if len(unpacked_value) > 1:
                config[key] = unpacked_value
    for key, value in config.items():
        if type(value) == list and len(value) > 1 and not key.endswith("_list"):
            for v in value:
                temp_config = copy.deepcopy(config)
                temp_config[key] = v
                config_list += _unpack_dictionary(temp_config)
            break
    if len(config_list) == 0:
        config_list.append(config)
    return [{key.replace("_list", ""): value for key, value in cfg.items()} for cfg in config_list]


def process_dictionary(config: Dict, nesting_keyword: str = "nested") -> List[Dict]:
    config_list = _unnest_dictionary_recursive(config, nesting_keyword)
    return list(reduce(lambda x, y: x + y, [_unpack_dictionary(x) for x in config_list], []))
