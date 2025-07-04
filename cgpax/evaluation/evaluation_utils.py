from functools import partial
from typing import Dict, Callable, Tuple, Union

import jax.numpy as jnp
from jax import vmap, random, jit

# data only
from cgpax.run_utils import update_config_with_data, load_dataset, compile_genome_evaluation

# control only
from cgpax.run_utils import init_environment_from_config, update_config_with_env_data

# boolean only
from cgpax.evaluation.boolean_evaluation import evaluate_cgp_genome as boolean_evaluate_cgp_genome
from cgpax.evaluation.boolean_evaluation import evaluate_lgp_genome as boolean_evaluate_lgp_genome
from cgpax.functions import function_set_boolean

# classification only
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cgpax.evaluation.classification_evaluation import evaluate_cgp_genome as classification_evaluate_cgp_genome
from cgpax.evaluation.classification_evaluation import evaluate_lgp_genome as classification_evaluate_lgp_genome
from cgpax.functions import function_set_numeric


def prepare_evaluation_functions_discrete_control(config: Dict) -> Tuple[Callable, Union[Callable, None]]:
    environment = init_environment_from_config(config)
    update_config_with_env_data(config, environment)
    evaluate_genomes = compile_genome_evaluation(config, environment, config["problem"]["episode_length"])
    replace_invalid_nan_reward = jit(partial(jnp.nan_to_num, nan=config["nan_replacement"]))

    # compose genome eval
    def _genomes_to_fitnesses(gs: jnp.ndarray, rnd_keys: jnp.ndarray) -> jnp.ndarray:
        evaluation_outcomes = evaluate_genomes(gs, jnp.array(rnd_keys))
        cumulative_rewards = evaluation_outcomes
        return replace_invalid_nan_reward(cumulative_rewards)

    return _genomes_to_fitnesses, None


def prepare_evaluation_functions_boolean(config: Dict) -> Tuple[Callable, Union[Callable, None]]:
    config["use_input_constants"] = False
    x_values, y_values = load_dataset(config["problem"])
    update_config_with_data(config, x_values.shape[1], y_values.shape[1], function_set=function_set_boolean)
    genome_evaluation_function = boolean_evaluate_cgp_genome if config[
                                                                    "solver"] == "cgp" else boolean_evaluate_lgp_genome
    genome_to_fitness = partial(genome_evaluation_function, config=config, x_values=x_values, y_values=y_values)

    def _genomes_to_fitnesses(genotypes: jnp.ndarray, fake_rnd_keys: jnp.ndarray = None) -> jnp.ndarray:
        return vmap(genome_to_fitness)(genotypes)["accuracy"]

    return _genomes_to_fitnesses, None


def prepare_evaluation_functions_classification(config: Dict) -> Tuple[Callable, Union[Callable, None]]:
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

    genome_evaluation_function = classification_evaluate_cgp_genome if config["solver"] == "cgp" \
        else classification_evaluate_lgp_genome
    genome_to_fitness = partial(genome_evaluation_function, config=config, x_values=x_values, y_values=y_values)

    def _genomes_to_fitnesses(genotypes: jnp.ndarray, fake_rnd_keys: jnp.ndarray = None) -> jnp.ndarray:
        return vmap(genome_to_fitness)(genotypes)["accuracy"]

    def _genome_to_test_accuracy(genotype: jnp.ndarray, fake_rnd_key: random.PRNGKey = None) -> float:
        return genome_evaluation_function(genotype, config=config, x_values=x_test, y_values=y_test)["accuracy"]

    return _genomes_to_fitnesses, _genome_to_test_accuracy
