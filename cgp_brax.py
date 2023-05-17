import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from datetime import datetime

from functools import partial

from jax import jit, vmap, random
from jax import numpy as jnp
from brax import envs
from brax.envs.wrapper import EpisodeWrapper

from cgpax.jax_individual import compute_genome_mask, generate_population, compute_mutation_prob_mask, \
    mutate_genome_n_times
from cgpax.jax_evaluation import evaluate_genome
from cgpax.jax_selection import fp_selection

config = {
    "problem": {"environment": "halfcheetah", "maximize": True, "episode_length": 1000},
    "backend": "positional",
    "n_nodes": 100,
    "n_individuals": 500,
    "elite_size": 50,
    "functions": ["plus", "minus", "times", "prot_div"],
    "p_mut_inputs": 0.2,
    "p_mut_functions": 0.2,
    "p_mut_outputs": 0.2,
    "constrain_outputs": True,
    "seed": 1
}

if __name__ == "__main__":
    rnd_key = random.PRNGKey(config["seed"])

    env = envs.get_environment(env_name=config["problem"]["environment"], backend=config["backend"])
    env = EpisodeWrapper(
        env, episode_length=config["problem"]["episode_length"], action_repeat=1
    )

    config["n_in"] = env.observation_size
    config["n_out"] = env.action_size
    config["buffer_size"] = config["n_in"] + config["n_nodes"]
    n_mutations = int(config["n_individuals"] / config["elite_size"])

    genome_mask = compute_genome_mask(config, config["n_in"], config["n_out"])
    mutation_mask = compute_mutation_prob_mask(config, config["n_out"])

    # initialize population
    print(f"{datetime.now()} - {config['n_individuals']} individuals generation called")
    rnd_key, genome_key = random.split(rnd_key, 2)
    genomes = generate_population(pop_size=config["n_individuals"], genome_mask=genome_mask, rnd_key=genome_key)
    print(f"{datetime.now()} - {config['n_individuals']} individuals generation done\n")

    # evaluate population
    partial_eval_genome = partial(evaluate_genome, config=config, env=env)
    vmap_evaluate_genome = vmap(partial_eval_genome, in_axes=(0, None))
    jit_vmap_evaluate_genome = jit(vmap_evaluate_genome)
    print(f"{datetime.now()} - {config['n_individuals']} CGP called")
    rnd_key, eval_key = random.split(rnd_key, 2)
    fitness_values = jit_vmap_evaluate_genome(genomes, eval_key)
    print(f"{datetime.now()} - {config['n_individuals']} CGP done (fitness values)\n")

    # select parents
    print(f"{datetime.now()} - FP selection called")
    partial_fp_selection = partial(fp_selection, n_elites=config["elite_size"])
    jit_partial_fp_selection = jit(partial_fp_selection)
    rnd_key, fp_key = random.split(rnd_key, 2)
    parents = jit_partial_fp_selection(genomes, fitness_values, fp_key)
    print(f"{datetime.now()} - FP selection done\n")

    # compute offspring
    print(f"{datetime.now()} - mutation called")
    rnd_key, mutate_key = random.split(rnd_key, 2)
    mutate_keys = random.split(mutate_key, len(parents))
    partial_multiple_mutations = partial(mutate_genome_n_times, n_mutations=n_mutations, genome_mask=genome_mask,
                                         mutation_mask=mutation_mask)
    vmap_multiple_mutations = vmap(partial_multiple_mutations)
    jit_vmap_multiple_mutations = jit(vmap_multiple_mutations)
    new_genomes_matrix = jit_vmap_multiple_mutations(parents, mutate_keys)
    new_genomes = jnp.reshape(new_genomes_matrix, (-1, new_genomes_matrix.shape[-1]))
    print(f"{datetime.now()} - mutation done\n")

    # evaluate offspring
    print(f"{datetime.now()} - {config['n_individuals']} mutated CGP called")
    rnd_key, mutated_eval_key = random.split(rnd_key, 2)
    new_fitness_values = jit_vmap_evaluate_genome(new_genomes, mutated_eval_key)
    print(f"{datetime.now()} - {config['n_individuals']} mutated CGP done (fitness values)\n")
