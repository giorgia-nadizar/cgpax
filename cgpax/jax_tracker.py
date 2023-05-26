import jax.numpy as jnp
from jax import jit
import chex

from functools import partial
from time import time
from pathlib import Path


# NOTE: once initialized, the object should not be modified in compiled functions
class Tracker:
    def __init__(self, config: dict, top_k: int = 3) -> None:
        self.config: dict = config
        self.top_k: int = top_k

    @partial(jit, static_argnums=(0,))
    def init(self) -> chex.ArrayTree:
        return {
            "training": {
                # Fitness of the top k individuals during training (ordered)
                "top_k_fit": jnp.zeros((self.config["n_generations"], self.top_k)),
                "fitness_mean": jnp.zeros((self.config["n_generations"],)),
                "fitness_std": jnp.zeros((self.config["n_generations"],)),
                "fitness_median": jnp.zeros((self.config["n_generations"],)),
                "fitness_1q": jnp.zeros((self.config["n_generations"],)),
                "fitness_3q": jnp.zeros((self.config["n_generations"],)),
                "selection_time": jnp.zeros(self.config["n_generations"]),
                "mutation_time": jnp.zeros(self.config["n_generations"]),
                "evaluation_time": jnp.zeros(self.config["n_generations"])
            },
            "backup": {
                # best individual in the population, (n_gen, indiv_size)
                "best_individual": jnp.zeros((self.config["n_generations"], self.config["genome_size"],)),
            },
            "generation": 0,
        }

    @partial(jit, static_argnums=(0,))
    def update(self, tracker_state: chex.ArrayTree, fitness: chex.Array, best_individual: chex.Array,
               selection_time: float, mutation_time: float, evaluation_time: float) -> chex.ArrayTree:
        i = tracker_state["generation"]
        # [Training] - update top_k_fitness using old state (carry best over)
        # last_fit = (tracker_state["training"]["top_k_fit"].at[i - 1].get(mode="fill", fill_value=0.0))
        # top_k_f = jnp.sort(jnp.hstack((fitness, last_fit)))[::-1][: self.top_k]
        # TODO -> choose what makes most sense
        # without carrying the old state
        top_k_f = jnp.sort(fitness)[::-1][: self.top_k]

        # NOTE - Update top k fitness values
        tracker_state["training"]["top_k_fit"] = (tracker_state["training"]["top_k_fit"].at[i].set(top_k_f))

        # NOTE - Update fitness statistics
        tracker_state["training"]["fitness_mean"] = (
            tracker_state["training"]["fitness_mean"].at[i].set(fitness.mean())
        )
        tracker_state["training"]["fitness_std"] = (
            tracker_state["training"]["fitness_std"].at[i].set(fitness.std())
        )
        tracker_state["training"]["fitness_median"] = (
            tracker_state["training"]["fitness_median"].at[i].set(jnp.median(fitness))
        )
        tracker_state["training"]["fitness_1q"] = (
            tracker_state["training"]["fitness_1q"].at[i].set(jnp.quantile(fitness, 0.75))
        )
        tracker_state["training"]["fitness_3q"] = (
            tracker_state["training"]["fitness_3q"].at[i].set(jnp.quantile(fitness, 0.25))
        )

        # NOTE: Update times taken for the generation
        tracker_state["training"]["selection_time"] = (
            tracker_state["training"]["selection_time"].at[i].set(selection_time))
        tracker_state["training"]["mutation_time"] = (
            tracker_state["training"]["mutation_time"].at[i].set(mutation_time))
        tracker_state["training"]["evaluation_time"] = (
            tracker_state["training"]["evaluation_time"].at[i].set(evaluation_time))

        # NOTE: Update backup individuals
        tracker_state["backup"]["best_individual"] = (
            tracker_state["backup"]["best_individual"].at[i].set(best_individual)
        )

        # NOTE - Update current generation counter
        tracker_state["generation"] += 1
        return tracker_state

    def wandb_log(self, tracker_state, wdb_run) -> None:
        gen = tracker_state["generation"] - 1

        wdb_run.log(
            {
                "training": {
                    "generation": gen,
                    f"top_k_fit": {
                        f"top_{t}_fit": float(tracker_state["training"]["top_k_fit"][gen][t]) for t in range(self.top_k)
                    },
                    "fitness_mean": float(tracker_state["training"]["fitness_mean"][gen]),
                    "fitness_std": float(tracker_state["training"]["fitness_std"][gen]),
                    "fitness_median": float(tracker_state["training"]["fitness_median"][gen]),
                    "fitness_1q": float(tracker_state["training"]["fitness_1q"][gen]),
                    "fitness_3q": float(tracker_state["training"]["fitness_3q"][gen]),
                },
            }
        )

    def wandb_save_genome(self, genome, wdb_run) -> None:
        save_path = Path(wdb_run.dir) / "genomes" / f"{str(int(time()))}_best_individual.npy"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as temp_f:
            jnp.save(temp_f, genome)
