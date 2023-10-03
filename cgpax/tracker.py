from typing import Dict, List

import jax.numpy as jnp
from jax import jit
import chex

from functools import partial
from time import time
from pathlib import Path


# NOTE: once initialized, the object should not be modified in compiled functions
class Tracker:
    def __init__(self, config: Dict, top_k: int = 3, idx: int = 0, saving_interval: int = 100,
                 store_fitness_details: List[str] = None) -> None:
        self.config: Dict = config
        self.top_k: int = top_k
        self.idx: int = idx
        self.saving_interval: int = saving_interval
        self.store_fitness_details: List[str] = [] if store_fitness_details is None else store_fitness_details

    @partial(jit, static_argnums=(0,))
    def init(self) -> chex.ArrayTree:
        init_dict = {
            "training": {
                # Fitness of the top k individuals during training (ordered)
                "top_k_fit": jnp.zeros((self.config["n_generations"], self.top_k)),
                "top_k_reward": jnp.zeros((self.config["n_generations"], self.top_k)),
                "top_k_healthy_reward": jnp.zeros((self.config["n_generations"], self.top_k)),
                "top_k_ctrl_reward": jnp.zeros((self.config["n_generations"], self.top_k)),
                "top_k_forward_reward": jnp.zeros((self.config["n_generations"], self.top_k)),
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
        for fitness_detail in self.store_fitness_details:
            init_dict[fitness_detail] = jnp.zeros((self.config["n_generations"], self.config["n_individuals"]))
        return init_dict

    @partial(jit, static_argnums=(0,))
    def update(self, tracker_state: chex.ArrayTree, fitness: chex.Array, rewards: chex.Array, detailed_rewards: Dict,
               best_individual: chex.Array, times: Dict) -> chex.ArrayTree:
        i = tracker_state["generation"]
        # [Training] - update top_k_fitness using old state (carry best over)

        # without carrying the old state
        top_k_ids, _ = jnp.split(jnp.argsort(-fitness), [self.top_k])
        top_k_f = jnp.take(fitness, top_k_ids)
        top_k_r = jnp.take(rewards, top_k_ids)
        top_k_r_healthy = jnp.take(detailed_rewards["healthy"], top_k_ids)
        top_k_r_ctrl = jnp.take(detailed_rewards["ctrl"], top_k_ids)
        top_k_r_forward = jnp.take(detailed_rewards["forward"], top_k_ids)

        # NOTE - Update top k fitness and reward values
        tracker_state["training"]["top_k_fit"] = (tracker_state["training"]["top_k_fit"].at[i].set(top_k_f))
        tracker_state["training"]["top_k_reward"] = (tracker_state["training"]["top_k_reward"].at[i].set(top_k_r))
        tracker_state["training"]["top_k_healthy_reward"] = (
            tracker_state["training"]["top_k_healthy_reward"].at[i].set(top_k_r_healthy))
        tracker_state["training"]["top_k_ctrl_reward"] = (
            tracker_state["training"]["top_k_ctrl_reward"].at[i].set(top_k_r_ctrl))
        tracker_state["training"]["top_k_forward_reward"] = (
            tracker_state["training"]["top_k_forward_reward"].at[i].set(top_k_r_forward))

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
            tracker_state["training"]["selection_time"].at[i].set(times["selection_time"]))
        tracker_state["training"]["mutation_time"] = (
            tracker_state["training"]["mutation_time"].at[i].set(times["mutation_time"]))
        tracker_state["training"]["evaluation_time"] = (
            tracker_state["training"]["evaluation_time"].at[i].set(times["evaluation_time"]))

        # NOTE: Update backup individuals
        tracker_state["backup"]["best_individual"] = (
            tracker_state["backup"]["best_individual"].at[i].set(best_individual)
        )

        # NOTE: Update current generation counter
        tracker_state["generation"] += 1

        # NOTE: Update additional fitness information
        for fitness_detail in self.store_fitness_details:
            tracker_state[fitness_detail] = tracker_state[fitness_detail].at[i].set(detailed_rewards[fitness_detail])

        return tracker_state

    def wandb_log(self, tracker_state, wdb_run) -> None:
        gen = tracker_state["generation"] - 1
        # NOTE: Store best genome occasionally
        if tracker_state["generation"] % self.saving_interval == 0:
            prefix = f"{self.idx}_{gen}"
            self.wandb_save_genome(tracker_state["backup"]["best_individual"][gen], wdb_run, prefix)

        log_dict = {
            "training": {
                "run_id": self.idx,
                "generation": gen,
                f"top_k_fit": {
                    f"top_{t}_fit": float(tracker_state["training"]["top_k_fit"][gen][t]) for t in range(self.top_k)
                },
                f"top_k_reward": {
                    f"top_{t}_reward": float(tracker_state["training"]["top_k_reward"][gen][t]) for t in
                    range(self.top_k)
                },
                f"top_k_healthy_reward": {
                    f"top_{t}_healthy_reward": float(tracker_state["training"]["top_k_healthy_reward"][gen][t]) for
                    t in range(self.top_k)
                },
                f"top_k_ctrl_reward": {
                    f"top_{t}_ctrl_reward": float(tracker_state["training"]["top_k_ctrl_reward"][gen][t]) for t in
                    range(self.top_k)},
                f"top_k_forward_reward": {
                    f"top_{t}_forward_reward": float(tracker_state["training"]["top_k_forward_reward"][gen][t]) for
                    t in range(self.top_k)
                },
                "fitness_mean": float(tracker_state["training"]["fitness_mean"][gen]),
                "fitness_std": float(tracker_state["training"]["fitness_std"][gen]),
                "fitness_median": float(tracker_state["training"]["fitness_median"][gen]),
                "fitness_1q": float(tracker_state["training"]["fitness_1q"][gen]),
                "fitness_3q": float(tracker_state["training"]["fitness_3q"][gen]),
                "evaluation_time": float(tracker_state["training"]["evaluation_time"][gen])
            },
        }

        for fitness_detail in self.store_fitness_details:
            log_dict["training"][fitness_detail] = jnp.array_str(tracker_state[fitness_detail][gen])

        wdb_run.log(log_dict)

    @staticmethod
    def wandb_save_genome(genome, wdb_run, prefix=None) -> None:
        if prefix is None:
            prefix = str(int(time()))
        save_path = Path(wdb_run.dir) / "genomes" / f"{prefix}_best_genome.npy"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as temp_f:
            jnp.save(temp_f, genome)
