import os
from typing import Tuple

import jax.numpy as jnp
from brax.v1 import envs
from brax.v1.envs.wrappers import EpisodeWrapper

from jax import random
from brax.v1.io import html
import cgpax.jax_encoding

from jax import jit

from cgpax.utils import readable_cgp_program_from_genome, cgp_graph_from_genome, readable_lgp_program_from_genome, \
    lgp_graph_from_genome, cgp_expression_from_genome, lgp_expression_from_genome
from cgpax.jax_encoding import genome_to_cgp_program, genome_to_lgp_program
from cgpax.run_utils import __update_config_with_env_data__


def __write_readable_program__(genome: jnp.ndarray, config: dict, target_file: str = None):
    if config["solver"] == "cgp":
        readable_program = readable_cgp_program_from_genome(genome, config)
    else:
        readable_program = readable_lgp_program_from_genome(genome, config)
    if target_file is None:
        print(readable_program)
    else:
        with open(target_file, "w") as f:
            f.write(readable_program)


def __write_expression__(genome: jnp.ndarray, config: dict, target_file: str = None):
    if config["solver"] == "cgp":
        readable_expression = cgp_expression_from_genome(genome, config)
    else:
        readable_expression = lgp_expression_from_genome(genome, config)
    if target_file is None:
        print(readable_expression)
    else:
        with open(target_file, "w") as f:
            f.write(readable_expression)


def __save_graph__(genome: jnp.ndarray, config: dict, file: str, input_color: str = None, output_color: str = None):
    if config["solver"] == "cgp":
        graph = cgp_graph_from_genome(genome, config)
    else:
        graph = lgp_graph_from_genome(genome, config)
    graph.layout()
    if input_color is not None or output_color is not None:
        for node in graph.iternodes():
            if node.startswith("i_") and input_color is not None:
                node.attr["color"] = input_color
            elif node.startswith("o_") and output_color is not None:
                node.attr["color"] = output_color
        graph.draw(file)


def __save_html_visualization__(genome: jnp.ndarray, config: dict, env: EpisodeWrapper, file: str = None):
    if file is None:
        file = f'{config["problem"]["environment"]}.html'

    # full episode visualization
    if config["solver"] == "cgp":
        program = genome_to_cgp_program(genome, config)
    else:
        program = genome_to_lgp_program(genome, config)

    jit_env_reset = jit(env.reset)
    jit_env_step = jit(env.step)
    jit_program = jit(program)
    rollout = []
    rng = random.PRNGKey(seed=config["seed"])
    state = jit_env_reset(rng=rng)
    buffer = jnp.zeros(config["buffer_size"]) if config["solver"] == "cgp" else jnp.zeros(config["n_registers"])
    reward = 0
    for timestep in range(config["problem"]["episode_length"]):
        rollout.append(state)
        buffer, actions = jit_program(state.obs, buffer)
        state = jit_env_step(state, actions)
        done = state.done
        if done:
            print(f"DONE [timestep {timestep}]")
            if config["unhealthy_termination"]:
                break
        else:
            reward += state.reward

    with open(file, "w") as f_episode:
        html_content = html.render(env.sys, [s.qp for s in rollout])
        f_episode.write(html_content.replace("<title>brax visualizer",
                                             f"<title>{file.split('/')[-1].replace('.html', '')} - reward {reward}"))
    return reward


def __load_genome__(base_path: str, seed: int, generation: int = "last") -> Tuple[jnp.ndarray, int]:
    generations = []
    for gene_file in os.listdir(base_path):
        if gene_file == "config.yaml":
            continue
        generations.append(int(gene_file.split("_")[1]))

    gen = max(generations) if generation < 0 else min(generations, key=lambda x: abs(generation - x))
    return jnp.load(f"{base_path}/{seed}_{gen}_best_genome.npy", allow_pickle=True).astype(int), gen


if __name__ == '__main__':
    analysis_config = cgpax.get_config("../configs/analysis.yaml")
    seed = analysis_config["seed"]
    base_path = None
    for folder in os.listdir("genomes"):
        if folder.endswith(f"_{seed}"):
            print(f"{folder}")
            base_path = f"genomes/{folder}"

            target_dir = f"{analysis_config['target_dir']}"
            if f"{folder}.png" in os.listdir(target_dir):
                continue

            cfg = cgpax.get_config(f"{base_path}/config.yaml")
            genes, generation = __load_genome__(base_path, seed, analysis_config["generation"])

            environment = envs.get_environment(env_name=cfg["problem"]["environment"])
            environment = EpisodeWrapper(environment, episode_length=cfg["problem"]["episode_length"], action_repeat=1)
            __update_config_with_env_data__(cfg, environment)

            __write_readable_program__(genes, cfg, f"{target_dir}/{folder}.txt")
            __write_expression__(genes, cfg, f"{target_dir}/{folder}_expression.txt")
            __save_graph__(genes, cfg, f"{target_dir}/{folder}.png", analysis_config["input_color"],
                           analysis_config["output_color"])

            if analysis_config["save_visualization"]:
                replay_reward = __save_html_visualization__(genes, cfg, environment,
                                                            f"{target_dir}/{folder}.html")
                print(f"Total reward = {replay_reward}")
                print()
