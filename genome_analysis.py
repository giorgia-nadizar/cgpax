import jax.numpy as jnp
from brax import envs
from brax.envs.wrapper import EpisodeWrapper

from jax import random
from brax.io import html
import cgpax.jax_encoding

from jax import jit

from cgpax.utils import readable_cgp_program_from_genome, graph_from_genome, readable_lgp_program_from_genome
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


def __save_graph__(genome: jnp.ndarray, config: dict, file: str, input_color: str = None, output_color: str = None):
    if config["solver"] == "cgp":
        graph = graph_from_genome(genome, config)
        graph.layout()
        if input_color is not None or output_color is not None:
            for node in graph.iternodes():
                if node.startswith("i_") and input_color is not None:
                    node.attr["color"] = input_color
                elif node.startswith("o_") and output_color is not None:
                    node.attr["color"] = output_color
            graph.draw(file)
    else:
        print(f"Cannot save graph for {config['solver']}")


def __save_html_visualization__(genome: jnp.ndarray, config: dict, file_prefix: str = None):
    if file_prefix is None:
        file_prefix = environment

    # initial state visualization
    state = jit(env.reset)(rng=random.PRNGKey(seed=config["seed"]))
    with open(f"{file_prefix}_initial.html", "w") as f_initial:
        f_initial.write(html.render(env.sys, [state.pipeline_state]))

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
    for _ in range(config["problem"]["episode_length"]):
        rollout.append(state.pipeline_state)
        buffer, actions = jit_program(state.obs, buffer)
        state = jit_env_step(state, actions)
        reward += state.reward

    with open(f"{file_prefix}_episode.html", "w") as f_episode:
        f_episode.write(html.render(env.sys.replace(dt=env.dt), rollout))

    return reward


analysis_config = {
    "run": "run-20230607_213638-kcc4izum",
    "seed": 0,
    "generation": 4099,
    "target_dir": "outcomes",
    "program_file": True,
    "graph_file": True,
    "save_visualization": True,
}

if __name__ == '__main__':
    base_path = f"wandb/{analysis_config['run']}/files"
    target_dir = analysis_config['target_dir']

    config = cgpax.get_config(f"{base_path}/config.yaml")
    environment = config['problem']['environment']
    solver = config['solver']
    genome = jnp.load(f"{base_path}/genomes/{analysis_config['seed']}_{analysis_config['generation']}_best_genome.npy",
                      allow_pickle=True).astype(int)

    env = envs.get_environment(env_name=environment, backend=config["backend"])
    env = EpisodeWrapper(env, episode_length=config["problem"]["episode_length"], action_repeat=1)
    __update_config_with_env_data__(config, env)

    program_file = f"{target_dir}/{solver}_{environment}.txt" if analysis_config['program_file'] else None
    __write_readable_program__(genome, config, program_file)
    if analysis_config["graph_file"]:
        __save_graph__(genome, config, f"{target_dir}/{solver}_{environment}.png", "green", "red")

    if analysis_config["save_visualization"]:
        replay_reward = __save_html_visualization__(genome, config, f"{target_dir}/{solver}_{environment}")
        print(f"\nTotal reward = {replay_reward}")
