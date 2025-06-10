import cgpax
import jax.numpy as jnp
from cgpax.run_utils import process_dictionary, init_environment_from_config, update_config_with_env_data
from cgpax.utils import cgp_expression_from_genome, lgp_expression_from_genome


def readable_policy_from_config(config):
    environment = init_environment_from_config(config)
    update_config_with_env_data(config, environment)

    genomes = jnp.load(f"results/{config['run_name']}/genomes.npy")
    fitnesses = jnp.load(f"results/{config['run_name']}/fitnesses.npy")
    best_genome = genomes[jnp.argmax(fitnesses)]
    if config["solver"] == "cgp":
        readable_expression = cgp_expression_from_genome(best_genome, config)
    elif config["solver"] == "lgp":
        readable_expression = lgp_expression_from_genome(best_genome, config)
    else:
        raise ValueError(config["solver"])
    return readable_expression, config, jnp.max(fitnesses)


def count_variables(readable_expr, n_env_ins):
    counter = 0
    for in_var in range(n_env_ins):
        if f"i{in_var}" in readable_expr:
            counter += 1
    return counter


def replace_expression(readable_expr, env_name):
    replacements = {
        "MountainCar-v0": {
            "o0": "l",
            "o1": "s",
            "o2": "r",
            "i0": "x",
            "i1": "v",
            "i2": "0.1",
            "i3": "1",
        },
        "CartPole-v1": {
            "o0": "l",
            "o1": "r",
            "i0": "x",
            "i1": "v",
            "i2": "\\alpha",
            "i3": "\omega",
            "i4": "0.1",
            "i5": "1"
        },
        "Acrobot-v1": {
            "o0": "t_{-1}",
            "o1": "t_{0}",
            "o2": "t_{1}",
            "i0": "\cos(\\theta_1)",
            "i1": "\sin(\\theta_1)",
            "i2": "\cos(\\theta_2)",
            "i3": "\sin(\\theta_2)",
            "i4": "\omega(\\theta_1)",
            "i5": "\omega(\\theta_2)",
            "i6": "0.1",
            "i7": "1"
        },
        "LunarLander-v3": {
            "o0": "s",
            "o1": "l",
            "o2": "m",
            "o3": "r",
            "i0": "x",
            "i1": "y",
            "i2": "v_x",
            "i3": "v_y",
            "i4": "\\alpha",
            "i5": "\omega",
            "i6": "c_l",
            "i7": "c_r",
            "i8": "0.1",
            "i9": "1"
        },
    }
    repl = replacements[env_name]
    for key, value in repl.items():
        readable_expr = readable_expr.replace(key, value)
    return readable_expr.replace("=", "&=").replace("\n", " \\\\ \n").replace("*", "\cdot")


if __name__ == '__main__':
    config_files = ["configs/graph_gp_discrete.yaml"]
    unpacked_configs = []

    for config_file in config_files:
        unpacked_configs += process_dictionary(cgpax.get_config(config_file))

    for cfg in unpacked_configs:
        seed = 105
        for env in ["MountainCar-v0", "CartPole-v1", "Acrobot-v1", "LunarLander-v3"]:
            for solver in ["lgp", "cgp"]:
                if cfg["seed"] == seed and cfg["problem"]["environment"] == env and cfg["solver"] == solver:
                    processed_env = cfg["problem"]["environment"].lower().split("-")[0]
                    cfg["run_name"] = f"ga_{cfg['solver']}_{processed_env}_{cfg['seed']}"
                    readable_expression, updated_cfg, fitness = readable_policy_from_config(cfg)
                    n_vars = count_variables(readable_expression, n_env_ins=updated_cfg["n_in_env"])
                    print(seed, env, solver, n_vars, fitness)
                    print(replace_expression(readable_expression, env))
                    print()
