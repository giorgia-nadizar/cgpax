from jax import default_backend

import cgpax
from cgpax.run_utils import process_dictionary
from run_gomea_boolean import run_gomea_boolean
from run_gomea_classification import run_gomea_classification
from run_gomea_discrete_control import run_gomea_discrete_control
from run_gomea_continuous_control import run_gomea_continuous_control

if __name__ == '__main__':

    print(f"Starting the run with {default_backend()} as backend...")

    problems = {
        "boolean": run_gomea_boolean,
        "classification": run_gomea_classification,
        "discrete_control": run_gomea_discrete_control,
        "continuous_control": run_gomea_continuous_control
    }

    for problem, run_gomea in problems.items():

        config_files = [f"configs/graph_gp_gomea_{problem}.yaml"]
        unpacked_configs = []

        for config_file in config_files:
            unpacked_configs += process_dictionary(cgpax.get_config(config_file))

        print(f"\n\nRunning {problem}...")
        print(f"Total configs found: {2 * len(unpacked_configs)}")
        for fos_mode in ["U", "RT"]:
            for cfg in unpacked_configs:
                cfg["fos_mode"] = fos_mode
                cfg[
                    "run_name"] = f"gomea_{cfg['solver']}_{cfg['problem']['environment']}_{cfg['fos_mode']}_{cfg['seed']}"
                print(cfg["run_name"])
                run_gomea(cfg)
                print()
