from pathlib import Path

import yaml as yaml


def get_config(filename: str) -> dict:
    config = yaml.safe_load(Path(filename).read_text())
    flat_config = {}
    for key in config:
        if type(config[key]) == dict and "value" in config[key]:
            flat_config[key] = config[key]["value"]
        else:
            flat_config[key] = config[key]
    return flat_config
