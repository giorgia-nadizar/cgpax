from pathlib import Path
from typing import Dict

import yaml as yaml


def get_config(filename: str) -> Dict:
    config = yaml.safe_load(Path(filename).read_text())
    flat_config = {}
    for key in config:
        if type(config[key]) == dict and "value" in config[key]:
            flat_config[key] = config[key]["value"]
        else:
            flat_config[key] = config[key]
    return flat_config
