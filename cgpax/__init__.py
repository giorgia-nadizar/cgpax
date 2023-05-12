import yaml as yaml


def get_config(filename: str) -> dict:
    return yaml.safe_load(filename)
