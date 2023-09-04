import functools
import os
from typing import Dict, Callable

from jax import default_backend

import cgpax
import wandb
from brax.v1 import envs
from brax.v1.io import model

from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac

from run import __unpack_dictionary__


def get_training_function(trainer: str, seed: int) -> Callable:
    return {
        "ppo": functools.partial(ppo.train,
                                 learning_rate=3e-4,
                                 entropy_cost=0.0,  # often 10^-2 or 10^-3
                                 num_minibatches=32,
                                 discounting=0.99,
                                 normalize_observations=True,
                                 seed=seed,
                                 episode_length=1000,

                                 num_evals=20,  # depends on the env

                                 num_timesteps=50_000_000,
                                 reward_scaling=1, # in the loss function
                                 action_repeat=1,
                                 unroll_length=20,
                                 num_updates_per_batch=8,
                                 num_envs=2048,
                                 batch_size=512),
        "sac": functools.partial(sac.train, num_timesteps=7_864_320, num_evals=20, reward_scaling=5,
                                 episode_length=1000, normalize_observations=True, action_repeat=1,
                                 discounting=0.997, learning_rate=6e-4, num_envs=128, batch_size=128,
                                 grad_updates_per_step=32, max_devices_per_host=1, max_replay_size=1048576,
                                 min_replay_size=8192, seed=seed)
    }[trainer]


def run(config: Dict):
    wdb_run = wandb.init(config=config, project="cgpax",
                         name=f'RL_{config["environment"]}_{config["rl"]["trainer"]}_{config["seed"]}')
    train_fn = get_training_function(config["rl"]["trainer"], config["seed"])

    env = envs.get_environment(env_name=config["environment"])

    def progress(step: int, metrics: Dict):
        wdb_run.log(data=metrics, step=step)

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
    model.save_params(os.path.join(wdb_run.dir, "params"), params)
    wdb_run.finish()


if __name__ == '__main__':
    assert default_backend() == "gpu"

    config_file = "configs/rl.yaml"
    configs = __unpack_dictionary__(cgpax.get_config(config_file))
    for cfg in configs:
        run(cfg)
