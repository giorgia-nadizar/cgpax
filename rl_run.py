import functools
import os
from typing import Dict, Callable

import telegram
from brax.envs import ant
from jax import default_backend

import cgpax
import wandb
from brax import envs
from brax.io import model

from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac

from cgpax.run_utils import process_dictionary, notify_update


def training_function(env: str, solver: str, seed: int) -> Callable:
    if solver == "sac":
        return functools.partial(sac.train, num_timesteps=7_864_320, num_evals=20, reward_scaling=5,
                                 episode_length=1000, normalize_observations=True, action_repeat=1,
                                 discounting=0.997, learning_rate=6e-4, num_envs=128, batch_size=128,
                                 grad_updates_per_step=32, max_devices_per_host=1, max_replay_size=1048576,
                                 min_replay_size=8192, seed=seed)
    else:
        ppo_fns = {
            'inverted_pendulum': functools.partial(ppo.train, num_timesteps=2_000_000, num_evals=20, reward_scaling=10,
                                                   episode_length=1000, normalize_observations=True, action_repeat=1,
                                                   unroll_length=5, num_minibatches=32, num_updates_per_batch=4,
                                                   discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2,
                                                   num_envs=2048,
                                                   batch_size=1024, seed=seed),
            'inverted_double_pendulum': functools.partial(ppo.train, num_timesteps=20_000_000, num_evals=20,
                                                          reward_scaling=10,
                                                          episode_length=1000, normalize_observations=True,
                                                          action_repeat=1,
                                                          unroll_length=5, num_minibatches=32, num_updates_per_batch=4,
                                                          discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2,
                                                          num_envs=2048, batch_size=1024, seed=seed),
            'ant': functools.partial(ppo.train, num_timesteps=50_000_000, num_evals=20, reward_scaling=10,
                                     episode_length=1000, normalize_observations=True, action_repeat=1, unroll_length=5,
                                     num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4,
                                     entropy_cost=1e-2, num_envs=4096, batch_size=2048, seed=seed),
            'miniant': functools.partial(ppo.train, num_timesteps=50_000_000, num_evals=20, reward_scaling=10,
                                         episode_length=1000, normalize_observations=True, action_repeat=1,
                                         unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97,
                                         learning_rate=3e-4, entropy_cost=1e-2, num_envs=4096, batch_size=2048,
                                         seed=seed),
            'reacher': functools.partial(ppo.train, num_timesteps=50_000_000, num_evals=20, reward_scaling=5,
                                         episode_length=1000, normalize_observations=True, action_repeat=4,
                                         unroll_length=50, num_minibatches=32, num_updates_per_batch=8,
                                         discounting=0.95,
                                         learning_rate=3e-4, entropy_cost=1e-3, num_envs=2048, batch_size=256,
                                         max_devices_per_host=8, seed=seed),
            'hopper': functools.partial(ppo.train, num_timesteps=50_000_000, num_evals=20, reward_scaling=10,
                                        episode_length=1000, normalize_observations=True, action_repeat=1,
                                        unroll_length=20, num_minibatches=32, num_updates_per_batch=4, discounting=0.97,
                                        learning_rate=3e-4, entropy_cost=1e-3, num_envs=4096, batch_size=2048,
                                        seed=seed),
            'walker2d': functools.partial(ppo.train, num_timesteps=50_000_000, num_evals=20, reward_scaling=10,
                                          episode_length=1000, normalize_observations=True, action_repeat=1,
                                          unroll_length=20, num_minibatches=32, num_updates_per_batch=4,
                                          discounting=0.97,
                                          learning_rate=3e-4, entropy_cost=1e-3, num_envs=4096, batch_size=2048,
                                          seed=seed),
            'halfcheetah': functools.partial(ppo.train, num_timesteps=50_000_000, num_evals=20, reward_scaling=1,
                                             episode_length=1000, normalize_observations=True, action_repeat=1,
                                             unroll_length=20, num_minibatches=32, num_updates_per_batch=8,
                                             discounting=0.95, learning_rate=3e-4, entropy_cost=0.001, num_envs=2048,
                                             batch_size=512, seed=seed),
            'swimmer': functools.partial(ppo.train, num_timesteps=50_000_000, num_evals=20, reward_scaling=1,
                                         episode_length=1000, normalize_observations=True, action_repeat=1,
                                         unroll_length=20, num_minibatches=32, num_updates_per_batch=8,
                                         discounting=0.95,
                                         learning_rate=3e-4, entropy_cost=0.001, num_envs=2048,
                                         batch_size=512, seed=seed),
        }
        return ppo_fns[env]


def run(config: Dict, run_name: str):
    wdb_run = wandb.init(config=config, project="cgpax", name=run_name)
    train_fn = training_function(env=config["environment"], seed=config["seed"], solver=config["rl"]["trainer"])
    if config["environment"] == "miniant":
        env = functools.partial(ant.Ant, use_contact_forces=False)()
    else:
        env = envs.get_environment(env_name=config["environment"])

    def progress(step: int, metrics: Dict):
        wdb_run.log(data=metrics, step=step)

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
    model.save_params(os.path.join(wdb_run.dir, "params"), params)
    wdb_run.finish()


if __name__ == '__main__':

    print(f"Starting the run with {default_backend()} as backend...")

    telegram_config = cgpax.get_config("telegram/token.yaml")
    telegram_bot = telegram.Bot(telegram_config["token"])

    config_file = "configs/rl.yaml"
    configs = process_dictionary(cgpax.get_config(config_file))
    for count, cfg in enumerate(configs):
        run_name = f'RL_{cfg["environment"]}_{cfg["rl"]["trainer"]}_{cfg["seed"]}'
        notify_update(f"{count + 1}/{len(configs)} - {run_name} starting\n{cfg}", telegram_bot,
                      telegram_config["chat_id"])
        run(cfg, run_name)
