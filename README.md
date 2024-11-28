# CGPAX

Simple implementations of Cartesian Genetic Programming (CGP) and Linear Genetic Programming (LGP) in JAX.
The main goal is to interface CGP and LGP with Brax (an implementation of several Mujoco environments in JAX) to
leverage parallelization on GPU and speed up the optimization.

## Set up
Clone the current repo, create the conda environment for the project, and activate it.
```
conda env create -f environment.yml
conda activate gpax
```

You will need to login to [Weights&Biases](https://docs.wandb.ai/) for your experiments to be tracked online:
```
wandb login
```

The project also uses a [Telegram bot](https://python-telegram-bot.org/) to send you notifications about the status of experiments.
For it to work you need a `telegram/token.yaml` file structured as follows:
```yaml
token: abc
chat_id: 012
```
More info on telegram bots and their creation is available at [https://core.telegram.org/bots/tutorial](https://core.telegram.org/bots/tutorial).

## Non-JAX version
This project does not natively support non-Jax environments.
A simple version of this project edited to work with non-Jax environments is available at [https://github.com/giorgia-nadizar/MarioGP-T](https://github.com/giorgia-nadizar/MarioGP-T).

## Citation
If you use this code in your research paper please cite:
```
@inproceedings{nadizar2024naturally,
  title={Naturally Interpretable Control Policies via Graph-Based Genetic Programming},
  author={Nadizar, Giorgia and Medvet, Eric and Wilson, Dennis G},
  booktitle={European Conference on Genetic Programming (Part of EvoStar)},
  pages={73--89},
  year={2024},
  organization={Springer}
}
```