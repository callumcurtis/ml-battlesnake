# ML-Battlesnake: Playing Battlesnake with Machine Learning

This is a project to play the competitive multiplayer game [Battlesnake](https://play.battlesnake.com/) using machine learning.

NOTE: This project is still in development and is not yet optimized for competitive play.

## Getting Started

Start by cloning the repository and opening the project in a DevContainer. This will ensure that you have all the necessary dependencies installed.

Try running [scripts/ppo.py](scripts/ppo.py) to train a model. Point a tensorboard server at the tensorboard log directory to see the training progress. Refer to the [stable-baselines3 documentation](https://stable-baselines3.readthedocs.io/en/master/) for more information about the PPO algorithm and tensorboard log format.

## Project Structure

- [bin](bin): compiled executables and libraries, such as game engines
- [ml_battlesnake](ml_battlesnake): project package
    - [common](ml_battlesnake/common): common code for all package modules
    - [deployment](ml_battlesnake/deployment): utilities and application code for Battlesnake services and their deployment. This directory will support the development/production deployment of the ML Battlesnake when it is ready.
    - [learning](ml_battlesnake/learning): training and evaluation of machine learning snakes
- [scripts](scripts): scripts for running the project, including local deployment and training
- [tests](tests): unit tests for the project

## Progress

- [x] Create multi-agent, simultaneous action and observation training environment
- [x] Wrap official Battlesnake game engine in a Python API for use in training environment
- [x] Adapt Stable Baselines3 PPO algorithm to work with the environment
- [x] Add observation preprocessing options to the environment
- [x] Add reward shaping to the environment
- [x] Add support for parallel environments to improve data collection time and reduce correlation between samples
- [x] Add scripts and utilities for local deployment of the Battlesnake game engine, browser spectator, and snake servers
- [x] Checkpoint and restore training progress
- [x] Restart training from last checkpoint if resource limits are exceeded
- [x] Experiment training a model to play the game using self-play
    - Reached ~120 mean episode length (game length) for each of the core game modes (solo, duel, and standard) before plateauing. Performance is still suboptimal in terms of observed snake strategy and mean rewards.

## TODO

Plateauing during training is likely due to the observation preprocessing and network architecture. The observation preprocessing uses single-channel (grayscale) image encoding which is not meaningful to the actor/critic networks in their current state. The following items should address these issues and allow for training to progress past the plateau.

Other items below should also improve training efficiency and performance.

- [ ] Swap grayscale image encoding in range [0, 255] to binary encoding with multiple channels, where each channel represents a different feature of the game state
- [ ] Add action masking to the environment to prevent snakes from moving into walls or other snakes, allowing the network to focus on learning less trivial behaviours
- [ ] Add CNN network for feature extraction
- [ ] Shape actor/critic networks to suit the CNN feature extraction network
- [ ] Tune hyperparameters for training
- [ ] Train 4+ snakes to imitate the best performing snakes on the global leaderboard to use in training to avoid [ray interference](https://arxiv.org/abs/1904.11455) that could be emphasized by the self-play training method
- [ ] Add new environment and matchmaking system for pitting the snake in training against the imitation snakes (and potentially its own previous versions)
- [ ] Train at scale
- [ ] Deploy for competitive play on the global leaderboard