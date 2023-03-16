#!/usr/bin/env bash

curl -sSL https://install.python-poetry.org | python3 -
poetry config virtualenvs.in-project true
poetry install
# Depends on Gym 0.21 which is not supported for installation by poetry.
# Must use a workaround (found below) until updated to
# use Gymnasium (see: https://github.com/DLR-RM/stable-baselines3/pull/1327).
# TODO: install using poetry once stable-baselines3 is updated to use Gymnasium
# stable-baselines3 = "==1.7.0"
poetry run pip install stable-baselines3==1.7.0