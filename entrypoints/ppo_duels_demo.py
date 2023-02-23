from stable_baselines3 import PPO
from pettingzoo.test import api_test

from common import paths
from environment.rules import Rules
from environment.envs import duels_v0


raise NotImplementedError("This file is not yet updated to the new environment")


rules = Rules(paths.BIN_DIR/"rules.dll")
env = duels_v0.env(rules)  # Create a default duels environment
api_test(env)

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
