import pathlib

import gymnasium
import sys
sys.path.append("")
sys.modules["gym"] = gymnasium

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import supersuit
from supersuit.vector import ConcatVecEnv
import numpy as np

from common import paths
from environment import make_env, BattlesnakeDllEngine, adapt_engine_for_parallel_env, BattlesnakeEnvironmentConfiguration, Movement
from environment import ObservationToImage


def add_default_mode_to_render_args(env):
    wrapped = env.render
    def wrapper(self, mode='human'):
        return wrapped(self)
    env.render = wrapper
    return env

add_default_mode_to_render_args(ConcatVecEnv)

engine = BattlesnakeDllEngine(paths.BIN_DIR / "rules.dll")
configuration = BattlesnakeEnvironmentConfiguration(possible_agents=["agent_0", "agent_1"])
observation_transformer = ObservationToImage(configuration)
engine_adapter = adapt_engine_for_parallel_env(engine)
env = make_env(engine_adapter, observation_transformer, configuration)

env = supersuit.flatten_v0(env)
env = supersuit.pettingzoo_env_to_vec_env_v1(env)
env = supersuit.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

def combine_truncation_and_termination_into_done_in_steps(env):
    def make_wrapper(wrapped):
        def wrapper(*args, **kwargs):
            observations, rewards, terminations, truncations, infos = wrapped(*args, **kwargs)
            dones = np.maximum(terminations, truncations)
            return observations, rewards, dones, infos
        return wrapper
    env.step = make_wrapper(env.step)
    env.step_wait = make_wrapper(env.step_wait)
    return env

env = combine_truncation_and_termination_into_done_in_steps(env)
env = VecMonitor(env)


experiment_name = "ppo_duels_demo_v0"
total_timesteps = 1000000  # Try: 1000000 * 32
model_name = f"model"  # TODO: Use better naming scheme

model_file = paths.RESULTS_DIR / experiment_name / (model_name + ".zip")
tensorboard_log_dir = paths.RESULTS_DIR / experiment_name / "tensorboard"
train = True

if train:
    if pathlib.Path(model_file).exists():
        model = PPO.load(model_file)
    else:
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_log_dir)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_file)

model = PPO.load(model_file)


obs = env.reset()
dones = [False, False]
while not all(dones):
    env.render()
    actions, _ = model.predict(obs)
    obs, rewards, dones, info = env.step(actions)
    if all(dones):
        print(f"Episode finished after actions({list(Movement(action) for action in actions)}), and reward({rewards})")
