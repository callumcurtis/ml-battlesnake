# stable-baselines 3 depends on gym, but we want to use its direct successor gymnasium
import gymnasium
import sys
sys.path.append("")
sys.modules["gym"] = gymnasium

import pathlib

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import supersuit
import numpy as np

from common import paths
from environment import (
    make_gymnasium_vec_env,
    BattlesnakeDllEngine,
    adapt_engine_for_parallel_env,
    BattlesnakeEnvironmentConfiguration,
    Movement,
    ObservationToFlattenedArray,
    MemoryBuffer,
    RewardWinLoseDrawSurvival
)


def combine_truncation_and_termination_into_done_in_steps(env):
    """
    Combine the truncation and termination arrays into a single done array in the step method of the given environment.

    stable-baselines 3 is using the step method signature from gymnasium < 0.26.0,
    meaning that it expects the wrapped step method to return a tuple of 4 elements
    instead of 5 elements. This is a workaround to combine the truncation and termination
    arrays into a single done array in the wrapped step method.

    TODO: Create a wrapper class dedicated to this workaround.
    TODO: Remove this workaround once stable-baselines 3 is updated to use gymnasium >= 0.26.0
    """
    def make_wrapper(wrapped):
        def wrapper(*args, **kwargs):
            result = wrapped(*args, **kwargs)
            if len(result) == 4:
                return result
            observations, rewards, terminations, truncations, infos = result
            dones = np.maximum(terminations, truncations)
            return observations, rewards, dones, infos
        return wrapper
    env.step = make_wrapper(env.step)
    env.step_wait = make_wrapper(env.step_wait)
    return env


def make_logarithmic_learning_rate_schedule(initial: float):
    def schedule(progress_remaining: float) -> float:
        return initial * 0.1 / (1.1 - progress_remaining)
    return schedule


def main():
    num_agents = 4
    num_envs = 4
    # Anecdotally: agents(1)=0.0003, agents(4)=0.000003
    initial_learning_rate = 0.000003
    total_timesteps = 3_000_000
    train = True
    base_model_name = "model"

    agents = [f"agent_{i}" for i in range(num_agents)]
    game_type = "solo" if num_agents == 1 else "standard"
    engine = BattlesnakeDllEngine(paths.BIN_DIR / "rules.dll")
    configuration = BattlesnakeEnvironmentConfiguration(possible_agents=agents, game_type=game_type)
    observation_transformer = ObservationToFlattenedArray(configuration)
    engine_adapter = adapt_engine_for_parallel_env(engine)
    reward_function = RewardWinLoseDrawSurvival()
    memory_buffer = MemoryBuffer(0)

    base_env = make_gymnasium_vec_env(
        engine_adapter,
        observation_transformer,
        reward_function,
        memory_buffer,
        configuration,
    )

    experiment_name = f"ppo-agents({num_agents})"
    model_file = paths.RESULTS_DIR / experiment_name / (base_model_name + ".zip")
    tensorboard_log_dir = paths.RESULTS_DIR / experiment_name / "tensorboard"

    if train:
        env = supersuit.concat_vec_envs_v1(base_env, num_envs, num_cpus=num_envs, base_class="stable_baselines3")
        env = combine_truncation_and_termination_into_done_in_steps(env)
        env = VecMonitor(env)
        if pathlib.Path(model_file).exists():
            model = PPO.load(model_file, env, verbose=1, tensorboard_log=tensorboard_log_dir)
        else:
            model = PPO(
                'MlpPolicy',
                env,
                verbose=1,
                tensorboard_log=tensorboard_log_dir,
                learning_rate=make_logarithmic_learning_rate_schedule(initial_learning_rate),
            )
        model.learn(total_timesteps=total_timesteps)
        model.save(model_file)
        env.close()

    model = PPO.load(model_file, env=None)
    env = base_env
    obs = env.reset()
    dones = [False, False]
    while not all(dones):
        env.render()
        actions, _ = model.predict(obs)
        obs, rewards, terminations, truncations, info = env.step(actions)
        if (terminations | truncations).all():
            print(f"Episode finished after actions({list(Movement(action) for action in actions)}), and reward({rewards})")
            break
    env.close()


if __name__ == "__main__":
    main()
