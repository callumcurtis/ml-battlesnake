# stable-baselines 3 depends on gym, but we want to use its direct successor gymnasium
import gymnasium
import sys
sys.path.append("")
sys.modules["gym"] = gymnasium

import pathlib
import argparse

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


class Arguments:

    def __init__(
        self,
        num_agents: int,
        num_envs: int,
        initial_learning_rate: float,
        total_timesteps: int,
        train: bool,
        demo: bool,
        base_model_name: str,
    ) -> None:
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.initial_learning_rate = initial_learning_rate
        self.total_timesteps = total_timesteps
        self.train = train
        self.demo = demo
        self.base_model_name = base_model_name


class ArgumentParser:

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            prog="PPO Demo",
            description="Trains a PPO agent to play Battlesnake",
        )
        parser.add_argument(
            "--num-agents",
            type=int,
            default=4,
            help="Number of agents to train",
        )
        parser.add_argument(
            "--num-envs",
            type=int,
            default=4,
            help="Number of environments to train on",
        )
        parser.add_argument(
            "--initial-learning-rate",
            type=float,
            default=0.000003,
            help="Initial learning rate",
        )
        parser.add_argument(
            "--total-timesteps",
            type=int,
            default=3_000_000,
            help="Total number of timesteps to train for",
        )
        parser.add_argument(
            "--train",
            action="store_true",
            help="Train the model",
        )
        parser.add_argument(
            "--demo",
            action="store_true",
            help="Demo the model",
        )
        parser.add_argument(
            "--base-model-name",
            type=str,
            default="model",
            help="Base name of the model file",
        )
        self._parser = parser
    
    def parse_args(self) -> Arguments:
        args = self._parser.parse_args()
        if not args.train and not args.demo:
            self._parser.error("Either --train or --demo must be specified")
        return Arguments(
            num_agents=args.num_agents,
            num_envs=args.num_envs,
            initial_learning_rate=args.initial_learning_rate,
            total_timesteps=args.total_timesteps,
            train=args.train,
            demo=args.demo,
            base_model_name=args.base_model_name,
        )


def train(
    env,
    num_envs,
    model_file,
    tensorboard_log_dir,
    initial_learning_rate,
    total_timesteps,
):
    env = supersuit.concat_vec_envs_v1(env, num_envs, num_cpus=num_envs, base_class="stable_baselines3")
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


def demo(
    env,
    model_file,
):
    model = PPO.load(model_file, env=None)
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


def main():
    args = ArgumentParser().parse_args()

    agents = [f"agent_{i}" for i in range(args.num_agents)]
    game_type = "solo" if args.num_agents == 1 else "standard"
    configuration = BattlesnakeEnvironmentConfiguration(possible_agents=agents, game_type=game_type)
    observation_transformer = ObservationToFlattenedArray(configuration)
    engine = BattlesnakeDllEngine(paths.BIN_DIR / "rules.dll")
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

    experiment_name = f"ppo-agents({args.num_agents})"
    model_file = paths.RESULTS_DIR / experiment_name / (args.base_model_name + ".zip")
    tensorboard_log_dir = paths.RESULTS_DIR / experiment_name / "tensorboard"

    if args.train:
        train(
            env=base_env,
            num_envs=args.num_envs,
            model_file=model_file,
            tensorboard_log_dir=tensorboard_log_dir,
            initial_learning_rate=args.initial_learning_rate,
            total_timesteps=args.total_timesteps,
        )
    
    if args.demo:
        demo(
            env=base_env,
            model_file=model_file,
        )


if __name__ == "__main__":
    main()
