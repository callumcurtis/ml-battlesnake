"""Example of how to train a PPO agent to play Battlesnake using the ml-battlesnake library."""

# stable-baselines 3 depends on gym, but we want to use its direct successor gymnasium
# TODO: remove this workaround once stable-baselines 3 is updated to use gymnasium
# see: https://github.com/DLR-RM/stable-baselines3/pull/1327
import gymnasium
import sys
sys.modules["gym"] = gymnasium

import pathlib
from typing import Optional
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import supersuit
import numpy as np
import psutil

from ml_battlesnake.common import paths
from ml_battlesnake.learning.environment import (
    make_gymnasium_vec_env,
    BattlesnakeDllEngine,
    adapt_engine_for_parallel_env,
    BattlesnakeEnvironmentConfiguration,
    Movement,
    ObservationToFlattenedArray,
    MemoryBuffer,
    RewardChain,
    RewardWinLoseDraw,
    RewardSurvival,
    RewardOpponentDeath,
    RewardFoodConsumption,
)


def combine_truncation_and_termination_into_done_in_steps(env):
    """Combine the truncation and termination arrays into a single done array in the step method of the given environment.

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


def make_logarithmic_learning_rate_schedule(
    initial_learning_rate: float,
    initial_progress: float,
):
    progress_remaining_at_start = 1.0 - initial_progress
    def schedule(progress_remaining: float) -> float:
        return initial_learning_rate * 0.1 / (1.1 - progress_remaining * progress_remaining_at_start)
    return schedule


class Arguments:

    def __init__(
        self,
        num_agents: int,
        num_envs: int,
        initial_learning_rate: float,
        total_timesteps: int,
        tensorboard_log_dir: pathlib.Path,
        checkpoint_period: Optional[int],
        train: bool,
        demo: bool,
        model_output_path: Optional[pathlib.Path],
        model_input_path: Optional[pathlib.Path],
        gamma: float,
    ) -> None:
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.initial_learning_rate = initial_learning_rate
        self.total_timesteps = total_timesteps
        self.tensorboard_log_dir = tensorboard_log_dir
        self.checkpoint_period = checkpoint_period
        self.train = train
        self.demo = demo
        self.model_output_path = model_output_path
        self.model_input_path = model_input_path
        self.gamma = gamma


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
            "--gamma",
            type=float,
            default=0.977,
            help="Discount factor",
        )
        parser.add_argument(
            "--total-timesteps",
            type=int,
            default=3_000_000,
            help="Total number of timesteps to train for",
        )
        parser.add_argument(
            "--tensorboard-log-dir",
            type=str,
            help="Path to save tensorboard logs to, defaults to the same directory as the model output",
        )
        parser.add_argument(
            "--checkpoint-period",
            type=int,
            default=-1,
            help="Checkpoint the model during training every n timesteps, defaults to -1 (no checkpointing)",
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
            "--model-output-path",
            type=str,
            help="Path to save the model to",
        )
        parser.add_argument(
            "--model-input-path",
            type=str,
            help="Path to load the model from",
        )
        self._parser = parser
    
    def parse_args(self) -> Arguments:
        args = self._parser.parse_args()
        if not args.train and not args.demo:
            self._parser.error("Either --train or --demo must be specified")
        model_output_path = pathlib.Path(args.model_output_path) if args.model_output_path else None
        model_input_path = pathlib.Path(args.model_input_path) if args.model_input_path else None
        if args.train and model_output_path is None:
            self._parser.error("To train, model output path must be specified")
        if args.train and model_output_path.exists():
            self._parser.error(f"Model output path {model_output_path} already exists")
        if args.train and model_output_path.suffix != ".zip":
            self._parser.error(f"Model output path {model_output_path} must have .zip extension")
        if args.demo and not args.train and model_input_path is None:
            self._parser.error("To demo, model input path must be specified or --train must be specified")
        if args.demo and not args.train and not model_input_path.exists():
            self._parser.error(f"Model input path {model_input_path} does not exist")
        if args.demo and not args.train and model_input_path.suffix != ".zip":
            self._parser.error(f"Model input path {model_input_path} must have .zip extension")
        if not (0 <= args.gamma <= 1):
            self._parser.error(f"Gamma must be between 0 and 1, received {args.gamma}")
        if args.train and args.tensorboard_log_dir is None:
            args.tensorboard_log_dir = model_output_path.parent
        if args.checkpoint_period < 1 and args.checkpoint_period != -1:
            self._parser.error(f"Checkpoint period must be -1 (no checkpoints) or >= 1, received {args.checkpoint_period}")
        if args.checkpoint_period == -1:
            args.checkpoint_period = None
        return Arguments(
            num_agents=args.num_agents,
            num_envs=args.num_envs,
            initial_learning_rate=args.initial_learning_rate,
            total_timesteps=args.total_timesteps,
            tensorboard_log_dir=args.tensorboard_log_dir,
            checkpoint_period=args.checkpoint_period,
            train=args.train,
            demo=args.demo,
            model_output_path=model_output_path,
            model_input_path=model_input_path,
            gamma=args.gamma,
        )


class CheckpointCallback(BaseCallback):

    def __init__(
        self,
        save_period: Optional[int],
        save_path: pathlib.Path,
        name_prefix: str,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._save_period = save_period
        self._save_path = save_path
        self._name_prefix = name_prefix

        self.num_checkpointed_timesteps = 0
        self.num_timesteps_since_last_checkpoint = 0

    @property
    def num_timesteps_across_restarts(self) -> int:
        return self.num_checkpointed_timesteps + self.num_timesteps_since_last_checkpoint

    def _init_callback(self) -> None:
        self._save_path.mkdir(exist_ok=True)
        self.num_timesteps_since_last_checkpoint = 0

    def last_checkpoint_path(self) -> Optional[pathlib.Path]:
        if self.num_checkpointed_timesteps == 0:
            return
        return self._checkpoint_path(self.num_checkpointed_timesteps)
    
    def _checkpoint_path(self, num_timesteps: int) -> str:
        return self._save_path / f"{self._name_prefix}_timesteps({num_timesteps}).zip"

    def _on_step(self) -> bool:
        self.num_timesteps_since_last_checkpoint += self.model.n_envs
        if self._save_period and self.num_timesteps_since_last_checkpoint >= self._save_period:
            next_checkpoint_timesteps = self.num_timesteps_across_restarts
            self.model.save(self._checkpoint_path(next_checkpoint_timesteps))
            self.num_checkpointed_timesteps = next_checkpoint_timesteps
            self.num_timesteps_since_last_checkpoint = 0
        return True
    
    def on_restart(self):
        self.num_timesteps_since_last_checkpoint = 0


class StopTrainingOnComputeResourceThreshold(BaseCallback):

    def __init__(
        self,
        available_memory_threshold: int = 2**31,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._available_memory_threshold = available_memory_threshold
        self.did_stop = False
    
    def _on_step(self) -> bool:
        stop = psutil.virtual_memory().available < self._available_memory_threshold
        if stop:
            self.did_stop = True
        return not stop


def train(
    base_env,
    num_envs,
    model_input_path: Optional[pathlib.Path],
    model_output_path: pathlib.Path,
    tensorboard_log_dir: pathlib.Path,
    initial_learning_rate: float,
    gamma: float,
    checkpoint_period: Optional[int],
    total_timesteps: int,
):
    checkpoint_callback = CheckpointCallback(
        save_period=checkpoint_period,
        save_path=pathlib.Path(model_output_path).parent,
        name_prefix=model_output_path.stem,
    )

    model_load_path = model_input_path    

    remaining_timesteps = lambda: total_timesteps - checkpoint_callback.num_timesteps_across_restarts

    while remaining_timesteps():
        env = supersuit.concat_vec_envs_v1(base_env, num_envs, num_cpus=num_envs, base_class="stable_baselines3")
        env = combine_truncation_and_termination_into_done_in_steps(env)
        env = VecMonitor(env)
        learning_rate = make_logarithmic_learning_rate_schedule(
            initial_learning_rate=initial_learning_rate,
            initial_progress=checkpoint_callback.num_timesteps_across_restarts / total_timesteps,
        )
        resource_monitor_callback = StopTrainingOnComputeResourceThreshold()
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            tensorboard_log=tensorboard_log_dir,
            learning_rate=learning_rate,
            gamma=gamma,
            policy_kwargs={"net_arch": {"pi": [90, 90], "vf": [90, 90]}}
        )
        if model_load_path is not None:
            model.set_parameters(model_load_path, exact_match=True)
        try:
            model.learn(total_timesteps=remaining_timesteps(), callback=[checkpoint_callback, resource_monitor_callback])
            model.save(model_output_path)
        except (OSError, EOFError):
            pass
        if remaining_timesteps():
            checkpoint_callback.on_restart()
            model_load_path = cp if (cp := checkpoint_callback.last_checkpoint_path()) else model_input_path
        model.env.close()


def demo(
    env,
    model_file,
):
    model = PPO.load(model_file, env=None)
    obs = env.reset()
    while True:
        env.render()
        actions, _ = model.predict(obs)
        obs, rewards, terminations, truncations, info = env.step(actions)
        print(f"Actions: {[Movement(a) for a in actions]}, rewards: {rewards}")
        if (terminations | truncations).all():
            print(f"Game over")
            break
    env.close()


def main():
    args = ArgumentParser().parse_args()

    agents = [f"agent_{i}" for i in range(args.num_agents)]
    game_type = "solo" if args.num_agents == 1 else "standard"
    configuration = BattlesnakeEnvironmentConfiguration(possible_agents=agents, game_type=game_type)
    observation_transformer = ObservationToFlattenedArray(configuration)
    engine = BattlesnakeDllEngine(paths.BIN_DIR / "engine.dll")
    engine_adapter = adapt_engine_for_parallel_env(engine)
    reward_function = RewardChain(
        [
            RewardWinLoseDraw(),
            RewardSurvival(),
            RewardOpponentDeath(),
            RewardFoodConsumption(),
        ]
    )
    memory_buffer = MemoryBuffer(reward_function.required_memory_size)

    base_env = make_gymnasium_vec_env(
        engine_adapter,
        observation_transformer,
        reward_function,
        memory_buffer,
        configuration,
    )

    if args.train:
        train(
            base_env=base_env,
            num_envs=args.num_envs,
            model_input_path=args.model_input_path,
            model_output_path=args.model_output_path,
            tensorboard_log_dir=args.tensorboard_log_dir,
            initial_learning_rate=args.initial_learning_rate,
            gamma=args.gamma,
            checkpoint_period=args.checkpoint_period,
            total_timesteps=args.total_timesteps,
        )
    
    if args.demo:
        model_file = args.model_output_path if args.train else args.model_input_path
        demo(
            env=base_env,
            model_file=model_file,
        )


if __name__ == "__main__":
    main()
