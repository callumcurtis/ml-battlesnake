# stable-baselines 3 depends on gym, but we want to use its direct successor gymnasium
import gymnasium
import sys
sys.path.append("")
sys.modules["gym"] = gymnasium

import pathlib
from typing import Optional
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
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
    RewardChain,
    RewardWinLoseDraw,
    RewardSurvival,
    RewardOpponentDeath,
    RewardFoodConsumption,
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
        tensorboard_log_dir: pathlib.Path,
        checkpoint_period: int,
        train: bool,
        demo: bool,
        model_output_path: Optional[pathlib.Path],
        model_input_path: Optional[pathlib.Path],
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
            "--tensorboard-log-dir",
            type=str,
            help="Path to save tensorboard logs to, defaults to the same directory as the model output",
        )
        parser.add_argument(
            "--checkpoint-period",
            type=int,
            default=1_000_000,
            help="Number of timesteps between checkpoints",
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
        if args.train and args.tensorboard_log_dir is None:
            args.tensorboard_log_dir = model_output_path.parent
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
        )


class CheckpointForRecoveryCallback(BaseCallback):

    def __init__(
        self,
        save_period: int,
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
        if self.num_timesteps_since_last_checkpoint >= self._save_period:
            next_checkpoint_timesteps = self.num_timesteps_across_restarts
            self.model.save(self._checkpoint_path(next_checkpoint_timesteps))
            self.num_checkpointed_timesteps = next_checkpoint_timesteps
            self.num_timesteps_since_last_checkpoint = 0
        return True
    
    def on_restart(self):
        self.num_timesteps_since_last_checkpoint = 0


def train(
    base_env,
    num_envs,
    model_input_path: Optional[pathlib.Path],
    model_output_path: pathlib.Path,
    tensorboard_log_dir: pathlib.Path,
    initial_learning_rate: float,
    checkpoint_period: int,
    total_timesteps: int,
):
    recovery_callback = CheckpointForRecoveryCallback(
        save_period=checkpoint_period,
        save_path=pathlib.Path(model_output_path).parent,
        name_prefix=model_output_path.stem,
    )

    model_load_path = model_input_path    

    while recovery_callback.num_timesteps_across_restarts < total_timesteps:
        env = supersuit.concat_vec_envs_v1(base_env, num_envs, num_cpus=num_envs, base_class="stable_baselines3")
        env = combine_truncation_and_termination_into_done_in_steps(env)
        env = VecMonitor(env)
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            tensorboard_log=tensorboard_log_dir,
            learning_rate=make_logarithmic_learning_rate_schedule(initial_learning_rate),
        )
        if model_load_path is not None:
            model.set_parameters(model_load_path, exact_match=True)
        remaining_timesteps = total_timesteps - recovery_callback.num_timesteps_across_restarts
        try:
            model.learn(total_timesteps=remaining_timesteps, callback=recovery_callback)
            model.save(model_output_path)
        except (OSError, EOFError):
            recovery_callback.on_restart()
            model_load_path = recovery_callback.last_checkpoint_path()
        model.env.close()


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