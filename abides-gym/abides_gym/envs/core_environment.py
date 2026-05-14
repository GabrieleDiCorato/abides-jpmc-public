from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np

from abides_core import Kernel, NanosecondTime
from abides_core.generators import InterArrivalTimeGenerator
from abides_core.utils import subdict
from abides_markets.utils import config_add_agents


class AbidesGymCoreEnv(gym.Env, ABC):
    """
    Abstract class for core gym to inherit from to create usable specific ABIDES Gyms
    """

    def __init__(
        self,
        background_config_pair: tuple[Callable, dict[str, Any] | None],
        wakeup_interval_generator: InterArrivalTimeGenerator,
        state_buffer_length: int,
        first_interval: NanosecondTime | None = None,
        gymAgentConstructor=None,
    ) -> None:

        self.background_config_pair: tuple[Callable, dict[str, Any] | None] = (
            background_config_pair
        )
        if background_config_pair[1] is None:
            background_config_pair[1] = {}

        self.wakeup_interval_generator: InterArrivalTimeGenerator = (
            wakeup_interval_generator
        )
        self.first_interval = first_interval
        self.state_buffer_length: int = state_buffer_length
        self.gymAgentConstructor = gymAgentConstructor

        # Initialize random number generator
        self.np_random = None
        self._seed = None

        self.state: np.ndarray | None = None
        self.reward: float | None = None
        self.terminated: bool | None = None
        self.truncated: bool | None = None
        self.info: dict[str, Any] | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the state of the environment and returns an initial observation.

        Parameters
        ----------
        seed : int, optional
            The seed for the random number generator.
        options : dict, optional
            Additional options for reset.

        Returns
        -------
        observation : np.ndarray
            The initial observation of the space.
        info : dict
            Additional information about the reset.
        """
        # Handle seeding
        if seed is not None:
            self._seed = seed
        super().reset(seed=seed)

        # get seed to initialize random states for ABIDES
        internal_seed = self.np_random.integers(low=0, high=2**32, dtype="uint64")
        # instanciate back ground config state
        background_config_args = self.background_config_pair[1]
        background_config_args.update(
            {"seed": internal_seed, **self.extra_background_config_kvargs}
        )
        background_config_state = self.background_config_pair[0](
            **background_config_args
        )
        # instanciate gym agent and add it to config and gym object
        nextid = len(background_config_state["agents"])
        # Create random state for gym agent from the environment's random state
        gym_agent_seed = self.np_random.integers(low=0, high=2**32, dtype="uint64")
        gym_agent_random_state = np.random.RandomState(seed=gym_agent_seed)
        gym_agent = self.gymAgentConstructor(
            nextid,
            "ABM",
            first_interval=self.first_interval,
            wakeup_interval_generator=self.wakeup_interval_generator,
            state_buffer_length=self.state_buffer_length,
            random_state=gym_agent_random_state,
            **self.extra_gym_agent_kvargs,
        )
        config_state = config_add_agents(
            background_config_state, [gym_agent], gym_agent_random_state
        )
        self.gym_agent = config_state["agents"][-1]
        # KERNEL
        # instantiate the kernel object
        kernel = Kernel(
            random_state=np.random.RandomState(seed=internal_seed),
            runner_hook=self.gym_agent,
            **subdict(
                config_state,
                [
                    "start_time",
                    "stop_time",
                    "agents",
                    "agent_latency_model",
                    "default_computation_delay",
                    "oracle",
                ],
            ),
        )
        kernel.initialize()
        # kernel will run until GymAgent has to take an action
        raw_state = kernel.runner()
        state = self.raw_state_to_state(deepcopy(raw_state["result"]))
        # attach kernel
        self.kernel = kernel
        info = self.raw_state_to_info(deepcopy(raw_state["result"]))
        return state, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : Discrete

        Returns
        -------
        observation, reward, terminated, truncated, info : tuple
            observation (object) :
                an environment-specific object representing your observation of
                the environment.

            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.

            terminated (bool) :
                whether the episode has ended due to reaching a terminal state.
                (For example, the agent achieved its goal or failed definitively.)

            truncated (bool) :
                whether the episode has ended due to a time limit or other
                truncation condition outside the scope of the MDP.

            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        assert self.action_space.contains(
            action
        ), f"Action {action} is not contained in Action Space"

        abides_action = self._map_action_space_to_ABIDES_SIMULATOR_SPACE(action)

        raw_state = self.kernel.runner((self.gym_agent, abides_action))
        self.state = self.raw_state_to_state(deepcopy(raw_state["result"]))

        assert self.observation_space.contains(
            self.state
        ), f"INVALID STATE {self.state}"

        self.reward = self.raw_state_to_reward(deepcopy(raw_state["result"]))

        # In gymnasium, done is split into terminated and truncated
        # terminated: episode ended due to environment dynamics (e.g., agent failed)
        # truncated: episode ended due to time limit
        self.terminated = self.raw_state_to_done(deepcopy(raw_state["result"]))
        self.truncated = raw_state["done"] and not self.terminated

        if self.terminated or self.truncated:
            self.reward += self.raw_state_to_update_reward(
                deepcopy(raw_state["result"])
            )

        self.info = self.raw_state_to_info(deepcopy(raw_state["result"]))

        return (self.state, self.reward, self.terminated, self.truncated, self.info)

    def render(self, mode: str = "human") -> None:
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
        """
        print(self.state, self.reward, self.info)

    def close(self) -> None:
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        # kernel.termination()
        ##TODO: look at whether some cleaning functions needed for abides

    @abstractmethod
    def raw_state_to_state(self, raw_state: dict[str, Any]) -> np.ndarray:
        """
        abstract method that transforms a raw state into a state representation

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - state: state representation defining the MDP
        """
        raise NotImplementedError

    @abstractmethod
    def raw_state_to_reward(self, raw_state: dict[str, Any]) -> float:
        """
        abstract method that transforms a raw state into the reward obtained during the step

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - reward: immediate reward computed at each step
        """
        raise NotImplementedError

    @abstractmethod
    def raw_state_to_done(self, raw_state: dict[str, Any]) -> float:
        """
        abstract method that transforms a raw state into the flag if an episode is done

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - done: flag that describes if the episode is terminated or not
        """
        raise NotImplementedError

    @abstractmethod
    def raw_state_to_update_reward(self, raw_state: dict[str, Any]) -> bool:
        """
        abstract method that transforms a raw state into the final step reward update (if needed)

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - reward: update reward computed at the end of the episode
        """
        raise NotImplementedError

    @abstractmethod
    def raw_state_to_info(self, raw_state: dict[str, Any]) -> dict[str, Any]:
        """
        abstract method that transforms a raw state into an info dictionnary

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - reward: info dictionnary computed at each step
        """
        raise NotImplementedError
