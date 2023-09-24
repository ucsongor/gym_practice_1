import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces as sp


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size
        self.window_size = 512

        self.observation_space = sp.Dict(
            {
                "agent": sp.Box(0, size-1, shape=(2,), dtype=int),
                "target": sp.Box(0, size-1, shape=(2,0), dtype=int),
            }
        )

        self.action_space = sp.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),                                                       # go right
            1: np.array([0, 1]),                                                       # go up
            2: np.array([-1, 0]),                                                      # go left
            3: np.array([0, -1]),                                                      # go down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        def _get_obs(self):
            return {"agent": self._agent_location, "target": self._target_location}

        def _get_info(self):
            return {
                "distance": np.linalg.norm(
                    self._agent_location - self._target_location, ord=1
                )
            }
