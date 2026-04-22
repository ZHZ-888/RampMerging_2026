import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class PlatoonSplitEnv(gym.Env):
    """
    A simplified environment for training RL to decide which AV should split a long platoon.
    This mockup version randomly generates platoon and AV positions.
    """

    def __init__(self):
        super().__init__()

        # === Observation space: 8-dim state (same as your state vector)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(8,), dtype=np.float32)

        # === Action space: score (float between 0~1) for single AV
        # In training we use dummy action = 0 (score not used)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Internal state
        self.state = None
        self.step_count = 0
        self.max_steps = 50

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # === Generate random plausible state vector
        self.state = np.array([
            np.clip(np.random.normal(0.8, 0.1), 0, 1),   # AV speed
            np.random.rand(),                            # AV lane position
            np.random.uniform(-4, 1),                    # gap_tail
            np.random.uniform(0, 4),                     # gap_head
            np.random.uniform(1.1, 1.4),                 # platoon size (scaled)
            np.clip(np.random.normal(0.7, 0.1), 0, 1),   # avg speed
            np.random.uniform(0.3, 0.9),                 # num_front / max_size
            np.random.uniform(0.2, 0.9),                 # num_back / max_size
        ], dtype=np.float32)

        self.step_count = 0
        return self.state, {}

    def step(self, action):
        """
        In this mock env, we reward the agent based on whether it 'chose' a plausible split.
        Later, you will replace this with your real SUMO-integrated env.
        """
        self.step_count += 1

        # Simple reward rule: reward higher if front/back both within [0.3, 0.7]
        front_ratio = self.state[6]
        back_ratio = self.state[7]

        valid_split = (front_ratio <= 1.0) and (back_ratio <= 1.0) \
                      and (front_ratio >= 0.3) and (back_ratio >= 0.3)

        reward = 1.0 if valid_split else -1.0

        terminated = False
        truncated = self.step_count >= self.max_steps

        # Next state (randomized again)
        self.state = self.reset()[0]

        return self.state, reward, terminated, truncated, {}

    def render(self, mode='human'):
        print("Current state:", self.state)
