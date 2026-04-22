# platoon_split_env_sumo.py
import gym
import numpy as np
from gym import spaces

class PlatoonSplitEnvSUMO(gym.Env):
    """
    Environment for RL-based AV scoring during platoon splitting.
    Action = scalar score (regression), not discrete action.
    """
    def __init__(self, traci, state_builder, platoon_ids, candidate_av_ids, threshold=0.5):
        super(PlatoonSplitEnvSUMO, self).__init__()
        self.traci = traci
        self.state_builder = state_builder
        self.platoon_ids = platoon_ids
        self.candidate_av_ids = candidate_av_ids
        self.threshold = threshold
        self.current_index = 0

        # Observation space: e.g., 8-dim vector from state builder
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        # Action space: scalar score ∈ [0, 1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self):
        self.current_index = 0
        if self.current_index >= len(self.candidate_av_ids):
            return self.observation_space.sample()

        self.av_id = self.candidate_av_ids[self.current_index]
        self.platoon_id = self.platoon_ids[0]  # optional: support multiple platoons later
        state = self.state_builder.build_state2(self.av_id, self.platoon_id)
        return state.astype(np.float32)

    def step(self, action):
        """
        action: scalar score (output by PPO)
        """
        av_id = self.av_id
        platoon_id = self.platoon_id
        score = action[0]

        reward = self.evaluate_insertion_reward(av_id, platoon_id, score)
        done = True
        info = {"av_id": av_id, "score": score}

        self.current_index += 1
        return np.zeros_like(self.observation_space.sample()), reward, done, info

    def evaluate_insertion_reward(self, av_id, platoon_id, score):
        """
        Placeholder reward function.
        Replace with your real insertion success / middle-position reward logic.
        """
        # You can integrate: traci.vehicle.getLanePosition(av_id), platoon center, etc.
        if score > self.threshold:
            return 1.0  # Acceptable score
        else:
            return -1.0  # Not acceptable

    def render(self, mode='human'):
        pass
