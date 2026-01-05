# policy_wrapper.py
import sys
sys.path.append('/home/zzha/PycharmProjects/RampMerging4_250208/PlatoonSplit_RL_model')

from stable_baselines3 import PPO
from state_builder import StateBuilder

class PolicyWrapper:
    """
    RL-based AV selector for platoon split.
    Loads trained PPO model and selects best side-lane AV based on state score.
    """

    def __init__(self, model_path, traci, threshold=0.5):
        """
        :param model_path: path to trained RL model (e.g. .zip)
        :param traci: active SUMO Traci instance
        :param threshold: minimum score required to trigger lane change
        """
        self.model = PPO.load(model_path)
        self.traci = traci
        self.threshold = threshold
        self.state_builder = StateBuilder(traci)

    def select_best_av(self, candidate_avs, platoon_member_ids, platoon_info):
        """
        Select the best AV from candidates using policy score.

        :param candidate_avs: list of AV IDs on the side lane
        :param platoon_member_ids: list of vehicle IDs in the target platoon
        :param platoon_info: [head_pos, tail_pos, avg_speed, size]
        :return: (selected_av_id or None, score)
        """
        best_score = -float('inf')
        best_av = None

        for av_id in candidate_avs:
            state = self.state_builder.build_state(av_id, platoon_member_ids, platoon_info)
            action, _ = self.model.predict(state, deterministic=True)
            score = action  # Output is a scalar score

            if score > best_score:
                best_score = score
                best_av = av_id

        if best_score >= self.threshold:
            return best_av, best_score
        else:
            return None, best_score

