# state_builder.py

import numpy as np

class StateBuilder:
    """
    State vector constructor for RL-based AV lane change decision.
    Inputs: candidate AV ID and target platoon info.
    Output: normalized state vector (8-dimensional).
    """
    def __init__(self, traci, vmax=30.0, max_gap=100.0, max_lane_pos=1200.0, max_size=11):
        self.traci = traci
        self.vmax = vmax
        self.max_gap = max_gap
        self.max_lane_pos = max_lane_pos # take note this number
        self.max_size = max_size

    def build_state2(self, av_id: str, pMember, p_info: list, ls_upA) -> np.ndarray:
        """
        Build the state vector for a given candidate AV.

        Parameters:
        - av_id: candidate vehicle ID (on side lane)
        - pMember: members list of oversized platoon
        - p_info: list = [head_pos, tail_pos, avg_speed, size]

        Returns:
        - np.ndarray: normalized state vector with 10 elements
        """
        if av_id == 'mbav1274':
            pass
        try:
            # Unpack platoon info
            head_pos, tail_pos, avg_speed, size = p_info

            # AV features
            v_av = self.traci.vehicle.getSpeed(av_id)
            av_pos = self.traci.vehicle.getLanePosition(av_id)

            # veh number before and after lc_av
            positions = [(vid, self.traci.vehicle.getLanePosition(vid)) for vid in pMember]
            num_front = len([vid for vid, pos in positions if pos >= av_pos])
            num_back = len([vid for vid, pos in positions if pos < av_pos])

            # Relative position
            dis_to_tail = (tail_pos - av_pos) / self.max_gap
            dis_to_head = (head_pos - av_pos) / self.max_gap
            # Normalize relative distances with tanh to retain scale and avoid hard clipping
            dis_to_tail_norm = np.tanh(dis_to_tail)
            dis_to_head_norm = np.tanh(dis_to_head)

            # gap before and after lc_av
            front_gap, rear_gap = self.get_insertion_gap(pMember, ls_upA, av_pos)
            front_gap_norm = np.clip(front_gap/self.max_gap, 0.0, 1.0)
            rear_gap_norm = np.clip(rear_gap / self.max_gap, 0.0, 1.0)

            # Construct normalised state vector
            state = np.array([
                v_av / self.vmax,
                av_pos / self.max_lane_pos,
                dis_to_tail_norm, # Normalized to [-1, 1] range to preserve directional information
                dis_to_head_norm, # Normalized to [-1, 1] range to preserve directional information
                size / self.max_size,
                avg_speed / self.vmax,
                num_front / self.max_size,
                num_back / self.max_size,
                front_gap_norm,
                rear_gap_norm
            ], dtype=np.float32)

        except Exception as e:
            print(f"[StateBuilder] Failed to extract state for {av_id}: {e}")
            state = np.zeros(10, dtype=np.float32)

        return state

    def get_insertion_gap(self, ls_pMember, ls_upA, av_pos):
        '''
        Compute the front and rear gap for a candidate AV insertion position,
        based on platoon members and downstream vehicles on the same lane.
        :param ls_pMember: ['mav635', 'mhv666', 'mhv710', 'mhv735', 'mhv760']
        :param ls_upA: ['mhv1092', 'mhv1069', 'mhv1035', 'mhv960']
        :return:
            front_gap (float): Distance to the nearest vehicle in front of AV.
            rear_gap (float): Distance to the nearest vehicle behind AV.
        '''
        ls_front_gap = []
        ls_rear_gap = []

        leader_id = ls_pMember[0]

        # Get all vehicles from leader backward (i.e., all vehicles behind the leader in lane order)
        ls_upA_re = ls_upA[::-1]
        idx_leader = ls_upA_re.index(leader_id)
        ls_id = ls_upA_re[idx_leader:]
        for id in ls_id:
            try:
                pos = self.traci.vehicle.getLanePosition(id)
            except self.traci.TraCIException:
                continue
            gap = pos - av_pos
            if gap > 0:
                ls_front_gap.append(gap)
            else:  # gap <= 0
                ls_rear_gap.append(gap)
        # Nearest front gap = smallest positive gap
        if not ls_front_gap:
            pass
        front_gap = abs(min(ls_front_gap))
        # Nearest rear gap = smallest (least negative) → largest value
        rear_gap = abs(max(ls_rear_gap)) if ls_rear_gap else av_pos
        return front_gap, rear_gap