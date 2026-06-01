# state_builder.py

import numpy as np

class StateBuilder:
    """
    State vector constructor for RL-based AV lane change decision.
    Inputs: candidate AV ID and target platoon info.
    Output: normalized state vector (8-dimensional).
    """
    def __init__(self, traci, data_recorder, vmax=30.0, max_gap=100.0, max_lane_pos=1200.0, max_size=11):
        self.traci = traci
        self.data_recorder = data_recorder

        self.vmax = vmax
        self.max_gap = max_gap
        self.max_lane_pos = max_lane_pos # take note this number
        self.max_size = max_size

    def build_state2(self, av_id: str, pMember, p_info: list, ls_upA_asc) -> np.ndarray:
        """
        split_insert

        Build the state vector for a given candidate AV.
        Parameters:
        - av_id: candidate vehicle ID (on side lane)
        - pMember: members list of oversized platoon
        - p_info: list = [head_pos, tail_pos, avg_speed, size]

        Returns:
        - np.ndarray: normalized state vector with 10 elements
        """
        try:
            # Unpack platoon info
            head_pos, tail_pos, avg_speed, size = p_info

            # AV features
            v_av = self.data_recorder.get_vid_states(av_id)['v']
            av_pos = self.data_recorder.get_vid_states(av_id)['pos']

            # veh number before and after lc_av
            positions = [(vid, self.data_recorder.get_vid_states(vid)['pos']) for vid in pMember]
            num_front = len([vid for vid, pos in positions if pos >= av_pos])
            num_back = len([vid for vid, pos in positions if pos < av_pos])

            # Relative position
            dis_to_tail = (tail_pos - av_pos) / self.max_gap
            dis_to_head = (head_pos - av_pos) / self.max_gap
            # Normalize relative distances with tanh to retain scale and avoid hard clipping
            dis_to_tail_norm = np.tanh(dis_to_tail)
            dis_to_head_norm = np.tanh(dis_to_head)

            # gap before and after lc_av
            front_gap, rear_gap = self._get_insertion_gap(pMember, ls_upA_asc, av_pos)
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

    def build_state_free(self, cand_leader: str, target_sparse_platoon, dic_platoon_member) -> np.ndarray:
        """
        free_insert
        Build the state vector for a given candidate AV.

        Parameters:
        - cand_leader (candidate_av): candidate AV leader (on side lane)
        - target_sparse_platoon: {leader_id: first_free_follower}
        - dic_platoon_member: {leader_id: [leader_id, follower1, follower2, ...], ...}

        Returns:
        - np.ndarray: normalized state vector with 10 elements
        """
        try:
            # cand_leader(side_AV) featuress
            pos_this = self.data_recorder.get_vid_states(cand_leader)['pos'] # 1
            v_this = self.data_recorder.get_vid_states(cand_leader)['v'] # 7

            # first free follower features
            first_free_follower = next(iter(target_sparse_platoon.values()))
            pos_first_free = self.data_recorder.get_vid_states(first_free_follower)['pos']
            v_first_free = self.data_recorder.get_vid_states(first_free_follower)['v'] # 9

            p_veh_info = self.traci.vehicle.getLeader(first_free_follower)
            pv_first_free, dis_to_pv = p_veh_info if p_veh_info is not None else (None, 0.0)

            # platoon leader and followers
            leader = next(iter(target_sparse_platoon.keys()))
            ls_follower = dic_platoon_member[leader]

            # Build list of free followers starting from the first free follower (if present in ls_follower)
            try:
                idx_first_free = ls_follower.index(first_free_follower)
                free_followers = ls_follower[idx_first_free:]  # include the first free and any after it
            except ValueError:
                free_followers = [first_free_follower]

            last_free_follower = free_followers[-1]
            pos_last_free = self.data_recorder.get_vid_states(last_free_follower)['pos']

            # Get positions for free followers and count relative to candidate AV
            free_positions = []
            for fid in free_followers:
                try:
                    free_positions.append(self.data_recorder.get_vid_states(fid)['pos'])
                except Exception:
                    continue

            num_free = len(free_positions) # 10
            # pos > pos_this
            num_free_ahead = len([p for p in free_positions if p >= pos_this]) # 5
            # pos < pos_this
            num_free_behind = len([p for p in free_positions if p < pos_this]) # 6

            # pv of first free follower features
            pos_pv = self.data_recorder.get_vid_states(pv_first_free)['pos']
            v_pv = self.data_recorder.get_vid_states(pv_first_free)['v'] # 8

            # relative_pos
            dis_this_to_first_free = np.tanh((pos_first_free - pos_this)/self.max_gap) # 2
            dis_this_to_last_free = np.tanh((pos_last_free - pos_this)/self.max_gap) # 3
            dis_first_free_to_pv = np.tanh(dis_to_pv/self.max_gap) # 4 ?

            # Construct normalised state vector
            state = np.array([
                pos_this / self.max_lane_pos,  # 1. AV absolute position [0, 1]
                dis_this_to_first_free,  # 2. Distance to first free [-1, 1]
                dis_this_to_last_free,  # 3. Distance to last free [-1, 1]
                dis_first_free_to_pv,  # 4. Gap before first free [-1, 1]
                num_free_ahead / self.max_size,  # 5. Free followers ahead [0, 1]
                num_free_behind / self.max_size,  # 6. Free followers behind [0, 1]
                v_this / self.vmax,  # 7. AV velocity [0, 1]
                v_pv / self.vmax,  # 8. Leader velocity [0, 1]
                v_first_free / self.vmax,  # 9. First free velocity [0, 1]
                num_free / self.max_size  # 10. Total free followers [0, 1]
            ], dtype=np.float32)

        except Exception as e:
            print(f"[StateBuilder] Failed to extract state for {cand_leader}: {e}")
            state = np.zeros(10, dtype=np.float32)

        return state

    def _get_insertion_gap(self, ls_pMember, ls_upA_asc, av_pos):
        '''
        Compute the front and rear gap for a candidate AV insertion position,
        based on platoon members and downstream vehicles on the same lane.
        :param ls_pMember: ['mav635', 'mhv666', 'mhv710', 'mhv735', 'mhv760']
        :param ls_upA: ['mhv960', 'mhv1069', 'mhv1092']
        :return:
            front_gap (float): Distance to the nearest vehicle in front of AV.
            rear_gap (float): Distance to the nearest vehicle behind AV.
        '''
        ls_front_gap = []
        ls_rear_gap = []

        leader_id = ls_pMember[0]

        # Get all vehicles from leader backward (i.e., all vehicles behind the leader in lane order)
        idx_leader = ls_upA_asc.index(leader_id)
        ls_id = ls_upA_asc[idx_leader:]
        for id in ls_id:
            # try:
            #     pos = self.traci.vehicle.getLanePosition(id)
            # except self.traci.TraCIException:
            #     continue
            # pos = self.data_recorder.dic_pos.get(id)
            pos = self.data_recorder.get_vid_states(id)['pos']
            if pos is None:
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