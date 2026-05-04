# split_insert_agent_handler.py

import os
from datetime import datetime

from rl_model.rl_module import RLScoringAgent

current_dir = os.path.dirname(os.path.abspath(__file__)) # Get the absolute path of the current script's directory
project_root = os.path.dirname(current_dir) # Get the parent directory as the project root

# model_name = 'split_score_model_251124_1900.pt'
# path_pt = os.path.join(project_root, 'rl_model', 'saved_models', model_name)

class AgentHandler:
    def __init__(self, traci, data_recorder, scoring_interval=10, mode='train',
                 exp_name='default_run', lr=5e-4, model_name=None):
        self.traci = traci
        self.data_recorder = data_recorder
        self.mode = mode

        active_exp_name = exp_name if mode == 'train' else f"EVAL_{exp_name}"

        if model_name: # 'sa_i8_0.0005'
            path_pt = os.path.join(project_root, 'rl_model', 'saved_models', 'sa_5732367', model_name)
        else:
            default_model = 'split_score_model_251124_1900.pt'
            path_pt = os.path.join(project_root, 'rl_model', 'saved_models', default_model)
        self.agent = RLScoringAgent(traci, data_recorder,
                                    exp_name=active_exp_name,
                                    model_path=path_pt if mode == "predict" else None,
                                    lr=lr)

        self.scoring_interval = scoring_interval  # Minimum interval (in steps) between scoring attempts
        self.training_warmup_steps = 20000
        self.next_save_step = 10000
        self.collected = 0  # Track how many transitions have been collected
        self.target_lane = 0

        self.ls_splited_platoon = [] # splited platoon leader
        self.insert_buffer = []
        self.last_score_step = {} # Records last scoring step for each leader_id (cooldown control)
        self.ls_score = []
        self.dic_insertedAV = {} # plan taking action but may still in process
        # dic_insertedAV = {AV_id: type, ...} record promotedAV and its type; here type = 'split'
        self.dic_score_reward = {} # record lc_av and [score, reward], dic = {lc_av:[score, reward]}

        self.last_update_payload_pair = None # {target leader: candidate leader}
        self.payload = None

    def run_agent_decision_ori(self, step, laneChange_buffer, dic_platoon_members, dic_oversized_platoon_states,
                           dic_leader_candidates, ls_upA, gating_value=None):
        """
        Main function to handle AV insertion decisions.
        """
        if not dic_oversized_platoon_states:
            return {}
        for oversize_leader in dic_oversized_platoon_states:
            if oversize_leader in self.ls_splited_platoon or oversize_leader not in dic_leader_candidates:
                continue
            selected_av, selected_state, score = (
                self._evaluate_candidates(step, oversize_leader, dic_platoon_members,
                                          dic_leader_candidates, dic_oversized_platoon_states,
                                          ls_upA, gating_value))
            if selected_av:
                payload = (oversize_leader, selected_av, selected_state, dic_platoon_members, score)
                laneChange_buffer.push(step, payload)  # Add to buffer for delayed execution
                delayed_payload = laneChange_buffer.maybe_release(step)

                self._execute_insertion(step, oversize_leader, selected_av, selected_state, dic_platoon_members, score)

        return self.dic_insertedAV

    def run_agent_decision(self, step, dic_platoon_members, dic_oversized_platoon_states,
                           dic_leader_candidates, ls_upA, gating_value=None):
        """
        Main function to handle AV insertion decisions.
        """
        self.payload = None
        if not dic_oversized_platoon_states:
            return {}

        for oversize_leader in dic_oversized_platoon_states:
            if oversize_leader in self.ls_splited_platoon or oversize_leader not in dic_leader_candidates:
                continue

            selected_av, selected_state, score = (
                self._evaluate_candidates(step, oversize_leader, dic_platoon_members,
                                          dic_leader_candidates, dic_oversized_platoon_states,
                                          ls_upA, gating_value))

            if selected_av and (self.last_update_payload_pair != {oversize_leader:selected_av}):
                self.payload = (oversize_leader, selected_av, selected_state, dic_platoon_members, score)
                self.last_update_payload_pair = {oversize_leader: selected_av}

        # self._release_insertion(step, payload, laneChange_buffer)
        return self.dic_insertedAV

    def release_insertion(self, step, laneChange_buffer):
        payload = self.payload
        if laneChange_buffer: # loss_rate != 0
            if payload:
                laneChange_buffer.push(step, payload)  # Add to buffer for delayed execution
                self.payload = None
            delayed_payload = laneChange_buffer.maybe_release(step)
            if delayed_payload:
                self._execute_insertion(step, *delayed_payload)
        else:
            if payload:
                self._execute_insertion(step, *payload)
                self.payload = None


    def update_reward(self, current_step, st, dic_platoon_members, train_interval):
        '''
        Check insert_buffer and issue rewards if leader has exited control zone.
        '''
        updated = False
        # iterate over a shallow copy to safely remove items during loop
        for record in self.insert_buffer[:]:
            leader_id = record['leader_id']
            try:
                lane_id = self.traci.vehicle.getLaneID(leader_id)
            except self.traci.TraCIException:
                lane_id = None

            if lane_id == 'ws_1': # inflow_highway_0
                lc_av = record['av_id']
                platoon_snapshot = record["platoon_snapshot"]
                reward = self.evaluate_insertion_reward(lc_av, platoon_snapshot, dic_platoon_members)
                self.dic_score_reward[lc_av].append(reward)
                if self.mode == 'train':
                    self.agent.record_transition(record['state'], reward)
                print(f"[SplitInsert] {lc_av} reward: {'+' if reward > 0 else ''}{reward:.3f} ")
                self.insert_buffer.remove(record)
                updated = True
        if updated and self.mode == 'train':
            self.collected += 1
            if self.collected >= train_interval: # update interval/2 = batch_size
                self.agent.log_training_metrics(current_step)  # log performance metrics BEFORE updating model
                self.agent.train_on_recorded(current_step, epochs=5, batch_size=int(train_interval/2))
                self.collected = 0
                self._save_model_if_needed(current_step, st)
        return self.dic_score_reward

    def evaluate_insertion_reward(self, lc_av, platoon_snapshot, dic_platoon_members):
        """
        Evaluate insertion success and return a reward score for split_insert scenario.

        REWARD SETTINGS SUMMARY (Split Agent):
        ========================================

        1. FAILURE PENALTIES (return -0.1):
           - Wrong lane: AV not on 'inflow_highway_0' (inner lane)
           - Missed insertion: AV position < platoon tail position (didn't cut in)
           - Invalid split: AV inserted at head/tail only (num_front < 1 or num_rear < 1)
           - TraCI exception: Vehicle not found or simulation error

        2. SUCCESS REWARDS (return value in (0, 1]):
           Formula: reward = 1.0 - imbalance_ratio
           where: imbalance_ratio = |num_front - num_rear| / (total_vehicles - 1)

           Reward Scale Examples:
           - Perfect balance (5 front, 5 rear): reward = 1.0
           - Moderate imbalance (6 front, 4 rear): reward ≈ 0.8
           - High imbalance (10 front, 2 rear): reward ≈ 0.27

           Goal: Encourage splits near the middle of oversized platoons to create
                 two balanced sub-platoons for optimal throughput.

        3. EVALUATION CASES:
           Case 1: AV becomes new leader with followers (dic_platoon_members[lc_av] exists)
                   - Count rear: all followers in new platoon (len(new_platoon) - 1)
                   - Count front: position of split_anchor_id in original platoon snapshot

           Case 2: AV not yet recognized as leader (transitional state)
                   - Count by position: compare lanePosition of all platoon members
                   - num_front starts at 1 (original leader), num_rear starts at 0

        Args:
            lc_av: ID of the inserted autonomous vehicle.
            platoon_snapshot: oversized platoon members at the moment of insertion decision
            dic_platoon_members: dict mapping leader ID to list of current platoon member IDs.

        Returns:
            float: Reward in [-0.1, 1.0] range
                  -0.1 = failure (wrong lane, missed insertion, invalid split)
                  (0, 1.0] = success (quality based on split balance)
        """
        try:
            leader_av = platoon_snapshot[0]
            tail_id = platoon_snapshot[-1]

            current_laneID = self.traci.vehicle.getLaneID(lc_av)
            this_pos = self.traci.vehicle.getLanePosition(lc_av)
            tail_pos = self.traci.vehicle.getLanePosition(tail_id)
            if current_laneID != 'inflow_highway_0': # upstream_0
                # AV must be on inflow_highway_0 (upstream_0)'
                return -0.1
            if this_pos < tail_pos:
                # lc_AV didn't cut into target oversized platoon
                return -0.1

            if lc_av in dic_platoon_members and len(dic_platoon_members[lc_av]) > 1:
                # Case1: when lc_av becomes a new leader or has followers (a valid split)
                new_platoon = dic_platoon_members[lc_av]
                num_rear = len(new_platoon)-1 # all followers
                # Try to find the first AV in new_platoon that also appears in platoon_snapshot
                split_anchor_id = None
                for vid in new_platoon[1:]:
                    if vid in platoon_snapshot:
                        split_anchor_id = vid
                        break
                # Use its position in platoon_snapshot as the split index
                if split_anchor_id:
                    num_front = platoon_snapshot.index(split_anchor_id)
                else:
                    num_front = 0  # fallback: no match found
            else:
                # Case2: lc_av currently not a leader, or its new platoon has no followers
                num_front = 1 # add av_leader
                num_rear = 0
                for id in platoon_snapshot[1:]: # exclude original leader
                    if id == lc_av:
                        continue # skip self
                    pos = self.traci.vehicle.getLanePosition(id)
                    if pos > this_pos:
                        num_front += 1
                    else: # pos <= current_pos
                        num_rear += 1

            if num_front >= 1 and num_rear >= 1:
                imbalance = abs(num_front - num_rear)
                total = num_front + num_rear + 1
                imbalance_ratio = imbalance / (total - 1)
                reward = 1.0 - imbalance_ratio
                return reward
            else:
                return -0.1
        except self.traci.TraCIException:
            return -0.1

    def record_loss(self, current_step, st):
        if current_step != st*10-1 or self.mode != 'train':
            return
        self.agent.record_plot_loss()  # plot loss curve

    def record_scores(self, current_step, st):
        if current_step != st*10-1 or self.mode != 'train':
            return
        self.agent.record_plot_scores(self.ls_score)

    def _save_model_if_needed(self, current_step, st):
        """
        Periodically save the trained model.
        """
        save_interval = 30000  # every 10k steps
        if current_step > self.next_save_step or current_step == st*10-1:
            # os.makedirs("saved_models", exist_ok=True)
            timestamp = datetime.now().strftime("%y%m%d_%H%M")  # e.g. 250517_1915
            filename = f"split_score_model_{timestamp}.pt"
            # full_path = os.path.join("/home/zzha/PycharmProjects/RampMerging4_250208/platoon_split_rl_model/saved_models", filename)
            self.agent.save_model(filename)
            print(f"[Model] Auto-saved at step {current_step}")
            self.next_save_step += save_interval  # set next checkpoint

    def _evaluate_candidates(self, step, leader_id, dic_platoon_members, dic_leader_candidates,
                             dic_oversized_platoon_states, ls_upA, gating_value):
        """
        Evaluate candidate AVs for a given leader and select the one with the highest score.
        """
        last_step = self.last_score_step.get(leader_id, -999)
        if step - last_step < self.scoring_interval:
            return None, None, None
        self.last_score_step[leader_id] = step

        ls_candidateAV = dic_leader_candidates[leader_id]
        if not ls_candidateAV:
            return None, None, None

        best_score = -float('inf')
        selected_av = None
        selected_state = None
        pMember = dic_platoon_members[leader_id]
        platoon_states = dic_oversized_platoon_states[leader_id]

        for av_id in ls_candidateAV:
            state = self.agent.state_builder.build_state2(av_id, pMember, platoon_states, ls_upA)
            score = self.agent.predict_score(state)
            if score > best_score:
                best_score = score
                selected_av = av_id
                selected_state = state

        if gating_value is not None and best_score < gating_value:
            return None, None, best_score
        self.dic_score_reward[selected_av] = [score]
        self.ls_score.append(score)
        return selected_av, selected_state, best_score

    def _execute_insertion(self, step, leader_id, selected_av, selected_state,
                           dic_platoon_members, score):
        """
        Execute the lane change for the selected AV and update tracking structures.
        """
        try:
            self.traci.vehicle.changeLane(selected_av, self.target_lane, duration=100)
            print(f"[SplitInsert] {selected_av} selected with score {score:.3f}")
            platoon_snapshot = dic_platoon_members[leader_id]
            self.ls_splited_platoon.append(leader_id)

            self.insert_buffer.append({
                "leader_id": leader_id,
                "platoon_snapshot": platoon_snapshot,
                "av_id": selected_av,
                "state": selected_state,
                "step": step
            }) # this data is for training

            self.dic_insertedAV[selected_av] = 'split_insert'
            print(f'dic_insertedAVcands: {self.dic_insertedAV}')
        except self.traci.TraCIException:
            print(f"[SplitInsert] {selected_av} failed to insert due to TraCI exception")
