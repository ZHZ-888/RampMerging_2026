# free_insert_agent_handler.py (also collect agent)

import os
import numpy as np
from datetime import datetime

from rl_model.rl_module import RLScoringAgent

# Model path for free-insert agent
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

class FreeInsertAgentHandler:
    """RL agent for free-insert scenario: Score side-lane AVs for inserting
    ahead of free followers in sparse platoons.

    Goal: Maximize collection of free followers into new dense platoons.
    """

    def __init__(self, traci, data_recorder, p_basic, scoring_interval=10,
                 mode='train', tsg_mode='off', exp_name='default_run',
                 lr=5e-4, gate_agent=None):
        """
        Initialise the free-insert agent.

        Args:
            traci: SUMO traci connection
            data_recorder: DataRecording instance
            merge_regular: MergeRegular instance (for lane change execution)
            scoring_interval: Minimum steps between scoring decisions
            mode: 'train' or 'predict'
        """
        self.traci = traci
        self.data_recorder = data_recorder
        self.p_basic = p_basic
        self.mode = mode
        self.tsg_mode = tsg_mode

        active_exp_name = exp_name if mode == 'train' else f"EVAL_{exp_name}"

        default_model = 'free_insert_score_model_260303_2333_second_version.pt'
        path_pt = os.path.join(project_root, 'rl_model', 'saved_models', default_model)
        # Initialize RL scoring agent
        self.agent = RLScoringAgent(traci, data_recorder,
                                    exp_name=active_exp_name,
                                    model_path=path_pt if mode == "predict" else None,
                                    lr=lr) # 0.0005

        self.gate_agent = gate_agent

        if self.tsg_mode in ("train", "predict") and self.gate_agent is None:
            raise ValueError("[CA-Gate] tsg_mode requires a shared gate_agent")

        self.task_name = 'collecting'
        self.gate_collected = 0

        # Configuration
        self.scoring_interval = scoring_interval
        self.training_warmup_steps = 20000
        self.next_save_step = 10000
        self.collected = 0  # Transition count
        self.target_lane = 0  # Inner lane

        # Tracking structures
        self.ls_free_inserted = []  # Successfully inserted AVs
        self.insert_buffer = []  # Pending reward evaluation
        self.last_score_step = {}  # Cooldown control per sparse platoon
        self.ls_score = []  # Score history for plotting
        self.dic_insertedAV = {}  # {av_id: 'free_insert'}
        self.dic_score_reward = {}  # {av_id: [score, reward]}

        self.last_update_payload_pair = None  # {target leader: candidate leader}
        self.payload = None

    def run_free_insert_decision(self, step, dic_platoon_members, dic_sparse_platoons,
                                 dic_sparse_candidates, gating_value=0):
        """
        Main decision loop: Score candidate AVs and select best one for each sparse platoon.

        Args:
            step: Current simulation step
            dic_platoon_members: {leader_id: [members]}
            dic_sparse_platoons: {sparse_leader: first_free_follower_id}
            dic_sparse_candidates: {sparse_leader: [candidate_av_ids]}
            gating_value: Minimum score threshold for insertion

        Returns:
            dic_insertedAV: {av_id: 'free_insert'}
        """
        if not dic_sparse_candidates:
            return {}
        for sparse_leader, ls_candidates in dic_sparse_candidates.items():
            if sparse_leader not in dic_sparse_platoons:
                continue
            # Evaluate candidates and select best
            selected_av, selected_state, best_score, gate_input = self._evaluate_candidates(
                step, sparse_leader, dic_platoon_members, dic_sparse_platoons,
                ls_candidates, gating_value)

            if selected_av and (self.last_update_payload_pair != {sparse_leader:selected_av}):
                self.payload = (sparse_leader, selected_av, selected_state,
                                dic_platoon_members, dic_sparse_platoons, best_score, gate_input)
                self.last_update_payload_pair = {sparse_leader: selected_av}

        return self.dic_insertedAV

    def release_insertion(self, step, laneChange_buffer):
        payload = self.payload
        if laneChange_buffer:
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

    def update_reward_ori(self, current_step, st, dic_platoon_members, train_interval):
        """
        Check insert_buffer for completed insertions and calculate rewards.
        Safely iterates using a shallow copy to remove vehicles.
        """
        for entry in self.insert_buffer:
            sparse_leader = entry['sparse_leader']
            # Check if original leader has exited control zone or disappeared
            try:
                lane_id = self.traci.vehicle.getLaneID(sparse_leader)
            except self.traci.TraCIException:
                # Leader already exited or removed
                self.insert_buffer.remove(entry)
                continue

            if lane_id != 'ws_1':  # Exit lane
                continue

            # --- Leader exited → evaluate reward ---
            lc_av = entry['lc_av']
            reward = self.evaluate_free_insert_reward(lc_av, entry['first_free_follower'],
                                                      entry['sparse_snapshot'], dic_platoon_members)

            # Update reward tracking & print
            if lc_av in self.dic_score_reward:
                self.dic_score_reward[lc_av].append(reward)
            print(f"[FreeInsert] {lc_av} reward: {reward:+.3f}")

            # Record transition for learning
            if self.mode == 'train':
                self.agent.record_transition(entry['state'], reward)
                self.collected += 1

                # Trigger training after warmup
                if self.collected >= train_interval and current_step > self.training_warmup_steps:
                    self.agent.log_training_metrics(current_step) # log performance metrics BEFORE updating model
                    self.agent.train_on_recorded(current_step, epochs=5, batch_size=int(train_interval/2))
                    self.collected = 0
            # Clean up processed entry
            self.insert_buffer.remove(entry)

        if self.mode == 'train': # Periodic model saving
            self._save_model_if_needed(current_step, st)

        return self.dic_score_reward

    def update_reward(self, current_step, st, dic_platoon_members, train_interval):
        """
        Check insert_buffer for completed insertions and calculate rewards.
        """
        updated = False

        # use shallow copy because entries may be removed
        for entry in self.insert_buffer[:]:
            sparse_leader = entry['sparse_leader']

            try:
                lane_id = self.traci.vehicle.getLaneID(sparse_leader)
            except self.traci.TraCIException:
                self.insert_buffer.remove(entry)
                continue

            if lane_id != 'ws_1':
                continue

            # --- Leader exited → evaluate reward ---
            lc_av = entry['lc_av']

            reward = self.evaluate_free_insert_reward(
                lc_av,
                entry['first_free_follower'],
                entry['sparse_snapshot'],
                dic_platoon_members
            )

            if lc_av in self.dic_score_reward:
                self.dic_score_reward[lc_av].append(reward)

            print(f"[FreeInsert] {lc_av} reward: {reward:+.3f}")

            # === Train CA scorer ===
            if self.mode == 'train':
                self.agent.record_transition(
                    entry['state'],
                    reward
                )
                self.collected += 1

            # === Train TSG gate ===
            elif self.tsg_mode == 'train':
                self.gate_agent.record_transition(
                    gate_input=entry['gate_input'],
                    reward=reward,
                    task_name=self.task_name
                )
                self.gate_collected += 1

            self.insert_buffer.remove(entry)
            updated = True

        # === Update CA scorer ===
        if updated and self.mode == 'train':
            if self.collected >= train_interval and current_step > self.training_warmup_steps:
                self.agent.log_training_metrics(current_step)
                self.agent.train_on_recorded(
                    current_step,
                    epochs=5,
                    batch_size=int(train_interval / 2)
                )
                self.collected = 0

        if self.mode == 'train':
            self._save_model_if_needed(current_step, st)

        return self.dic_score_reward

    def evaluate_free_insert_reward(self, lc_av, first_free_follower,
                                    sparse_snapshot, dic_platoon_members):
        """
        Calculate reward for free-insert action.

        REWARD LOGIC:
        - Goal: Maximize number of free followers converted to following mode
        - Success: captured_free_fol / total_free - captured_norm_fol * 0.01 (penalty for capturing non-free followers)
        - Failure: -0.1 (wrong lane, no followers, exception)

        Args:
            lc_av: Inserted AV ID
            sparse_snapshot: {sparse_leader: orignal followers} at decision time
            first_free_follower
            dic_platoon_members: Current platoon structure, only record platoon leaders on inflow_highway

        Returns:
            float: Reward value in [-0.1, 1.0]
        """
        penalty = -0.1

        try:
            # Check if AV is on correct lane
            lane_id = self.traci.vehicle.getLaneID(lc_av)
            if 'inflow_highway_0' not in lane_id:
                print(f"[FreeInsert] {lc_av} not on inner lane: {lane_id}")
                return penalty
            # Check if AV became a leader with followers
                # no follower => penalty
            if lc_av not in dic_platoon_members:
                print(f"[FreeInsert] {lc_av} not a platoon leader")
                return penalty
                # followers are not original free followers => penalty
            current_members = dic_platoon_members[lc_av]
            if len(current_members) <= 1:  # Only leader, no followers
                print(f"[FreeInsert] {lc_av} has no followers")
                return penalty

            # Count captured free followers
            original_leader = next(iter(sparse_snapshot.keys()))
            original_members = sparse_snapshot[original_leader] # leader + followers
            first_free_idx = original_members.index(first_free_follower)
            original_free_list = original_members[first_free_idx:]
            captured_free = [fid for fid in current_members[1:] if fid in original_free_list]
            total_free = len(original_free_list)
            # Count captured norm followers
            original_norm_list = original_members[1:first_free_idx]
            captured_norm = [fid for fid in current_members[1:] if fid in original_norm_list]

            # capture 0 = penalty
            if len(captured_free) == 0:
                print(f"[FreeInsert] {lc_av} captured no free followers")
                return penalty
            # capture > 0 => reward proportional to capture rate
            captured_free_rate = len(captured_free) / max(total_free, 1)
            reward = captured_free_rate - len(captured_norm) * 0.01
            print(f"[FreeInsert] {lc_av} captured {len(captured_free)}/{total_free} free followers: {reward:.3f}"
                  f"; captured norm followers: {len(captured_norm)}")
            return reward

        except Exception as e:
            print(f"[FreeInsert] {lc_av} exception: {e}")
            return penalty

    def _get_pos(self, veh_id, fallback=0.0):
        try:
            return float(self.data_recorder.get_vid_states(veh_id)['pos'])
        except Exception:
            return float(fallback)

    def _build_tsg_timing_features_collecting(
            self,
            sparse_leader,
            selected_av,
            dic_platoon_members,
            dic_sparse_platoons,
            offset_scale=100.0
    ):
        """
        Build timing-related TSG features for collecting/free-insert.

        d_target_to_MCZ_norm:
            normalized remaining distance of target sparse leader.

        signed_insert_offset_norm:
            signed distance from top AV to ideal insertion position,
            normalized to [-1, 1].
        """
        d_target_raw = self._get_pos(sparse_leader)
        d_target_norm = np.clip(d_target_raw / self.agent.state_builder.max_lane_pos, 0.0, 1.0)

        try:
            first_free = dic_sparse_platoons[sparse_leader]
            members = dic_platoon_members.get(sparse_leader, [])

            d_first = self._get_pos(first_free, fallback=d_target_raw)

            if first_free in members:
                idx = members.index(first_free)
                if idx > 0:
                    preceding_vehicle = members[idx - 1]
                    d_prev = self._get_pos(preceding_vehicle, fallback=d_first)
                    d_ideal = 0.5 * (d_first + d_prev)
                else:
                    d_ideal = d_first
            else:
                d_ideal = d_first

            d_top = self._get_pos(selected_av, fallback=d_ideal)
            offset_raw = d_top - d_ideal
            offset_norm = np.clip(offset_raw / offset_scale, -1.0, 1.0)

        except Exception:
            offset_raw = 0.0
            offset_norm = 0.0

        return float(d_target_norm), float(offset_norm)


    def _evaluate_candidates(self, step, sparse_leader, dic_platoon_members, dic_sparse_platoons,
                             ls_candidates, gating_value):
        """
        Score all candidate AVs and select the best one.
        Also build gate_input for TSG.
        """
        # Cooldown check
        last_step = self.last_score_step.get(sparse_leader, -999)
        if step - last_step < self.scoring_interval:
            return None, None, None, None

        self.last_score_step[sparse_leader] = step

        if not ls_candidates:
            return None, None, None, None

        candidate_states = []
        scores = []
        valid_candidates = []

        # === Score all candidate AVs ===
        for av_id in ls_candidates:
            try:
                state = self.agent.state_builder.build_state_free(
                    cand_leader=av_id,
                    target_sparse_platoon={sparse_leader: dic_sparse_platoons[sparse_leader]},
                    dic_platoon_member=dic_platoon_members,
                )

                score = self.agent.predict_score(state)

                valid_candidates.append(av_id)
                candidate_states.append(state)
                scores.append(score)

            except Exception as e:
                print(f"[FreeInsert] {av_id} failed to score: {e}")
                continue

        if not scores:
            return None, None, None, None

        # === Select top-ranked AV ===
        top_idx = int(np.argmax(scores))
        selected_av = valid_candidates[top_idx]
        selected_state = candidate_states[top_idx]
        best_score = float(scores[top_idx])

        # === Build gate input ===
        gate_input = None

        if self.tsg_mode in ("train", "predict"):
            d_target_to_MCZ_norm, signed_insert_offset_norm = (
                self._build_tsg_timing_features_collecting(
                    sparse_leader=sparse_leader,
                    selected_av=selected_av,
                    dic_platoon_members=dic_platoon_members,
                    dic_sparse_platoons=dic_sparse_platoons
                )
            )

            gate_input = self.gate_agent.build_gate_input(
                x_top=selected_state,
                scores=scores,
                top_idx=top_idx,
                task_name=self.task_name,
                d_target_to_MCZ_norm=d_target_to_MCZ_norm,
                signed_insert_offset_norm=signed_insert_offset_norm
            )
        # === Decide whether to execute ===
        if self.tsg_mode == "train":
            # During TSG training, always execute top candidate to collect labels
            execute_decision = True

        elif self.tsg_mode == "predict":
            execute_decision, gate_logits, gate_probs = self.gate_agent.predict_execute(gate_input)

            print(
                f"[CA-Gate] sparse_leader={sparse_leader}, top_av={selected_av}, "
                f"score={best_score:.3f}, "
                f"reject_prob={gate_probs[0]:.3f}, "
                f"execute_prob={gate_probs[1]:.3f}, "
                f"execute={execute_decision}"
            )

        else: # self.tsg_mode = None; off
            # Original fixed-gating logic
            if gating_value is not None and best_score < gating_value:
                execute_decision = False
            else:
                execute_decision = True

        # === If rejected ===
        if not execute_decision:
            self.ls_score.append(best_score)
            print(
                f"[CA-Gate] Reject candidate: {selected_av}, "
                f"score={best_score:.3f}"
            )
            return None, None, best_score, gate_input

        # === If executed ===
        self.dic_score_reward[selected_av] = [best_score]
        self.ls_score.append(best_score)

        return selected_av, selected_state, best_score, gate_input

    def _execute_insertion(self, step, sparse_leader, selected_av, selected_state,
                           dic_platoon_members, dic_sparse_platoons, score, gate_input):
        """
        Execute lane change and update tracking structures.

        selected_state:
        sparse_snapshot: {sparse_leader: ori_followers} at decision time
        """
        try:
            if selected_av == 'mb_av1738':
                pass
            # Check if this sparse_leader is already being tracked
            if any(entry['sparse_leader'] == sparse_leader for entry in self.insert_buffer):
                # print(f"[FreeInsert] {sparse_leader} already being tracked, skipping insertion")
                return
            self.traci.vehicle.changeLane(selected_av, self.target_lane, duration=100)
            print(f"[FreeInsert] {selected_av} selected with score {score:.3f}")

            first_free_follower = dic_sparse_platoons[sparse_leader]
            sparse_snapshot = {sparse_leader: dic_platoon_members.get(sparse_leader, [])}
            # Record insertion for delayed reward
            self.insert_buffer.append({
                'sparse_leader': sparse_leader,
                'step': step,
                'first_free_follower': first_free_follower,
                'lc_av': selected_av,
                'state': selected_state,
                'sparse_snapshot': sparse_snapshot,
                'gate_input': gate_input
            })

            self.ls_free_inserted.append(sparse_leader)
            self.dic_insertedAV[selected_av] = 'free_insert'
            # record free insert AV and tag it
            self.p_basic.dic_tags[selected_av] = 1 # tag as leader
            self.p_basic.dic_AVroleChange[selected_av] = 'free_insert'
            # print(f'[FreeInsert] dic_insertedAV: {self.dic_insertedAV}')

        except self.traci.TraCIException:
            print(f"[FreeInsert] {selected_av} insert failed as TraCI exception")

    def _save_model_if_needed(self, current_step, st, model_type='sa'):
        """
        Periodically save the trained model.
        """
        save_interval = 30000
        if current_step > self.next_save_step or current_step == st * 10 - 1:
            timestamp = datetime.now().strftime("%y%m%d_%H%M")
            filename = f'free_insert_score_model_{timestamp}.pt'
            self.agent.save_model(filename)
            print(f"[Model] Auto-saved at step {current_step}")
            self.next_save_step += save_interval

    def record_loss(self, current_step, st):
        if current_step != st * 10 - 1 or self.mode != 'train':
            return
        self.agent.record_plot_loss()  # plot loss curve

    def record_scores(self, current_step, st):
        """Plot distribution of predicted scores."""
        if current_step != st * 10 - 1 or self.mode != 'train':
            return
        self.agent.record_plot_scores(self.ls_score)
