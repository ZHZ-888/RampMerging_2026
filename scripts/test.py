
def run_agent_decision_old(self, step, dic_platoon_members, dic_oversized_platoon_states,
                       dic_leader_candidates, ls_upA, gating_value=None):
    '''
    split_insert (side AV insert into oversized platoon)

    :param dic_platoon_members: {leader_id: [leader_id, veh2, ...]}
            dic_oversized_platoon_states:
            dic_leader_candidates: candidate AVs for oversized_platoon (leader_AV),
                e.g. {leader_AV:[candidateAV1, candidateAV2]}
            gating_value: gating value, active action while top-score above it

    :return: dic_insertedAVcands = {AV_id: type, ...} record candidate promotedAV and its type;
            here type = 'split'
            Leader change triggered by oversized platoon splitting (split)
    '''
    if not dic_oversized_platoon_states:
        return {}  # {AV_id: type, ...}, type = 'split' or 'free', candidates
    inserted_results = [] # list of inserted AVs with (leader_id, av_id, state, action)
    for leader_id, platoon_states in dic_oversized_platoon_states.items():
        if leader_id in self.ls_splited_platoon or leader_id not in dic_leader_candidates:
            continue
        # === Add scoring interval control ===
        last_step = self.last_score_step.get(leader_id, -999)
        if step - last_step < self.scoring_interval:
            continue
        self.last_score_step[leader_id] = step
        ls_candidateAV = dic_leader_candidates[leader_id] # list of candidate AV
        if not ls_candidateAV:
            continue
        # Use low threshold during warm-up phase to encourage early exploration
        low_threshold = 0.0001
        high_threshold = 0.3
        threshold = low_threshold if step < self.training_warmup_steps else high_threshold
        pMember = dic_platoon_members[leader_id]  # platoon member list

        # Evaluate all candidate AVs for this platoon, select one with highest predicted score
        best_score = -float('inf')
        selected_av = None
        selected_state = None
        dic_lcAV_score = {}
        for av_id in ls_candidateAV:
            # state = self.agent.state_builder.build_state(av_id, pMember, platoon_states)
            state = self.agent.state_builder.build_state2(av_id, pMember, platoon_states, ls_upA)
            score = self.agent.predict_score(state)
            dic_lcAV_score[av_id] = score
            # self.dic_score_reward[av_id] = [score]
            if score > best_score:
                best_score = score
                selected_av = av_id
                selected_state = state

        if gating_value is not None:
            if best_score >= gating_value:
                # Only proceed if score exceeds gating_value
                state = selected_state
                score = best_score
            else:
                # Below gating_value: do not insert
                selected_av, state, score = None, None, best_score
        else:
            # No threshold gating: always insert top-scoring candidate
            state = selected_state
            score = best_score

        self.dic_score_reward[selected_av] = [score]

        # print(f"[Score] Leader {leader_id} → best AV score: {score:.3f}")
        self.ls_score.append(score)
        # === Execute AV lane change if selected ===
        if selected_av:
            try:
                self.traci.vehicle.changeLane(selected_av, self.target_lane, duration=100)

                print(f"[Agent] Insert decision: {selected_av} with score {score:.3f}")
                platoon_snapshot = dic_platoon_members[leader_id] # platoon snapshot at the moment of insertion decision
                self.ls_splited_platoon.append(leader_id)
                self.insert_buffer.append({
                    "leader_id": leader_id,
                    "platoon_snapshot": platoon_snapshot,
                    "av_id": selected_av,
                    "state": state,
                    "step": step
                })

                inserted_results.append((leader_id, selected_av, state))
                self.dic_insertedAV[selected_av] = 'split_insert'
                print(f'dic_insertedAVcands: {self.dic_insertedAV}')
            except self.traci.TraCIException:
                print(f"[Agent] Failed to insert {selected_av} due to TraCI exception")
    return self.dic_insertedAV


#%%
def main(args=None, root=None):
    """
    Unified entry point for CLI (command line interface) / HPC.
    This function is called by run.py.
    It parses command-line arguments and calls mpgc_main().
    """
    prc.PRINT_ENABLED = False

    # 1. Parse Args (HPC/CLI Mode)
    parser = hpc_utils.standard_arg_parser()
    parsed_args = parser.parse_args(args=args)

    # 2. Run simulation
    # Call the original algorithm
    dic_targets, ls_features = mpgc_main(
        av_p=parsed_args.av_p,
        r_fr=parsed_args.r_fr,
        m_fr=parsed_args.m_fr,
        seed=parsed_args.seed,
        gui=parsed_args.gui
    )


