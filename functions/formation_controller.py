from functions import platoon_basic as pbasic
from functions import platoon_oversized_handler as poversized
from functions import platoon_sparse_handler as psparse
from functions import platoon_lane_manager as plane
from functions import merging_control_regular as mcr

from platoon_split_rl_model import split_insert_agent_handler as agent
from platoon_split_rl_model import free_insert_agent_handler as free_agent

class FormationController:
    def __init__(self, data_recorder, traci):
        self.data_recorder = data_recorder
        # self.merge_regular = mcr.MergingControlRegular(traci, self.data_recorder)

        # Initialize platoon modules
        self.p_basic = pbasic.PlatoonBasic(traci, data_recorder)
        self.p_oversized = poversized.PlatoonOversizedHandler(traci, data_recorder, self.p_basic)
        self.p_sparse = psparse.PlatoonSparseHandler(traci, data_recorder, self.p_basic)
        self.p_lane = plane.PlatoonLaneManager(traci, data_recorder)

        # RL Agents
        self.split_agent = agent.AgentHandler(traci, data_recorder,
                                              mode='predict')  # predict or train
        self.free_insert_agent = free_agent.FreeInsertAgentHandler(traci, data_recorder,
                                                                   self.p_basic,
                                                                   mode='predict')

    def step(self, st, step, lc):
        '''

        :param st:
        :param step:
        :param lc:
        :return:
            dic_socre_reward: {'mbav1533': [0.6405384540557861]}
            dic_follower_state: {'mhv48': ['following_mode', 'mav38'], 'mhv65': ['following_mode', 'mav38']}
            his_dic_platoon_size: {'mav38': 11, 'mav278': 11, 'mav754': 11}; history of dic_platoon_size
            dic_id_features: {'mhv48': ['mhv48', 81.32679695334296, 16.66558558968912, 79.32679695334296, 5.1, 2, 'mav38'],
                                'mhv65': ['mhv65', 50.64862995222547, 24.270287929523167, 110.99295710039699, 5.1, 3, 'mav38']}
            dic_final_platoon_info: {66: 'AHHHHHHHHHH', 90: 'AHHHHH', 138: 'AHHHHHHHHH', 174: 'AH', 177: 'AHH'}
        '''
        # === Unpack veh info ===
        # print(f"*******step: {step}*********")
        dic_vid_groups = self.data_recorder.record_multi_lane_info()

        ls_vehid = dic_vid_groups['ls_vehid']  # tuple, all vehicle in this step

        ls_ihA = dic_vid_groups['ls_ihA']  # all veh on inflow_highway, big => small
        ls_ihAB_av = dic_vid_groups['ls_ihAB_av']
        ls_ihB = dic_vid_groups['ls_ihB']
        ls_ihB_av = dic_vid_groups['ls_ihB_av']

        ls_ihAB_hv = dic_vid_groups['ls_ihAB_hv']
        ls_wsBC_hv = dic_vid_groups['ls_wsBC_hv']
        ls_wsB_av = dic_vid_groups['ls_wsB_av']
        ls_centerA_av = dic_vid_groups['ls_centerA_av']
        ls_m_leader_up_asc = dic_vid_groups['ls_m_leader_up_asc']
        length_ih = self.data_recorder.length_ih  # obtain the length of inflow_highway

        # ******** HANDLE OVERSIZED PLATOONS ********
        dic_tags, ls_leader_AV, ls_follower_AV, dic_AVroleChange \
            = self.p_basic.tag_vehicles13(ls_ihA, max_team_size=11)  # SPLIT_PROMOTE
        his_dic_platoon_size, dic_platoon_size, dic_platoon_members \
            = self.p_basic.get_platoon_size3(ls_ihA, ls_leader_AV)
        # SPLIT_INSERT agent
        dic_oversized_platoon_states, dic_leader_candidates \
            = self.p_oversized.find_oversizedP_nearbyAV(ls_ihB_av, dic_platoon_size)
        # dic_insertedAV = self.split_agent.run_agent_decision(step, dic_platoon_members,
        #                                                      dic_oversized_platoon_states,
        #                                                      dic_leader_candidates,
        #                                                      ls_ihA, gating_value=0.5)
        # self.merge_regular.flashing_lane_changing(step, dic_insertedAV, ls_ihB)
        dic_score_reward = self.split_agent.update_reward(step, st, dic_platoon_members)
        # ahah.plot_scores(step, st)
        # ahah.plot_loss(step, st)

        # ******** HANDLE SPARSE PLATOONS ********
        dic_nonOversizedP = self.p_oversized.non_oversized_platoon(dic_platoon_members, dic_oversized_platoon_states)
        dic_id_preState, dic_id_features = self.p_sparse.predict_flw_state(dic_tags, ls_vehid, model=True)
        dic_sparseP, dic_standard_platoon = self.p_sparse.find_sparse_platoon(dic_nonOversizedP, dic_id_preState)
        # sparseP => sparse_platoon = {av_leader: first_free_hv}
        # self.p_sparse.free_promote(dic_sparseP, dic_platoon_members)  # FREE_PROMOTE
        # filter out AV followers from sparse platoons
        dic_sparseP_filered = self.p_sparse.filter_out_AV_followers(dic_sparseP, dic_platoon_members)
        # Collect free followers - find nearby AVs and execute collection (NEW)
        dic_sparse_candidates = self.p_sparse.find_sparseP_nearbyAV(ls_ihB_av, dic_sparseP_filered)
        # Use free-insert RL agent
        # dic_free_insertedAV = self.free_insert_agent.run_free_insert_decision(
        #     step, dic_platoon_members, dic_sparseP_filered, dic_sparse_candidates, ls_ihA,
        #     gating_value=0.4)
        dic_free_score_reward = self.free_insert_agent.update_reward(step, st, dic_platoon_members)
        # self.free_insert_agent.plot_scores(step, st)
        # self.free_insert_agent.plot_loss(step, st)

        # Lane change and speed control
        # self.p_lane.manage_lc_behavior_near_ws(lc, ls_ihAB_hv, ls_wsBC_hv, length_ih, p_to_inner=0.8)
        self.p_lane.manage_lc_behaviour(lc, dic_tags) # 260226, new design
        self.p_lane.restrict_strategic_lc(ls_ihAB_av)
        # encourage AV without follower move to inner lane (from lane0 to lane1)
        self.p_lane.move_av_no_followers(ls_leader_AV, dic_platoon_members)
        # encourage AV followers move to inner lane (from lane0 to lane1)
        self.p_lane.encourage_av_fol_to_out_lane(ls_leader_AV, dic_standard_platoon)

        # control gaps between platoons
        self.p_basic.form_platoon3(ls_vehid, ls_leader_AV, ls_follower_AV)
        self.p_basic.restore_speed_limit3(step, ls_leader_AV, ls_m_leader_up_asc)  # improve to 25 m/s if platoon formed
        self.p_basic.restore_speed_limit2(ls_wsB_av)  # restore to 27.78 m/s on wsB
        # set follower color as light green
        self.p_basic.set_follower_color()

        # record dic_platoon_members
        dic_member_to_leader = self.p_basic.update_member_to_leader(dic_platoon_members)
        self.data_recorder.dic_member_to_leader = dic_member_to_leader
        # record target value
        dic_follower_state, dic_final_platoon_info = (
            self.p_basic.record_follower_state2(step))

        return dic_score_reward, dic_follower_state, his_dic_platoon_size, dic_id_features, dic_final_platoon_info

