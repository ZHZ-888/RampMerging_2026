from functions import platoon_basic as pbasic
from functions import platoon_oversized_handler as poversized
from functions import platoon_sparse_handler as psparse
from functions import platoon_lane_manager as plane
from functions import merging_control_regular as mcr

from platoon_split_rl_model import main_agent_handler as agent


class FormationController:
    def __init__(self, data_recorder, traci):
        self.data_recorder = data_recorder
        self.merge_regular = mcr.MergingControlRegular(traci, self.data_recorder)

        # Initialize platoon modules
        self.p_basic = pbasic.PlatoonBasic(traci, data_recorder)
        self.p_oversized = poversized.PlatoonOversizedHandler(traci, data_recorder, self.p_basic)
        self.p_sparse = psparse.PlatoonSparseHandler(traci, data_recorder, self.p_basic)
        self.p_lane = plane.PlatoonLaneManager(traci, data_recorder)

        # RL module
        self.rl_agent = agent.AgentHandler(traci, self.merge_regular, mode='predict')  # predict or train

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
        dic_vid_groups = (
            self.data_recorder.dic_vid_groups
            if self.data_recorder.dic_vid_groups
            else self.data_recorder.record_multi_lane_info()
        )

        ls_vehid = dic_vid_groups['ls_vehid']  # tuple, all vehicle in this step

        ls_ihA = dic_vid_groups['ls_ihA']  # all veh on inflow_highway, big => small
        ls_ihAB_av = dic_vid_groups['ls_ihAB_av']
        ls_ihB = dic_vid_groups['ls_ihB']
        ls_ihB_av = dic_vid_groups['ls_ihB_av']

        ls_ihAB_hv = dic_vid_groups['ls_ihAB_hv']
        ls_wsBC_hv = dic_vid_groups['ls_wsBC_hv']
        ls_centerA_av = dic_vid_groups['ls_centerA_av']
        length_ih = self.data_recorder.length_ih  # obtain the length of inflow_highway

        # SC1: Handle oversized platoons
        dic_tags, ls_leader_AV, ls_follower_AV, dic_AVroleChange \
            = self.p_basic.tag_vehicles13(ls_ihA, max_team_size=11)  # SPLIT_PROMOTE
        his_dic_platoon_size, dic_platoon_size, dic_platoon_members \
            = self.p_basic.get_platoon_size3(ls_ihA, ls_leader_AV)
        dic_oversized_platoon_states, dic_leader_candidates \
            = self.p_oversized.find_oversizedP_nearbyAV(ls_ihB_av, dic_platoon_size)

        # record dic_platoon_members
        dic_member_to_leader = self.p_basic.update_member_to_leader(dic_platoon_members)
        self.data_recorder.dic_member_to_leader = dic_member_to_leader

        # RL modules - split oversized platoons
        dic_insertedAV = self.rl_agent.run_agent_decision(step, dic_platoon_members,
                                                 dic_oversized_platoon_states, dic_leader_candidates,
                                                 ls_ihA, gating_value=0.5)  # SPLIT_INSERT

        self.merge_regular.flashing_lane_changing(step, dic_insertedAV, ls_ihB)
        dic_score_reward = self.rl_agent.update_reward(step, st, dic_platoon_members)
        # ahah.plot_scores(step, st)
        # ahah.plot_loss(step, st)

        # SC2: Handle sparse platoons - predict and promote
        dic_nonOversizedP = self.p_oversized.non_oversized_platoon(dic_platoon_members, dic_oversized_platoon_states)
        dic_id_preState, dic_id_features = self.p_sparse.predict_flw_state(dic_tags, ls_vehid, model=True)
        dic_sparseP = self.p_sparse.find_sparse_platoon(dic_nonOversizedP, dic_id_preState)
        # sparseP => sparse_platoon = {av_leader: first_free_hv}
        self.p_sparse.free_promote(dic_sparseP, dic_platoon_members)  # FREE_PROMOTE

        # SC3: Collect free followers - find nearby AVs and execute collection (NEW)
        dic_sparse_candidates = self.p_sparse.find_sparseP_nearbyAV(ls_ihB_av, dic_sparseP)
        # self.p_sparse.collect_free_followers(dic_sparse_candidates, dic_sparseP)

        # Lane change and speed control
        self.p_lane.manage_lc_behavior_near_ws(lc, ls_ihAB_hv, ls_wsBC_hv, length_ih, p_to_inner=0.8)
        # encourage AV without follower move to outer lane (from lane0 to lane1)
        self.p_lane.move_av_no_followers(ls_leader_AV, dic_platoon_members)

        # record target value
        dic_follower_state, dic_final_platoon_info = (
            self.p_basic.record_follower_state2(step))
        # control gaps between platoons
        self.p_basic.form_platoon3(ls_leader_AV, ls_follower_AV)
        self.p_basic.restrict_strategic_lc(ls_ihAB_av)
        self.p_basic.restore_speed_limit2(ls_centerA_av)

        return dic_score_reward, dic_follower_state, his_dic_platoon_size, dic_id_features, dic_final_platoon_info

