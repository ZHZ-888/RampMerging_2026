from functions import platoon_basic as pbasic
from functions import platoon_oversized_handler as poversized
from functions import platoon_sparse_handler as psparse
from functions import platoon_lane_manager as plane
from functions import v2x_disturbance as v2x

from rl_model import split_insert_agent_handler as agent
from rl_model import free_insert_agent_handler as free_agent

class FormationController:
    def __init__(self, data_recorder, traci, splitting_agent='predict', collecting_agent='predict',
                 exp_name='default_run', loss_rate=0, learning_rate=5e-4, train_interval=32, model_name=None):
        '''
        Train split_agent, "splitting_agent='train', collecting_agent=None"
        Train free_insert_agent, "splitting_agent=None, collecting_agent='train'

        Evaluate split_agent, "splitting_agent='predict', collecting_agent=None"
        Evaluate free_insert_agent, "splitting_agent=None, collecting_agent='predict'
        '''

        self.data_recorder = data_recorder

        # Initialise platoon modules
        self.p_basic = pbasic.PlatoonBasic(traci, data_recorder)
        self.p_oversized = poversized.PlatoonOversizedHandler(traci, data_recorder, self.p_basic)
        self.p_sparse = psparse.PlatoonSparseHandler(traci, data_recorder, self.p_basic)
        self.p_lane = plane.PlatoonLaneManager(traci, data_recorder)

        self.loss_rate = loss_rate
        if self.loss_rate == 0:
            self.sa_buffer = None
            self.ca_buffer = None
        else:
            self.sa_buffer = v2x.UpdateDelayBuffer(loss_rate=self.loss_rate)  # buffer for splitting agent
            self.ca_buffer = v2x.UpdateDelayBuffer(loss_rate=self.loss_rate)  # buffer for collecting agent

        self.train_interval = train_interval # rl training update interval
        self.update_interval = 10 # test

        # RL Agents
        self.sa_gating = 0.5 if splitting_agent == 'predict' else 0
        self.ca_gating = 0.4 if collecting_agent == 'predict' else 0
        self.split_agent = agent.AgentHandler(
            traci, data_recorder, mode=splitting_agent, exp_name=exp_name,
            lr=learning_rate, model_name=model_name) if splitting_agent else None
        self.free_insert_agent = free_agent.FreeInsertAgentHandler(
            traci, data_recorder, self.p_basic,
            mode=collecting_agent, exp_name=exp_name,
            lr=learning_rate) if collecting_agent else None

    def platoon_initialise(self, ls_ihA):
        # ******** PLATOON INITIALISATION ********
        dic_tags, ls_leader_AV, ls_follower_AV, dic_AVroleChange \
            = self.p_basic.tag_vehicles13(ls_ihA, max_team_size=11)  # ** SPLIT_PROMOTE **
        his_dic_platoon_size, dic_platoon_size, dic_platoon_members \
            = self.p_basic.get_platoon_size3(ls_ihA, ls_leader_AV)
        return dic_tags, ls_leader_AV, ls_follower_AV, dic_platoon_size, dic_platoon_members, his_dic_platoon_size

    def splitting(self, st, step, ls_ihA, ls_ihB_av, dic_platoon_size, dic_platoon_members):
        # ******** HANDLE OVERSIZED PLATOONS ********
        self.split_agent.release_insertion(step, self.sa_buffer)
        if step % self.update_interval != 0:
            return {}
        dic_oversized_platoon_states, dic_split_candidates, dic_nonOversizedP \
            = self.p_oversized.find_oversizedP_nearbyAV(ls_ihB_av, dic_platoon_size, dic_platoon_members)
        # ** SPLIT_INSERT ** agent
        if self.split_agent:
            dic_split_insertedAV = self.split_agent.run_agent_decision(step, dic_platoon_members,
                                                                       dic_oversized_platoon_states,
                                                                       dic_split_candidates,
                                                                       ls_ihA, gating_value=self.sa_gating)
            dic_score_reward = self.split_agent.update_reward(step, st, dic_platoon_members,
                                                              train_interval=self.train_interval)
            self.split_agent.record_scores(step, st)
            self.split_agent.record_loss(step, st)
        return dic_nonOversizedP

    def collecting(self, st, step, ls_ihA, ls_ihB_av, dic_nonOversizedP, dic_platoon_members, dic_tags, ls_vehid):
        # ******** HANDLE SPARSE PLATOONS ********
        self.free_insert_agent.release_insertion(step, self.ca_buffer)
        if step % self.update_interval != 0:
            return {}, {}
        dic_id_preState, dic_id_features = self.p_sparse.predict_flw_state(dic_tags, ls_vehid, model=True)
        dic_sparseP, dic_standard_platoon = self.p_sparse.find_sparse_platoon(dic_nonOversizedP, dic_id_preState)
        self.p_sparse.free_promote(dic_sparseP, dic_platoon_members)  # ** FREE_PROMOTE **
        # filter out AV followers from sparse platoons
        dic_sparseP_filered = self.p_sparse.filter_out_AV_followers(dic_sparseP, dic_platoon_members)
        # Find nearby side-lane AVs
        dic_collect_candidates = self.p_sparse.find_sparseP_nearbyAV(ls_ihB_av, dic_sparseP_filered)
        # ** FREE_INSERT ** agent
        if self.free_insert_agent:
            dic_free_insertedAV = self.free_insert_agent.run_free_insert_decision(step, dic_platoon_members,
                                                                                  dic_sparseP_filered,
                                                                                  dic_collect_candidates,
                                                                                  ls_ihA,
                                                                                  gating_value=self.ca_gating)  # 0.4 for predict
            dic_free_score_reward = self.free_insert_agent.update_reward(step, st, dic_platoon_members,
                                                                         train_interval=self.train_interval)
            self.free_insert_agent.record_scores(step, st)
            self.free_insert_agent.record_loss(step, st)
        return dic_standard_platoon, dic_id_features

    def control_platoon_gap(self, step, ls_vehid, ls_leader_AV, ls_follower_AV, ls_m_leader_up_asc, ls_wsB_av):
        '''
        control gaps between platoons; do not rely on V2X, it is onboard decision
        '''
        if step % self.update_interval != 0:
            return
        self.p_basic.form_platoon3(ls_vehid, ls_leader_AV, ls_follower_AV)
        self.p_basic.restore_speed_limit3(step, ls_leader_AV, ls_m_leader_up_asc)  # improve to 25 m/s if platoon formed
        self.p_basic.restore_speed_limit2(ls_wsB_av)  # restore to 27.78 m/s on wsB
        # set follower color as light green
        self.p_basic.set_follower_color()

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

        dic_score_reward = {}  # Initialise dic_score_reward to avoid uninitialized variable issues
        # === Unpack veh info ===
        # print(f"*******step: {step}*********")
        dic_vid_groups = self.data_recorder.record_multi_lane_info()
        ls_vehid = dic_vid_groups['ls_vehid']  # tuple, all vehicle in this step
        ls_ihA = dic_vid_groups['ls_ihA']  # all veh on inflow_highway, big => small
        ls_ihAB_av = dic_vid_groups['ls_ihAB_av']
        ls_ihB_av = dic_vid_groups['ls_ihB_av']
        ls_wsB_av = dic_vid_groups['ls_wsB_av']
        ls_m_leader_up_asc = dic_vid_groups['ls_m_leader_up_asc']

        # ******** PLATOON INITIALISATION ********
        (dic_tags, ls_leader_AV, ls_follower_AV, dic_platoon_size,
         dic_platoon_members, his_dic_platoon_size) = self.platoon_initialise(ls_ihA)

        # ******** HANDLE OVERSIZED PLATOONS ********
        dic_nonOversizedP = self.splitting(st, step, ls_ihA, ls_ihB_av, dic_platoon_size, dic_platoon_members)

        # ******** HANDLE SPARSE PLATOONS ********
        dic_standard_platoon, dic_id_features = self.collecting(st, step, ls_ihA, ls_ihB_av, dic_nonOversizedP,
                                                                dic_platoon_members, dic_tags, ls_vehid)

        # Flashing lane change side AVs
        # self.merge_regular.flashing_lane_changing(step, dic_insertedAV, ls_ihB)

        # ******** Lane change and speed control ********
        self.p_lane.manage_hv_lc_behaviour(lc, dic_tags) # mange hv_followers lane change behavior
        self.p_lane.restrict_strategic_lc(ls_ihAB_av) # only restrict AV strategic lane change
        # encourage AV_leader without follower move to inner lane (from lane0 to lane1)
        self.p_lane.move_leader_no_fol_to_inner(ls_leader_AV, dic_platoon_members)
        # encourage AV followers move to inner lane (from lane0 to lane1)
        self.p_lane.move_av_fol_to_inner(ls_leader_AV, dic_standard_platoon)

        # control gaps between platoons
        self.control_platoon_gap(step, ls_vehid, ls_leader_AV, ls_follower_AV, ls_m_leader_up_asc, ls_wsB_av)

        # ******** RECORD INFO ********
        dic_member_to_leader = self.p_basic.update_member_to_leader(dic_platoon_members)
        self.data_recorder.dic_member_to_leader = dic_member_to_leader
        # record target value
        dic_follower_state, dic_final_platoon_info = (
            self.p_basic.record_follower_state2(step))

        return (dic_score_reward, dic_follower_state, his_dic_platoon_size,
                dic_id_features, dic_final_platoon_info)
