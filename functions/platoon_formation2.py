# platoon_formation2.py
import os
import numpy as np
import joblib  # model prediction
import random

current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the current script's directory
project_root = os.path.dirname(current_dir)  # Get the parent directory as the project root

class PlatoonForm:
    def __init__(self, traci, data_recorder):
        self.traci = traci
        self.data_recorder = data_recorder
        self.max_speed = self.data_recorder.max_speed
        self.dec_av = []
        self.dic_tags = {}
        self.max_team_size = 0
        self.dic_leaderAV_targetID = {}  # target_platoon and target_veh
        self.recover_time_map = {}  # record av speed setting recover time
        self.ls_speed_ok = []  # av_id that speed restore back to max(30m/s)
        self.leader_AV = set()  # all leader_AV
        self.dic_id_features = {}  # {id:[f1, f2,..., ], ...}
        self.dic_follower_state = {}  # state of each followers
        self.dic_id_preState = {}
        # follower_state_prediction prediction model
        # self.fs_model = joblib.load("/Models/follower_state_prediction_model_250501.pkl")
        # self.fs_model = joblib.load(
        #     os.path.join(project_root, 'rf_models', 'follower_state_prediction_model_250501.pkl'))
        self.fs_model = joblib.load(
            os.path.join(project_root, 'rf_models', 'follower_state_prediction_model_251121_ndarray.pkl'))
        self.ls_leader_AV = []
        self.ls_follower_AV = []
        self.ls_upA_lastStep = []  # ls_upA (upstream AV) last Step

        self.dic_oversizedPlatoon_info = {}  # {leader_AV: platoon_info}
        self.dic_platoon_size = {}  # all leaderAV and its platoon size
        self.dic_platoon_members = {}  # all leaderAV and its members
        self.free_triggered = False  # for record_predict3

        self.dic_AVroleChange = {}  # dic_AVroleChange = {AV_id: type, ...} record AV changed its role
        self.ls_follower_mc = []  # list of follower on merging control section of inflow_highway
        self.ls_leader_mc = []

        self.encourage_change_mark = set()  # record id that has been order to change to inner lane
        self.lcKeepRight_disabled = set()
        self.pending_changes = set()
        self.no_lc_av = set()
        self.no_lc_veh = set()
        self.dic_final_platoon_info = {} # m_dpt_type

    def get_platoon_size3(self, ls_upA, leader_AV):
        '''
        get the platoon size for each AV leader currently on the road
        record each platoon members
        :param ls_upA: ['mhv3305', 'mav3287', 'mhv3171', ...] decrease
        :param leader_AV: all leaderAV CURRENTLY on upstream_0, for example: ['mav2744', 'mav3038']
        :return: dic_id_size, current leader_AV and its corresponding size
                 dic_platoon_members = {
                                        'mav19': ['mav19', 'mhv46', 'mhv64', 'mhv74', 'mhv86'],
                                        "veh_2": ["veh_2", "veh_4"]
                                        }, current leader_AV and its corresponding members
                 self.dic_platoon_members (all history platoon and platoon members)
        '''
        dic_leaderAV_index = {}  # leader_id and its index in the traffic; {'mav2744': 1, 'mav3038': 8}
        ls_upA_re = ls_upA[::-1]

        for leader_id in leader_AV:
            idx_UPS = ls_upA_re.index(leader_id)  # idx on upstream_0
            dic_leaderAV_index[leader_id] = idx_UPS

        # get leader_id's corresponding size
        ids = list(dic_leaderAV_index.keys())
        indices = list(dic_leaderAV_index.values())
        dic_platoon_size = {}
        dic_platoon_members = {}
        for i in range(len(indices)):
            start = indices[i]
            end = indices[i + 1] if i < len(indices) - 1 else len(ls_upA_re)
            platoon = ls_upA_re[start:end]
            dic_platoon_size[ids[i]] = len(platoon)
            dic_platoon_members[ids[i]] = platoon
            self.dic_platoon_size[ids[i]] = len(platoon)
            self.dic_platoon_members[ids[i]] = platoon
        return self.dic_platoon_size, dic_platoon_size, dic_platoon_members

    def tag_vehicles13(self, ls_upA, max_team_size=11):
        '''
        label each vehicle: 0 => follower_HV, 1 => leader_AV, 2 => follower_AV;
        also for split_promote (promote a follower_AV to leader_AV to split oversized platoon)

        :param ls_upA: vehicle list ordered from newest to oldest
               ls_upA_re: oldest to newest
        :param max_team_size: maximum allowed platoon size
        :return: updated dic_tags, ls_leader_AV, ls_follower_AV
        '''
        ls_upA_re = ls_upA[::-1]  # Reverse to oldest → newest
        self.max_team_size = max_team_size

        if self.ls_upA_lastStep != ls_upA:
            # === get new_dic, make sure the order is correct ===
            # find the first id in ls_upA_re that already exists in dic_tags
            anchor_id = next((id for id in ls_upA_re if id in self.dic_tags), None)
            # Keep only the part of dic_tags before the anchor_id (maintain order)
            new_dic = {}
            for k in self.dic_tags:
                if k == anchor_id:
                    break
                new_dic[k] = self.dic_tags[k]
            # Extend new_dic with all IDs from ls_upA_re, assigning temporary value -1
            # This overwrites existing keys and inserts new ones in the given order
            for id in ls_upA_re:
                new_dic[id] = -1
            self.dic_tags = new_dic

            # === tag every id in ls_upA_re ===
            # current_leader= next((k for k in reversed(self.dic_tags) if self.dic_tags[k] == 1), None)
            current_leader = None
            current_team_size = 0

            for i, id in enumerate(ls_upA_re):
                if 'av' in id:
                    # Reconstruct leader and team size based on previous tagging
                    ls_keys = list(self.dic_tags.keys())
                    this_index = ls_keys.index(id)
                    n = 0
                    # range(start, stop, step); step = -1 (count backward)
                    for idx in range(this_index - 1, -1, -1):
                        n += 1
                        tagged_id = ls_keys[idx]
                        if self.dic_tags.get(tagged_id) == 1:
                            current_leader = tagged_id
                            current_team_size = n + 1
                            break

                    # === Proactive split: check if too many HVs follow this AV ===
                    too_many_hv_behind = False
                    if (current_leader is not None) and (current_team_size <= max_team_size):
                        hv_count = 0
                        remaining_slots = max_team_size - current_team_size
                        for j in range(i + 1, len(ls_upA_re)):
                            id_next = ls_upA_re[j]
                            if 'hv' in id_next:
                                hv_count += 1
                            else:
                                break
                        if hv_count > remaining_slots:
                            too_many_hv_behind = True

                    # === Assign AV as leader if needed ===
                    if ((current_leader is None) or (current_team_size > max_team_size)
                            or (id in self.dic_AVroleChange)):
                        self.dic_tags[id] = 1  # Mark as leader
                        current_leader = id
                        current_team_size = 1
                    elif too_many_hv_behind:
                        if 'b' in id:
                            self.dic_AVroleChange[id] = 'split_insert'
                        else:
                            self.dic_AVroleChange[id] = 'split_promote'
                        self.dic_tags[id] = 1  # Mark as leader
                        self.dic_id_preState.pop(id, None)  # if id becomes Leader, then no prediction for the state
                        current_leader = id
                        current_team_size = 1
                    else:
                        # Assign AV as follower
                        self.dic_tags[id] = 2
                        if id in self.dic_platoon_members:
                            del self.dic_platoon_members[id]
                        current_team_size += 1
                else:
                    # HV vehicle
                    self.dic_tags[id] = 0
                    if current_leader is not None:
                        current_team_size += 1

            self.ls_upA_lastStep = ls_upA

            # Update leader and follower lists for current control section
            dic_leader_AV = {k: v for k, v in self.dic_tags.items() if v == 1}
            dic_follower_AV = {k: v for k, v in self.dic_tags.items() if v == 2}

            dic_leader_AV_c = {k: v for k, v in dic_leader_AV.items() if k in ls_upA}
            dic_follower_AV_c = {k: v for k, v in dic_follower_AV.items() if k in ls_upA}

            self.ls_leader_AV = list(dic_leader_AV_c.keys())
            self.ls_follower_AV = list(dic_follower_AV_c.keys())

        return self.dic_tags, self.ls_leader_AV, self.ls_follower_AV, self.dic_AVroleChange

    def form_platoon3(self, ids_av, ids_f_av):
        '''
        based on desire_v to determining leader_AV, and leader_AV decrease to create gaps between platoons
        :param ids_av: the list of leader_AV (small=>large)
        :param ids_f_av: the list of follower_AV
        :return:
        110 km/h = 30.06 m/s
        100 km/h = 27.78 m/s
        80 = 22.22
        70 = 19.44
        60 = 16.67
        '''
        min_dis = 100
        v_max = 27.78  # max velocity of AV
        v_level1 = 19.44  # 22.22m/s => 80km/h; ori 20m/s
        v_level2 = 16.67  # 19.44m/s => 70km/h; ori 15m/s
        ls_av = list(ids_av)
        ls_f_av = list(ids_f_av)
        ls_second_decAV = []
        for id in ls_f_av:
            self.traci.vehicle.setColor(id, (255, 0, 0, 255))  # red
            self.traci.vehicle.setMaxSpeed(id, v_max)
        for id in ls_av:
            self.traci.vehicle.setColor(id, (255, 255, 0, 255))  # yellow
            # platoon space control
            if id not in self.dec_av:
                # self.traci.vehicle.slowDown(id, desire_v, 10)
                # self.traci.vehicle.setSpeed(id, v_level1)
                if id == 'mav792':
                    pass
                self.traci.vehicle.setMaxSpeed(id, v_level1)
                self.dec_av.append(id)
        # check if any leader av is very close to it's preceding vehicle
        for index, id in enumerate(ls_av):  # (small => large)
            leader_info = self.traci.vehicle.getLeader(id)
            if leader_info is not None:
                dis = leader_info[1]
                speed = self.traci.vehicle.getSpeed(id)
                if dis <= min_dis and speed == v_level1:
                    '''
                    if find one AV do not has enough space from its preceding veh, 
                    all veh after this AV need to take a second dec action (to level_2 speed)
                    '''
                    ls_second_decAV = ls_av[index:]
                    break
        for id in ls_second_decAV:
            self.set_hold_speed(id, v_level1, v_level2, 7)
        self.check_recovery(v_level1)

    def set_hold_speed(self, id, ori_v, set_v, hold_time):
        '''
        set veh's speed to set_v and hold_time seconds, then recover to ori_v

        :param id:
        :param ori_v: original speed
        :param set_v: set speed temp
        :param hold_time:
        :return:
        '''
        current_time = self.traci.simulation.getTime()
        if id not in self.recover_time_map:
            # self.traci.vehicle.setSpeed(id, set_v)
            self.traci.vehicle.setMaxSpeed(id, set_v)
            self.recover_time_map[id] = current_time + hold_time

    def check_recovery(self, ori_v):
        '''
        restore AV speed to level1
        :param ori_v:
        :return:
        '''
        current_time = self.traci.simulation.getTime()
        to_remove = []
        for veh_id, recover_time in self.recover_time_map.items():
            if current_time >= recover_time:
                # self.traci.vehicle.setSpeed(veh_id, ori_v)
                self.traci.vehicle.setMaxSpeed(veh_id, ori_v)
                to_remove.append(veh_id)
        for veh_id in to_remove:
            del self.recover_time_map[veh_id]

    def restore_speed_limit2(self, ls_centerA_av):
        '''

        :param ls_vehid: order is not important
        :return:
        '''
        # max_speed = 27.78
        for vid in ls_centerA_av:
            if vid == 'ravh40':
                pass
            if vid in self.ls_speed_ok:
                continue
            current_max = self.traci.vehicle.getMaxSpeed(vid)
            if current_max >= self.max_speed:
                self.ls_speed_ok.append(vid)
                continue
            self.traci.vehicle.setMaxSpeed(vid, self.max_speed)  # restore to 27.78 m/s
            

    def restrict_lane_changeBase(self, veh_id):
        self.traci.vehicle.setParameter(veh_id, "laneChangeModel.lcStrategic", "0.0")
        self.traci.vehicle.setParameter(veh_id, "laneChangeModel.lcSpeedGain", "0.0")

    def restrict_auto_lc(self, ls_id):
        '''
        forbid auto lane_change
        :param ls_id: list of veh id
        :return:
        '''
        for vid in ls_id:
            if vid not in self.no_lc_veh:
                self.traci.vehicle.setLaneChangeMode(vid, 256)
                self.no_lc_veh.add(vid)

    def find_oversizedP_nearbyAV(self, ls_upB_av, dic_platoon_size):
        '''
        find oversized Platoon, get target platoon info
        :param dic_platoon_size: {leader_AV : size, ...}
                ls_upB_av (decrease, big => small)
        :return: dic_current_oversizedP => leader_AV : [platoon_info]]
                dic_current_upBav => leader_AV: [ls_upB_av], all lane_B AV that behind target leader
        '''
        # leader_pos, leader_speed, size
        ls_upB_av_re = ls_upB_av[::-1]  # Reverse to oldest → newest
        dic_current_oversizedP = {}  # oversized platoon
        for leader_id, size in dic_platoon_size.items():
            if size > self.max_team_size:
                '''
                p_info = [head_pos,
                        tail_pos,
                        avg_speed,
                        size]
                '''
                ls_members = self.dic_platoon_members[leader_id]
                tail_id = ls_members[-1]
                head_pos = self.traci.vehicle.getLanePosition(leader_id)
                tail_pos = self.traci.vehicle.getLanePosition(tail_id)
                avg_speed = sum(self.traci.vehicle.getSpeed(vid) for vid in ls_members) / len(ls_members)
                ls_info = [head_pos, tail_pos, avg_speed, size]
                dic_current_oversizedP[leader_id] = ls_info
                self.dic_oversizedPlatoon_info[leader_id] = ls_info
        # nearby lane AV list
        dic_current_upBav = {}
        if dic_current_oversizedP:
            for leader_id, info in dic_current_oversizedP.items():
                leader_pos = info[0]
                for index, upB_av in enumerate(ls_upB_av_re):
                    pos = self.traci.vehicle.getLanePosition(upB_av)
                    if pos < leader_pos:
                        ls_upB_av_filtered = ls_upB_av_re[index:]
                        dic_current_upBav[leader_id] = ls_upB_av_filtered
                        break
        return dic_current_oversizedP, dic_current_upBav

    def check_state(self, id):
        '''
        check followers' state: decoupled free flow mode/coupled following mode
        :param id:
        :return:
        '''
        minGap = 4.5
        tau = 1
        v_except = 20
        dis_buffer = 5
        p_veh_info = self.traci.vehicle.getLeader(id)
        if p_veh_info is None:
            # no leader
            state = 'free_mode'
            return state
        pv_id, dis = p_veh_info
        # when its leader arrive at the end of upstream_0, the dis between this veh and its preceding veh
        dis_real = dis + minGap
        dis_except = minGap + v_except * tau
        if dis_real > dis_except + dis_buffer:
            state = 'free_mode'
        else:
            state = 'following_mode'
        return state

    def get_cor_leader(self, dic_platoon_members, follower_id):
        '''
        according to dic and follower_id, find its leader_id,
        and the follower's position index in that leader's follower list.
        :param dic_platoon_members:
        :param follower_id:
        :return:
        '''
        for leader, followers in dic_platoon_members.items():
            if leader == 'mav13580':
                pass
            if follower_id in followers:
                index = followers.index(follower_id)
                return leader, index
        return None, None

    def non_oversized_platoon(self, dic_platoon_members, dic_current_oversizedP):
        '''
        get current non oversized platoon
        :param dic_platoon_members: all platoon at this time step
        :param dic_current_oversizedP: all overseized platoon at this time step
        :return: dic_nonOversized
        '''
        oversized_leader = list(dic_current_oversizedP.keys())
        all_leader = list(dic_platoon_members.keys())
        non_oversized_leader = list(set(all_leader) - set(oversized_leader))
        dic_nonOversized = {k: dic_platoon_members[k] for k in non_oversized_leader}
        return dic_nonOversized

    def find_sparse_platoon(self, dic_nonOversized, dic_id_preState):
        '''
        identify sparse platoon
        :param dic_id_preState:
        :return: dic_sparse_platoon = {leader_id : first_free_follower, ...}
        '''
        dic_sparse_platoon = {}
        for leader, ls_followers in dic_nonOversized.items():
            if leader == 'mav3287':
                pass
            for follower in ls_followers:
                if (follower in dic_id_preState and dic_id_preState[follower] == 0
                        and leader not in self.dic_AVroleChange):  # temporary measure, future multiple step prediction
                    # tag as sparse platoon
                    first_free_follower = follower
                    dic_sparse_platoon[leader] = first_free_follower
                    break
        return dic_sparse_platoon

    def free_promote(self, dic_sparse_platoon, dic_platoon_members):
        '''
        promote AV_follower (from 2 to 1) between leader_AV and first_free_follower if possible
        :param dic_sparse_platoon: {leader : first_free_follower}
        :param dic_platoon_members: all platoon leader and its followers
        :return:
        '''
        for leader, first_free_follower in dic_sparse_platoon.items():
            platoon_members = dic_platoon_members[leader]
            idx_first_free = platoon_members.index(first_free_follower)
            ls_promote_cands = platoon_members[1:idx_first_free]
            # check if there are any av_follower
            ls_av_only = [vid for vid in ls_promote_cands if 'av' in vid]
            if ls_av_only:
                promote_av = ls_av_only[-1]
                # free_promote
                self.dic_tags[promote_av] = 1
                self.dic_AVroleChange[promote_av] = 'free_promote'
            else:
                promote_av = None
        return

    def predict_following_state(self, dic_id_type, ls_vehid, model=False):
        '''
        250520 updated version: platoon-wise prediction.
        If any follower is predicted as 'free', all subsequent followers in the same platoon
        will be automatically labeled as free without prediction (but still recorded).

        :param dic_id_type: the same as dic_tags = {id:tag, ..., }
               ls_vehid: the order is not important
               # dic_promotedAV: {AV_id: type, ...}, type = 'split' or 'free'
               model: whether to perform prediction using fs_model
        :return:
                0:free; 1:following
                self.dic_id_preState = {id: state,... } # id start from the first follower of the first AV_leader
                the sequence: decrease or increase
        '''
        if not dic_id_type:
            return self.dic_id_preState, self.dic_id_features
        # Only proceed when a new follower appears
        new_follower_id, newest_tag = next(reversed(dic_id_type.items()))  # get the new in veh_id and veh_tag
        # If a new platoon leader appears, reset free_triggered
        if newest_tag == 1:
            self.free_triggered = False
        # Only process new followers (skip if already processed or is a leader)
        if new_follower_id in self.dic_id_features or newest_tag == 1 or new_follower_id not in ls_vehid:
            return self.dic_id_preState, self.dic_id_features
        # == get features == always extract features for training
        arr_select_features = self.get_RFfeatures(new_follower_id)
        if arr_select_features is None:
            return self.dic_id_preState, self.dic_id_features  # or `continue` if used in a loop

        # == prediction ==
        if model:
            # if model == True, predict the state of the newest_follower
            if self.free_triggered:
                # free_triggered = True; No prediction needed, directly mark as free
                pre_state = [0]
            else:
                # free_triggered = False; Perform prediction using model
                pre_state = self.fs_model.predict(arr_select_features)
            self.dic_id_preState[new_follower_id] = pre_state[0]

            if pre_state[0] == 0:  # free
                self.free_triggered = True
        return self.dic_id_preState, self.dic_id_features  # self.dic_id_features includes all id & features

    def get_RFfeatures(self, new_follower_id):
        '''
        get Random Forest features of new_follower_id
        :return: df_select_features
        '''
        if new_follower_id == 'mhv2597':
            pass

        minGap = 4.5
        veh_length = 5
        # == get features ==
        preceding_info = self.traci.vehicle.getLeader(new_follower_id)
        if preceding_info is None:
            return None  # No leading vehicle, skip
        pv_id, pv_dis = preceding_info
        # real dis to the preceding veh; FEATURE 1
        dis_to_pv = pv_dis + minGap + veh_length
        # velocity of preceding veh; FEATURE 2
        v_pv = self.traci.vehicle.getSpeed(pv_id)
        # get pos of this veh; FEATURE 4
        pos_this = self.traci.vehicle.getLanePosition(new_follower_id)
        # get its leader_AV id and index of this veh (COResponding)
        leader_id, index_this = self.get_cor_leader(self.dic_platoon_members, new_follower_id)
        if leader_id is None:
            return None  # No leader found, skip
        # size (veh_num); FEATURE 5
        veh_num = index_this + 1  # size, how many veh between this veh and its leader_AV, start from 0
        # get pos of leader_id
        if leader_id in self.traci.vehicle.getIDList():
            pos_leader = self.traci.vehicle.getLanePosition(leader_id)
        else:
            pos_leader = 1600
        # dis to the leader_AV; FEATURE 3
        dis_to_leaderAV = pos_leader - pos_this
        leader_id_start = leader_id
        features = [new_follower_id, dis_to_pv, v_pv, dis_to_leaderAV, pos_this, veh_num, leader_id_start]
        self.dic_id_features[new_follower_id] = features

        # filtered features
        select_features = [dis_to_pv, v_pv, dis_to_leaderAV, veh_num]  # 1, 2, 3, 5
        # df_select_features = pd.DataFrame([select_features],
        #                                   columns=['dis_to_pv', 'v_pv', 'dis_leaderAV',
        #                                            'size'])  # TODO: dis_leaderAV => dis_to_leaderAV
        arr_select_features = np.array(select_features, dtype=float).reshape(1, -1)
        return arr_select_features

    def record_follower_state2(self, step, length_ih, dic_tags, ls_ihA):
        """
        When an AV leader enters the last 800 meters of inflow_highway_0 for the first time,
        record the current states (free_mode / platoon_follow) of all its platoon followers.

        :param length_ih: total length of inflow_highway_0
        :param dic_tags: {veh_id: tag}, where tag == 1 means AV leader, tag != 1 means follower
        :param ls_ihA: list of vehicle IDs currently on inflow_highway_0
        :return: self.dic_follower_state: {follower_id: [state, leader_id]}
        """

        # Length of the merging control section (last 800 meters)
        length_mc = 800
        # Starting position of the merging control section
        length_pf = length_ih - length_mc
        # 1. Collect all AV leaders (tag == 1)
        ls_leaders = [vid for vid, tag in dic_tags.items() if tag == 1]
        # 2. AV leaders currently located on inflow_highway_0
        ls_ihA_leaders = [vid for vid in ls_ihA if vid in ls_leaders]
        # 3. Leaders that have reached the merging control section (> length_pf)
        ls_mc_leaders = [
            lid for lid in ls_ihA_leaders
            if self.traci.vehicle.getLanePosition(lid) > length_pf
        ]
        # No leader in the merging control section
        if not ls_mc_leaders:
            return self.ls_leader_mc, self.dic_follower_state, self.dic_final_platoon_info

        # Take the most recently arrived leader
        leader_mc = ls_mc_leaders[0]
        if leader_mc == 'mav11232':
            pass
        # Ensure this leader is recorded only once
        if leader_mc in self.ls_leader_mc:
            return self.ls_leader_mc, self.dic_follower_state, self.dic_final_platoon_info
        self.ls_leader_mc.append(leader_mc)
        # 4. Retrieve all followers belonging to this leader's platoon
        platoon_followers = self.dic_platoon_members.get(leader_mc, [])[1:]
        # 5. Record the state of each follower at the moment the leader enters 800m
        for fol in platoon_followers:
            state = self.check_state(fol)  # Determine free_mode or platoon_follow
            self.dic_follower_state[fol] = [state, leader_mc]
        # record final platoon information
        self._get_final_platoon_info(step, self.dic_follower_state)
        return self.ls_leader_mc, self.dic_follower_state, self.dic_final_platoon_info

    def encourage_inner_lane_change(
            self,
            ls_ihA_hv: list,
            length_ih,
            p_to_inner: float = 0.8,
            weaving_influence_range: float = 200.0,
    ):
        """
        Let HVs on the outer lane (lane 0) near the ramp
        have a random tendency to move to the inner lane (lane 1).

        Parameters
        ----------
        ls_ihA_hv : list
            Vehicle IDs of HVs currently on outer lane (lane 0).
        p_to_inner : float
            Probability (0–1) that a HV attempts to change to the inner lane.
        weaving_influence_range : float
            Distance range (m) on inflow_highway influenced by the weaving which
            the inner-lane changing.
        """
        if not ls_ihA_hv:
            return
        for veh_id in ls_ihA_hv:
            if veh_id in self.encourage_change_mark:
                continue
            try:
                lane_id = self.traci.vehicle.getLaneIndex(veh_id)
                if lane_id != 0:
                    continue
                lane_pos = self.traci.vehicle.getLanePosition(veh_id)  # encourage only vehicles on lane 0
                dis_to_weaving = length_ih - lane_pos  # distance to the weaving section
            except Exception:
                continue  # vehicle might have left the network
            # only apply within upstream distance (e.g. 200 m before ramp)
            if dis_to_weaving > weaving_influence_range:
                continue
            self.encourage_change_mark.add(veh_id)
            # random tendency: some HVs will change, some not
            if random.random() > p_to_inner:
                continue
            try:
                # lane 1 = inner lane (away from ramp)
                self.traci.vehicle.changeLane(veh_id, 1, 3)
            except Exception:
                continue  # skip if unsafe or invalid

    def restrict_av_lc(self, lc_av, ls_av):
        '''
        restrict lane_changing behaviour of av, for training RF model
        :param lc_av:
        :param ls_av:
        :return:
        '''
        if lc_av:
            return  # lc_av = True
        for vid in ls_av:  # lc_av = False
            if vid not in self.no_lc_av:
                self.traci.vehicle.setLaneChangeMode(vid, 0)
                self.no_lc_av.add(vid)

    def manage_lc_behavior_near_ws(self, lc, ls_ihAB_hv, ls_wsBC_hv, length_ih,
                                   p_to_inner=0.8, weaving_influence_range=200.0):
        """
        Adaptive lane-changing control near ramp.
        """
        if not lc:  # if lc (lane_changing) is False, just skip
            return
        # if lc is True
        self._disable_keepRight_in_weaving(ls_ihAB_hv, ls_wsBC_hv)
        self._encourage_outer_to_inner(ls_ihAB_hv, length_ih, p_to_inner, weaving_influence_range)
        self._cancel_pending_changes_on_center()
        self._restore_keepRight_outside_weaving(length_ih, weaving_influence_range)

    def update_member_to_leader(self, dic_platoon_members):
        """Build a reverse mapping from follower ID to its platoon leader."""
        dic_member_to_leader = {}
        for leader_id, members in dic_platoon_members.items():
            for veh_id in members:
                dic_member_to_leader[veh_id] = leader_id
        return dic_member_to_leader

    def _get_final_platoon_info(self, step, dic_follower_state):
        """
        Count ONLY the followers belonging to the newest AV leader in dic_follower_state.
        Ignore all previous platoons from earlier leaders.
        """

        if not dic_follower_state:
            return
        # 1. Find the newest leader (the last leader_id in the dict order)
        # Since dict preserves insertion order in Python 3.7+
        last_item = next(reversed(dic_follower_state.items()))
        newest_leader = last_item[1][1]  # info[1] = leader_id
        # 2. Count followers that belong to ONLY this newest leader
        #    and are not free_mode
        count = 0
        for vid, (state, leader_id) in dic_follower_state.items():
            if leader_id == 'mav4690':
                pass
            if leader_id == newest_leader and state != "free_mode":
                count += 1
        # 3. If no valid followers, do not record
        if count == 0:
            return
        # 4. Construct platoon string: "A" + "H" * count
        platoon_string = "A" + "H" * count
        if platoon_string == 'A':
            pass
        # 5. Save result
        self.dic_final_platoon_info[step//10] = platoon_string
        # update to dic_avhid_ptype
        self.data_recorder.get_avhid_ptype(m_dpt_type={leader_id: platoon_string})

    def _disable_keepRight_in_weaving(self, ls_ihAB_hv, ls_wsBC_hv):
        "'_' for internal use within a class or module and not part of the public API."
        all_influenced_hv = set(ls_ihAB_hv) | set(ls_wsBC_hv)
        for veh_id in all_influenced_hv:
            if veh_id in self.lcKeepRight_disabled:
                continue
            try:
                self.traci.vehicle.setParameter(veh_id, "laneChangeModel.lcKeepRight", "0")
                self.lcKeepRight_disabled.add(veh_id)
            except Exception:
                continue

    def _restore_keepRight_outside_weaving(self, length_ih, weaving_influence_range):
        """
        Restore lcKeepRight=1 for HVs that have left the weaving-influenced region.
        The weaving zone = last `weaving_influence_range` meters of ih + entire ws.
        Vehicles entering 'center' or moving upstream beyond this zone are reset.
        """
        hv_to_reset = set()
        threshold = length_ih - weaving_influence_range
        for veh_id in list(self.lcKeepRight_disabled):
            try:
                lane_pos = self.traci.vehicle.getLanePosition(veh_id)
                road_id = self.traci.vehicle.getRoadID(veh_id)
            except Exception:
                continue

            # case 1: upstream (ih, before influence zone)
            if road_id.startswith("ih") and lane_pos < threshold:
                hv_to_reset.add(veh_id)

            # case 2: downstream (entered center)
            elif road_id.startswith("center"):
                hv_to_reset.add(veh_id)

            # case 3: still inside weaving (ih tail or ws)
            else:
                continue

        for veh_id in hv_to_reset:
            try:
                self.traci.vehicle.setParameter(veh_id, "laneChangeModel.lcKeepRight", "1")
            except Exception:
                pass
            self.lcKeepRight_disabled.discard(veh_id)

    def _encourage_outer_to_inner(self, ls_ihAB_hv, length_ih, p_to_inner, weaving_influence_range):
        for veh_id in ls_ihAB_hv:
            if veh_id in self.encourage_change_mark:
                continue
            try:
                lane_pos = self.traci.vehicle.getLanePosition(veh_id)
                dis_to_weaving = length_ih - lane_pos
                if dis_to_weaving > weaving_influence_range:
                    continue
                lane_id = self.traci.vehicle.getLaneIndex(veh_id)
                if lane_id != 0:
                    continue
                self.encourage_change_mark.add(veh_id)
                if random.random() <= p_to_inner:
                    self.traci.vehicle.changeLane(veh_id, 1, 1)
                    self.pending_changes.add(veh_id)  # Record: this vehicle received a change command
            except Exception:
                continue

    def _cancel_pending_changes_on_center(self):
        """
        Cancel any pending lane-change commands that were issued in the weaving region
        but never successfully executed before entering 'center'.
        """
        for vid in list(self.pending_changes):
            try:
                road_id = self.traci.vehicle.getRoadID(vid)
            except Exception:
                self.pending_changes.discard(vid)
                continue

            if road_id.startswith("center"):
                if vid == 'mbhv1162' or vid == 'mvhv1198':
                    pass
                try:
                    current_lane = self.traci.vehicle.getLaneIndex(vid)
                    # Cancel by re-commanding the current lane (duration=0)
                    self.traci.vehicle.changeLane(vid, current_lane, 0)
                except Exception:
                    pass
                self.pending_changes.discard(vid)
