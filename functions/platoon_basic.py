# platoon_basic.py
# Core platoon management operations: tagging, size tracking, speed control, recording

import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)


class PlatoonBasic:
    def __init__(self, traci, data_recorder):
        self.traci = traci
        self.data_recorder = data_recorder
        self.max_speed = self.data_recorder.max_speed
        self.max_team_size = 11

        self.dec_av = []
        self.dic_tags = {}
        self.recover_time_map = {}  # record av speed setting recover time
        self.ls_speed_ok = []  # av_id that speed restore back to max (27.78 m/s)
        self.dic_platoon_size = {}  # all leaderAV and its platoon size
        self.dic_platoon_members = {}  # all leaderAV and its members

        self.dic_AVroleChange = {}  # dic_AVroleChange = {AV_id: type, ...} record AV changed its role
        self.ls_leader_mc_checked = []

        self.ls_leader_AV = []
        self.ls_follower_AV = []
        self.ls_ihA_lastStep = []  # ls_upA (upstream AV) last Step

        self.dic_follower_state = {}  # state of each followers; free_mode/following_mode
        self.dic_final_platoon_info = {}  # m_dpt_type



    def get_platoon_size3(self, ls_ihA, ls_leader):
        '''
        get the platoon size/platoon members for each LEADER currently on the road
        record each platoon members
        :param ls_ihA: ['mhv3305', 'mav3287', 'mhv3171', ...] descending, vehicle list on inflow_highway innner
        :param ls_leader: all leaderAV CURRENTLY on upstream_0, for example: ['mav2744', 'mav3038']
        :return: dic_id_size = {'mav158': 10, 'mav44': 4, 'mav661': 2},
                        current ls_leader and its corresponding size
                 dic_platoon_members = {
                                        'mav19': ['mav19', 'mhv46', 'mhv64', 'mhv74', 'mhv86'],
                                        "veh_2": ["veh_2", "veh_4"]
                                        }, current ls_leader and its corresponding members
                 self.dic_platoon_members (all history platoon and platoon members)
        '''
        dic_leaderAV_index = {}  # leader_id and its index in the traffic; {'mav2744': 1, 'mav3038': 8}
        dic_platoon_size = {}
        dic_platoon_members = {}

        ls_ihA_asc = ls_ihA[::-1]
        for leader in ls_leader:
            idx_ih = ls_ihA_asc.index(leader)  # idx on inflow_highway_0 (inner)
            dic_leaderAV_index[leader] = idx_ih

        # get leader's corresponding size
        leaders = list(dic_leaderAV_index.keys())
        # index: one position; indices: multiple positions
        indices = list(dic_leaderAV_index.values())
        for i in range(len(indices)):
            start = indices[i]
            end = indices[i + 1] if i < len(indices) - 1 else len(ls_ihA_asc)
            platoon = ls_ihA_asc[start:end]
            dic_platoon_size[leaders[i]] = len(platoon)
            dic_platoon_members[leaders[i]] = platoon
            self.dic_platoon_size[leaders[i]] = len(platoon)
            self.dic_platoon_members[leaders[i]] = platoon
        # record leaderAV on mainline
        self.data_recorder.ls_m_leader_his_asc = list(self.dic_platoon_size.keys())
        return self.dic_platoon_size, dic_platoon_size, dic_platoon_members

    def tag_vehicles13(self, ls_ihA, max_team_size=11):
        '''
        260301 crucial update, only keep dic_tags for vehicles still on current road network

        label each vehicle: 0 => follower_HV, 1 => leader_AV, 2 => follower_AV;
        also for split_promote (promote a follower_AV to leader_AV to split oversized platoon)

        :param ls_ihA: vehicle list ordered from newest to oldest
               ls_ihA_asc: oldest to newest
        :param max_team_size: maximum allowed platoon size
        :return: updated dic_tags, ls_leader_AV, ls_follower_AV
        '''
        self.max_team_size = max_team_size
        ls_ihA_asc = ls_ihA[::-1]  # Reverse to oldest → newest

        if self.ls_ihA_lastStep != ls_ihA:
            # === get new_dic, make sure the order is correct ===
            # find the first id in ls_ihA_asc that already exists in dic_tags
            anchor_id = next((id for id in ls_ihA_asc if id in self.dic_tags), None)
            # Keep only the part of dic_tags before the anchor_id (maintain order)
            new_dic = {}
            for k in self.dic_tags:
                if k == anchor_id:
                    break
                new_dic[k] = self.dic_tags[k]
            # Extend new_dic with all IDs from ls_ihA_asc, assigning temporary value -1
            # This overwrites existing keys and inserts new ones in the given order
            for id in ls_ihA_asc:
                new_dic[id] = -1
            self.dic_tags = new_dic

            # === tag every id in ls_ihA_asc ===
            # current_leader= next((k for k in reversed(self.dic_tags) if self.dic_tags[k] == 1), None)
            current_leader = None
            current_team_size = 0

            for i, id in enumerate(ls_ihA_asc):
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
                        for j in range(i + 1, len(ls_ihA_asc)):
                            id_next = ls_ihA_asc[j]
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
                            if id == 'mav4686':
                                pass
                            self.dic_AVroleChange[id] = 'split_promote'
                        self.dic_tags[id] = 1  # Mark as leader
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

            self.ls_ihA_lastStep = ls_ihA

            # filter out vehicles that have left the road network
            ls_vehid = self.data_recorder.dic_vid_groups['ls_vehid'] # all vehicle in this step
            self.dic_tags = {k: v for k, v in self.dic_tags.items() if k in ls_vehid}

            # Update leader and follower lists for current control section
            dic_leader_AV = {k: v for k, v in self.dic_tags.items() if v == 1}
            dic_follower_AV = {k: v for k, v in self.dic_tags.items() if v == 2}

            dic_leader_AV_c = {k: v for k, v in dic_leader_AV.items() if k in ls_ihA}
            dic_follower_AV_c = {k: v for k, v in dic_follower_AV.items() if k in ls_ihA}

            self.ls_leader_AV = list(dic_leader_AV_c.keys())
            self.ls_follower_AV = list(dic_follower_AV_c.keys())

        return self.dic_tags, self.ls_leader_AV, self.ls_follower_AV, self.dic_AVroleChange

    def form_platoon3(self, ls_vehid, ls_leader_av, ls_follower_av):
        '''
        based on desire_v to determining leader_AV, and leader_AV decrease to create gaps between platoons
        :param ls_leader_av: the list of leader_AV (asc; small=>large)
        :param ls_follower_av: the list of follower_AV
        :return:
            110 km/h = 30.06 m/s
            100 km/h = 27.78 m/s
            90 km/h = 25.00 m/s
            80 = 22.22
            70 = 19.44
            60 = 16.67
        '''
        min_dis = 100

        v_level1 = 19.44  # 22.22m/s => 80km/h; ori 20m/s
        v_level2 = 16.67  # 19.44m/s => 70km/h; ori 15m/s
        # make sure all veh in ls_leader_av and ls_follower_av in ls_vehid
        ls_leader_av = [id for id in ls_leader_av if id in ls_vehid]
        ls_follower_av = [id for id in ls_follower_av if id in ls_vehid]
        ls_second_decAV = []
        for id in ls_follower_av:
            if id == 'mav281':
                pass
            self.traci.vehicle.setColor(id, (255, 0, 0, 255))  # red
            self.traci.vehicle.setMaxSpeed(id, self.data_recorder.max_speed)
            if id in self.dec_av:
                self.dec_av.remove(id)
        for leader in ls_leader_av:
            if leader == 'mav281':
                pass
            self.traci.vehicle.setColor(leader, (255, 255, 0, 255))  # yellow
            # platoon space control
            if leader not in self.dec_av: # av that has dec to level1 speed?
                self.traci.vehicle.setMaxSpeed(leader, v_level1)
                self.dec_av.append(leader)
        # check if any leader av is very close to it's preceding vehicle
        for index, leader in enumerate(ls_leader_av):  # ascending order
            if leader == 'mav281':
                pass
            preceding_veh_info = self.traci.vehicle.getLeader(leader)
            if preceding_veh_info is not None:
                dis_to_pv = preceding_veh_info[1]
                # speed = self.traci.vehicle.getSpeed(leader)
                speed_leader = self.data_recorder.get_vid_states(leader)['v']
                if dis_to_pv < min_dis and speed_leader == v_level1:
                    '''
                    if find LEADER do not has enough space from its preceding veh, 
                    this LEADER (and other LEADER after this LEADER) need to 
                    take a second dec action (to level_2 speed)
                    '''
                    ls_second_decAV = ls_leader_av[index:]
                    break
        self._set_hold_speed(ls_second_decAV, v_level2, 7)

    def restore_speed_limit2(self, ls_av):
        '''
        restor av max_speed to 27.78 m/s
        max_speed = 27.78
        :param ls_av: list of AV IDs
        :return:
        '''

        for vid in ls_av:
            if vid in self.ls_speed_ok:
                continue
            current_max = self.traci.vehicle.getMaxSpeed(vid)
            if current_max >= self.max_speed:
                self.ls_speed_ok.append(vid)
                continue
            self.traci.vehicle.setMaxSpeed(vid, self.max_speed)  # restore to 27.78 m/s

    def get_cor_leader(self, follower_id):
        '''
        according to dic and follower_id, find its leader_id,
        and the follower's position index in that leader's follower list.
        :param self.dic_platoon_members = {leader: [leader, follower1, follower2,...],...}}
        :param follower_id:
        :return:
        '''
        for leader, followers in self.dic_platoon_members.items():
            if follower_id in followers:
                index = followers.index(follower_id)
                return leader, index
        return None, None

    def update_member_to_leader(self, dic_platoon_members):
        """Build a reverse mapping from follower ID to its platoon leader.
        dic_member_to_leader = {follower_id: leader_id,...}
        """
        dic_member_to_leader = {}
        for leader_id, members in dic_platoon_members.items():
            for veh_id in members:
                dic_member_to_leader[veh_id] = leader_id
        return dic_member_to_leader

    def record_follower_state2(self, step):
        """
        When an AV leader enters the last 800 meters of inflow_highway_0 for the first time,
        record the current states (free_mode / following_mode) of all its platoon followers.

        :param
        :return: self.dic_follower_state: {follower_id: [state, leader_id]}
        """

        # 1. Leaders that have reached the merging control section (> length_pf)
        ls_mc_leaders = self.data_recorder.dic_vid_groups['ls_m_leader_up']
        # No leader in the merging control section
        if not ls_mc_leaders:
            return self.dic_follower_state, self.dic_final_platoon_info
        # Take the most recently arrived leader
        leader_mc_newest = ls_mc_leaders[0]
        # Ensure this leader is recorded only once
        if leader_mc_newest in self.ls_leader_mc_checked:
            return self.dic_follower_state, self.dic_final_platoon_info
        self.ls_leader_mc_checked.append(leader_mc_newest)
        # 2. Retrieve all followers belonging to this leader's platoon
        platoon_followers = self.dic_platoon_members.get(leader_mc_newest, [])[1:]
        # 3. Record the state of each follower at the moment the leader enters 800m
        for fol in platoon_followers:
            state = self._check_state(fol)  # Determine free_mode or following_mode
            self.dic_follower_state[fol] = [state, leader_mc_newest]
        self.data_recorder.dic_follower_state = self.dic_follower_state
        # record final platoon information
        self._get_final_platoon_info(step, self.dic_follower_state)
        return self.dic_follower_state, self.dic_final_platoon_info # change to leftmost lane and keep until the end of the road

    def _set_hold_speed_ori(self, id, set_v, hold_time):
        '''
        Set vehicle speed to set_v for hold_time seconds, then auto-recover to ori_v.
        Also checks and recovers any expired speed holds.
        params:
            self.recover_time_map = {veh_id: (target_speed, recover_time), ...}
        '''
        current_time = self.traci.simulation.getTime()

        # Check and recover expired holds
        to_remove = []
        for veh_id, (target_speed, recover_time) in self.recover_time_map.items():
            if veh_id == 'mbav142892':
                pass
            if current_time >= recover_time:
                self.traci.vehicle.setMaxSpeed(veh_id, target_speed)
                to_remove.append(veh_id)
        for veh_id in to_remove:
            del self.recover_time_map[veh_id]

        # Set new hold (store original speed for recovery)
        if id not in self.recover_time_map:
            ori_v = self.traci.vehicle.getMaxSpeed(id)  # Get current max speed
            self.traci.vehicle.setMaxSpeed(id, set_v)
            self.recover_time_map[id] = (ori_v, current_time + hold_time)

    def _set_hold_speed(self, ls_second_decAV, set_v, hold_time):
        '''
        Set vehicle speed to set_v for hold_time seconds, then auto-recover to ori_v.
        Also checks and recovers any expired speed holds.
        params:
            self.recover_time_map = {vid: (target_speed, recover_time), ...}
        '''
        current_time = self.traci.simulation.getTime()

        # Check and recover expired holds
        to_remove = []
        for vid, (target_speed, recover_time) in self.recover_time_map.items():
            if current_time >= recover_time:
                self.traci.vehicle.setMaxSpeed(vid, target_speed)
                to_remove.append(vid)
        for vid in to_remove:
            del self.recover_time_map[vid]

        # Set new hold (store original speed for recovery)
        for id in ls_second_decAV:
            if id not in self.recover_time_map:
                ori_v = self.traci.vehicle.getMaxSpeed(id)  # Get current max speed
                self.traci.vehicle.setMaxSpeed(id, set_v)
                self.recover_time_map[id] = (ori_v, current_time + hold_time)

    def _check_state(self, id):
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
        self.data_recorder.get_avhid_ptype(m_dpt_type={newest_leader: platoon_string})
