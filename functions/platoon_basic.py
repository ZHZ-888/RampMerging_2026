# platoon_basic.py
# Core platoon management operations: tagging, size tracking, speed control, recording

import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)


class PlatoonBasic:
    def __init__(self, traci, data_recorder):
        self.traci = traci
        self.data_recorder = data_recorder

        self.max_speed = self.data_recorder.max_speed # 27.78 m/s => 100km/h
        # for leader AV to create space between platoons and collect followers
        # PLAN A
        self.speed_level3 = 25
        self.speed_level2 = 22.22  # 19.44 m/s (70km/h);
        self.speed_level1 = 16.67  # 16.67 m/s (60km/h)
        # PLAN B
        # self.speed_level3 = 19.44
        # self.speed_level2 = 19.44
        # self.speed_level1 = 16.67  # 16.67 m/s (60km/h)

        self.max_team_size = 11

        self.dec_av = []
        self.dic_tags = {}
        self.recover_time_map = {}  # record av speed setting recover time
        self.recover_speed_map = {}
        self.ls_speed_ok = []  # av_id that speed restore back to max (27.78 m/s)
        self.ls_speed_level3 = []
        self.dic_platoon_size = {}  # all leaderAV and its platoon size
        self.dic_platoon_members = {}  # all leaderAV and its members

        self.dic_AVroleChange = {}  # dic_AVroleChange = {AV_id: type, ...} record AV changed its role
        self.ls_leader_fol_states_checked = []

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
        :return: updated dic_tags,
                    ls_leader_AV,
                    ls_follower_AV
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
        min_dis = 150 # 100
        speed_level2 = self.speed_level2  # 22.22m/s => 80km/h; 19.44m/s => 70km/h
        speed_level1 = self.speed_level1  # 16.67m/s => 60km/h

        # make sure all veh in ls_leader_av and ls_follower_av in ls_vehid
        ls_leader_av = [id for id in ls_leader_av if id in ls_vehid]
        ls_follower_av = [id for id in ls_follower_av if id in ls_vehid]
        ls_second_decAV = []
        for id in ls_follower_av:
            self.traci.vehicle.setColor(id, (255, 0, 0, 255))  # red
            self.traci.vehicle.setMaxSpeed(id, self.data_recorder.max_speed)
            if id in self.dec_av:
                self.dec_av.remove(id)
        for leader in ls_leader_av:
            self.traci.vehicle.setColor(leader, (255, 255, 0, 255))  # yellow
            # platoon space control
            if leader not in self.dec_av: # av that has dec to level1 speed?
                self.traci.vehicle.setMaxSpeed(leader, speed_level2)
                self.dec_av.append(leader)
        # check if any leader av is very close to it's preceding vehicle
        for index, leader in enumerate(ls_leader_av):  # ascending order
            preceding_veh_info = self.traci.vehicle.getLeader(leader)
            if preceding_veh_info is not None:
                if leader == 'm_av385':
                    pass
                dis_to_pv = preceding_veh_info[1]
                speed_leader = self.data_recorder.get_vid_states(leader)['v']
                current_max_speed = self.traci.vehicle.getMaxSpeed(leader)
                if dis_to_pv < min_dis and current_max_speed != speed_level1 and speed_leader <= speed_level2:
                    '''
                    if find LEADER do not has enough space from its preceding veh, 
                    this LEADER (and other LEADER after this LEADER) need to 
                    take a second dec action (to level_1 speed)
                    '''
                    ls_second_decAV = ls_leader_av[index:]
                    break
        self._set_hold_speed2(ls_second_decAV, speed_level1, min_dis)

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

    def restore_speed_limit3(self, step, ls_leader, ls_m_leader_up_asc):
        """
        Restore the maximum speed limit for platoon leaders based on their spatial location.

        Control Strategies:
        1. Merging Control Zone: Unconditional speed restoration.
        2. Platoon Formation Zone: Conditional speed restoration.
           - Condition A: All own followers are in 'following_mode'.
           - Condition B: ALL preceding leaders upstream have successfully restored speed.
                          (If a front leader waits, all downstream leaders must also wait).

        Parameters
        ----------
        ls_leader : list
            List of all leaders on inflow_highway_0. ascending order
        ls_m_leader_up_asc : list
            List of leaders that have already entered the merging control section.
        self.dic_platoon_members = {'mav19': ['mav19', 'mhv46', 'mhv64'],
                                    'veh_2': ['veh_2', 'veh_4']}
        """
        speed_level3 = self.speed_level3  # 25 m/s => 90km/h
        leaders_on_merging_control = set(ls_m_leader_up_asc)

        # CHAIN REACTION FLAG:
        # If any upstream leader fails to accelerate (waiting for followers),
        # this becomes True and blocks ALL subsequent leaders behind it.
        front_blocked = False

        # NOTE: Ensure ls_leader is ordered from FRONT to BACK (Downstream to Upstream).
        for i, leader in enumerate(ls_leader):
            if leader == 'mav40':
                pass
            # If already at max speed, it doesn't block anyone behind it. Skip.
            if leader in self.ls_speed_level3:
                continue

            current_max = self.traci.vehicle.getMaxSpeed(leader)
            # if current_max >= self.max_speed and leader in self.dec_av:
            if current_max >= speed_level3 and leader in self.dec_av:
                self.ls_speed_level3.append(leader)
                continue

            # STRATEGY A: Merging Zone
            if leader in leaders_on_merging_control:
                # Unconditional acceleration for merging zone
                # self.traci.vehicle.setMaxSpeed(leader, self.max_speed)
                self.traci.vehicle.setMaxSpeed(leader, speed_level3)

            # STRATEGY B: Formation Zone
            else:
                # NEW CONSTRAINT: Check if blocked by a leader ahead
                if front_blocked:
                    # A leader ahead is waiting for its followers.
                    # This leader MUST wait too, regardless of its own platoon state.
                    continue

                # Check if this is the FURTHEST UPSTREAM leader (the newest one spawned)
                is_newest_leader = (i == len(ls_leader) - 1)
                if is_newest_leader:
                    # it MUST wait because more followers might spawn behind it.
                    front_blocked = True  # Block state
                    continue

                ls_followers = self.dic_platoon_members.get(leader, [])[1:]

                # Single vehicle (no followers): accelerates immediately
                if not ls_followers:
                    self.traci.vehicle.setMaxSpeed(leader, speed_level3)
                    continue

                # Platoon integrity check
                all_following = True
                for follower in reversed(ls_followers):
                    if follower == 'mhv228':
                        pass
                    if self._check_state(follower) == 'free_mode':
                        all_following = False
                        break

                if all_following:
                    # Platoon is intact, leader accelerates
                    self.traci.vehicle.setMaxSpeed(leader, speed_level3)
                    for follower in ls_followers: # record followers's state as '1' (following mode)
                        self.traci.vehicle.setColor(follower, (144, 238, 144))
                        self.dic_follower_state[follower] = ['following_mode', leader]
                    self.ls_leader_fol_states_checked.append(leader) # record this leader then no need to check its followers' state
                    # record final platoon information and pass to merging controller for later use
                    self._get_final_platoon_info(step, self.dic_follower_state)
                else:
                    # PLATOON BROKEN! This leader cannot accelerate.
                    # TRIGGER CHAIN REACTION: Block all subsequent leaders behind this one!
                    front_blocked = True

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
        When an AV leader enters the merging control section for the first time,
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
        if leader_mc_newest in self.ls_leader_fol_states_checked:
            return self.dic_follower_state, self.dic_final_platoon_info
        self.ls_leader_fol_states_checked.append(leader_mc_newest)
        # 2. Retrieve all followers belonging to this leader's platoon
        platoon_followers = self.dic_platoon_members.get(leader_mc_newest, [])[1:]
        # 3. Record the state of each follower at the moment the leader enters 800m
        free_mode_detected = False # detect any free_mode fol, then all fol (same leader) behind it are in free_mode
        for fol in platoon_followers:
            state = self._check_state(fol)  # Determine free_mode or following_mode
            if free_mode_detected or state == 'free_mode':
                state = 'free_mode'
                free_mode_detected = True
            self.dic_follower_state[fol] = [state, leader_mc_newest]
        self.data_recorder.dic_follower_state = self.dic_follower_state
        # record final platoon information
        self._get_final_platoon_info(step, self.dic_follower_state)
        return self.dic_follower_state, self.dic_final_platoon_info # change to leftmost lane and keep until the end of the road

    def set_follower_color(self):
        '''
        set followers color as light green

        dic_follower_state = {follower_id: [state, leader_id]}
        state = free_mode/following_mode

        av_leader: yellow, av_follower: red
        action_av: flashing between yellow and red
        hv_follower: green, hv_free: white
        '''
        light_green = (144, 238, 144) # follower's color
        dic_follower_state = self.data_recorder.dic_follower_state
        try:
            ls_ihAB_hv = self.data_recorder.dic_vid_groups['ls_ihAB_hv']
            for vid in ls_ihAB_hv:
                if self.traci.vehicle.getColor(vid) == light_green:
                    continue
                item = dic_follower_state.get(vid)
                following_state = item[0] if item else None
                if following_state == 'following_mode':
                    self.traci.vehicle.setColor(vid, light_green)  # light green; green (0, 255, 0)
        except:
            pass

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

    def _set_hold_speed2(self, ls_second_decAV, set_v, gap_threshold):
        """
        Apply a temporary speed limit and recover it based on gap conditions,
        with a leader-first recovery constraint.

        Description:
            Vehicles are temporarily assigned a reduced maximum speed (set_v).
            A vehicle can recover its original speed only when:
                1. The gap to its leader exceeds a predefined threshold.
                2. Its leader has already recovered (or is not under control).

            This ensures a front-to-back recovery order and avoids gap shrinkage
            caused by rear vehicles accelerating earlier than their leaders.

        Parameters:
            ls_second_decAV (list): list of vehicle IDs (ordered from front to back)
            set_v (float): temporary speed limit
            gap_threshold (float): minimum gap required for speed recovery (meters)

        Internal state:
            self.recover_speed_map = {vid: original_speed, ...}
        """

        # Step 1: Collect all currently controlled vehicles
        controlled_ids = list(self.recover_speed_map.keys())
        # fileter out vehicles that left
        net_vids = self.data_recorder.dic_vid_groups['ls_vehid'] # all vehicle in this step
        controlled_ids = [
            vid for vid in controlled_ids
            if vid in net_vids
        ]

        # Step 2: Sort vehicles from front to back based on lane position
        veh_positions = {}
        for vid in controlled_ids:
            if vid == 'mb_av7600':
                pass
            try:
                veh_positions[vid] = self.data_recorder.get_vid_states(vid)['pos'] # == None; may have crash
            except:
                veh_positions[vid] = -1  # fallback if vehicle is missing

        for vid in controlled_ids:
            if vid is None:
                print("Error: vid is None")
                raise ValueError("vid is None before sorting")

            if vid not in veh_positions:
                print(f"Error: {vid} is missing in veh_positions")
                raise KeyError(f"{vid} is missing in veh_positions")

            if veh_positions[vid] is None:
                print(f"Error: {vid} has None position")
                print("controlled_ids:", controlled_ids)
                print("veh_positions:", veh_positions)
                raise ValueError(f"{vid} has None position before sorting")

        sorted_vids = sorted(controlled_ids, key=lambda x: veh_positions[x], reverse=True)
        recovered_set = set()  # vehicles recovered in this step
        to_remove = []  # vehicles to remove from tracking map

        # Step 3: Check recovery condition with leader-first constraint
        for vid in sorted_vids:
            ori_v = self.recover_speed_map[vid]
            try:
                # Retrieve leader information (leader_id, gap)
                leader_info = self.traci.vehicle.getLeader(vid)
                if leader_info is None:
                    # No leader (free-flow condition) → recover immediately
                    self.traci.vehicle.setMaxSpeed(vid, ori_v)
                    recovered_set.add(vid)
                    to_remove.append(vid)
                    continue
                leader_id, gap = leader_info

                # Enforce leader-first recovery:
                # if leader is still under control and not yet recovered, skip
                if leader_id in self.recover_speed_map and leader_id not in recovered_set:
                    continue
                # Recover if gap condition is satisfied
                if gap >= gap_threshold:
                    self.traci.vehicle.setMaxSpeed(vid, ori_v)
                    recovered_set.add(vid)
                    to_remove.append(vid)

            except Exception:
                # Handle edge cases (vehicle removed from simulation, etc.)
                to_remove.append(vid)

        # Step 4: Remove recovered vehicles from tracking map
        for vid in to_remove:
            if vid in self.recover_speed_map:
                del self.recover_speed_map[vid]

        # Step 5: Apply speed limit to new vehicles
        for vid in ls_second_decAV:
            if vid not in self.recover_speed_map:
                # Store original max speed
                ori_v = self.traci.vehicle.getMaxSpeed(vid)
                # Apply temporary speed constraint
                self.traci.vehicle.setMaxSpeed(vid, set_v)
                # Save for future recovery
                self.recover_speed_map[vid] = ori_v

    def _check_state(self, id):
        '''
        check followers' state: decoupled free flow mode/coupled following mode
        IDM-based headway check: normal range (1.2T ~ 2.0T)
        here set factor as 2.2 T
        :param id:
        :return:
        '''
        minGap = 2.5 # original: 4.5; default 2.5
        tau = 1.5 # standard: 1s
        following_headway_factor = 2.2
        v_expect = self.speed_level2
        p_veh_info = self.traci.vehicle.getLeader(id)
        if p_veh_info is None:
            # no leader
            state = 'free_mode'
            return state
        pv_id, dis = p_veh_info
        # when its leader arrive at the end of upstream_0, the dis between this veh and its preceding veh
        dis_real = dis + minGap
        dis_expect = v_expect * tau + minGap
        if dis_real > dis_expect * following_headway_factor:
            state = 'free_mode'
        else: # dis_real <= dis_expect * factor
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
            if leader_id == 'mav40':
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
        if len(platoon_string) > self.max_team_size:
            pass
        # 5. Save result
        self.dic_final_platoon_info[step//10] = platoon_string
        # update to dic_avhid_ptype; this is truly pass to merging controller
        self.data_recorder.get_avhid_ptype(m_dpt_type={newest_leader: platoon_string})

