'''
merging_control_jam.py
TODO: call traci to get time many times, maybe once is enough
'''

from functions import print_control as prc
from functions import v2x_disturbance as v2x

class MergingControlJam:
    def __init__(self, traci, instance_dr, instance_vcfunc, loss_rate, ml):
        self.traci = traci
        self.data_recorder = instance_dr
        self.vcfunc = instance_vcfunc
        self.ml = ml # multi-lane?
        self.dic_vid_groups = {}
        self.dic_id_speed = self.data_recorder.dic_speed
        self.first_stop_recorded = [] # ramp av only stop once
        self.first_resume_recorded = []  # ramp av only resume once
        self.timing = False
        self.dic_r_platoon_travel_time = None # tt => travel time from
        self.stop_state = False
        self.resume_state = False
        self.stop_times = {}
        self.resume_times = {}
        self.r_leader_stop = None # current stop r leader id; only one stop at the front is enough
        self.ls_skip_stop = [] # those r_leader can pass with it's preceding r_leader platoon doesn't need to stop
        self.mavh_action_dic = {} # the action dic of m_leader
        self.dic_disire_reach_ts = {} # drt => desire reaching timestamp
        # m_leader and its action parameters; self.dic_mavh_actionP => self.dic_m_leader_action_params
        self.dic_m_leader_action_params = {'':[]}
        self.ls_r_leader_proper = [] # list of ramp leader before acc (ls of ramp_proper leader)
        self.ls_r_proper = [] # list of ramp veh before acc
        self.mavh_acting = False
        self.loss_rate = loss_rate
        # self.delay_buffer = v2x.UpdateDelayBuffer(loss_rate=self.loss_rate)
        self.action_buffer = v2x.UpdateDelayBuffer(loss_rate=self.loss_rate)
        self.timing_buffer = v2x.UpdateDelayBuffer(loss_rate=self.loss_rate)
        self.last_action_payload = None
        self.last_timing_payload = None
        try:
            self.length_ih = self.traci.lane.getLength('inflow_highway_0')
        except:
            self.length_ih = self.traci.lane.getLength('upstream_0')
        self.r_leader_acc_dur = 12 # the time needed for ramp AV leader moving from stop point to the merging section (weaving section)
        self.dic_mplatoon_et = {} # dic_mplatoon_et: {m_leader:[platoon_type, ts_head, ts_tail, c_ts]}

    def jam_control(self, step, dic_platoon_info, ls_m_leader_up_asc, ls_m_veh_up,
                          dic_mplatoon_et, dic_vid_groups, ls_r_dep_times, mpc_interval, delta_t):
        disturb = self.loss_rate != 0
        self.dic_mplatoon_et = dic_mplatoon_et
        if disturb:
            return self._jam_control_disturbed(step, dic_platoon_info, ls_m_leader_up_asc, ls_m_veh_up,
                          dic_vid_groups, ls_r_dep_times, mpc_interval, delta_t)
        else:
            return self._jam_control_clean(step, dic_platoon_info, ls_m_leader_up_asc, ls_m_veh_up,
                          dic_vid_groups, ls_r_dep_times, mpc_interval, delta_t)

    def _monitor_vehicle_stop(self, veh_id):
        """
        Monitors the vehicle and records the time when its speed becomes zero.
        :param veh_id: The vehicle ID to monitor
        :return: The time when the vehicle stops (speed = 0), or None if not stopped yet
        """
        # Get the current speed of the vehicle
        # current_speed = self.traci.vehicle.getSpeed(veh_id)
        # current_pos = self.traci.vehicle.getLanePosition(veh_id)
        current_speed = self.data_recorder.dic_speed[veh_id]
        current_pos = self.data_recorder.dic_pos[veh_id]
        # If the speed is zero, record the simulation time
        if (current_speed == 0 and veh_id not in self.stop_times and
                current_pos > 195):
            stop_time = self.traci.simulation.getTime()  # Get the simulation time
            self.stop_times[veh_id] = stop_time  # Record the time the vehicle stopped
            return stop_time  # Return the stop time
        else:
            return None  # If the speed is not zero, return None

    def _get_r_leader_proper(self): # _get_ravhb_acc()
        '''
        ls_r_leader_up (ls_ravhb), vehicle on RAMP_PROPER (curved / straight approach)
        get ls of ramp leaders on RAMP_PROPER (section before acc lane)
        :return:
            ls_r_leader_proper(ls_ravhb_acc)//
            ls_ravhb_acc_r//['ravh1050', 'ravh920', 'ravh680', 'ravh440', 'ravh200', 'ravh70']
        '''
        ls_r_leader_up = self.dic_vid_groups["ls_r_leader_up"]  # ['ravh200', 'ravh70']
        self.ls_r_leader_proper = []
        if self.ml:
            self.ls_r_leader_proper = ls_r_leader_up
            return self.ls_r_leader_proper
        for id in ls_r_leader_up[::-1]:
            # pos = self.traci.vehicle.getLanePosition(id)
            pos = self.data_recorder.dic_pos[id]
            if pos < 203.5:
                self.ls_r_leader_proper.append(id)
        return self.ls_r_leader_proper

    def _get_r_proper(self): # _get_rvb_acc => _get_r_proper(self)
        '''
        get ls of ramp veh on RAMP_PROPER (section before acc lane)
        :return: self.ls_rvb_acc => self.ls_r_proper
        '''
        ls_r_veh_up = self.dic_vid_groups["ls_r_veh_up"]
        self.ls_r_proper = []
        if self.ml:
            self.ls_r_proper = ls_r_veh_up
            return self.ls_r_proper
        for id in ls_r_veh_up[::-1]:
            pos = self.data_recorder.dic_pos[id]
            if pos < 203.5:
                self.ls_r_proper.append(id)
        return self.ls_r_proper

    def _stop_ramp_fleet3(self):
        '''
        stop the ramp fleet closest to the weaving section
        :return:
        '''
        ls_r_leader_proper = self._get_r_leader_proper()
        first_r_leader_proper = ls_r_leader_proper[0] if ls_r_leader_proper else None # first ramp leader (on ramp_proper) before acc space
        # only dis < 203
        if first_r_leader_proper == 'ravh3890':
            pass

        if (self.r_leader_stop is None
                and first_r_leader_proper is not None
                and first_r_leader_proper not in self.ls_skip_stop
                and first_r_leader_proper not in self.first_stop_recorded):
            # stop the first ramp_proper leader and make sure it's not in skip list
            try:
                if first_r_leader_proper == 'ravh3890':
                    pass
                self.traci.vehicle.setStop(first_r_leader_proper, 'inflow_merge', pos=203.5, laneIndex=0)  # duration
                self.first_stop_recorded.append(first_r_leader_proper)
                self.r_leader_stop = first_r_leader_proper # ravh_stop => r_leader_stop
            except:
                pass

        # Monitor when the vehicle fully stops (speed becomes 0)
        stop_time = self._monitor_vehicle_stop(self.r_leader_stop) if self.r_leader_stop is not None else None
        return self.r_leader_stop

    def _check_resume_state4(self, dic_platoon_info):
        '''
        default: False
        notation: sometimes newest stop_r_leader already gone, however the next r_leader still in the process of stop
        between new stop and non-stop there is a gap where latest stop_r_leader out of ramp_proper (control area)
        under this condition, it still under resume_state = True
        :param dic_platoon_info:
        :return: resume_state
        '''

        ls_stopped_r_leader = list(self.stop_times.keys())

        if len(ls_stopped_r_leader) > 0:
            newest_stop_r_leader = ls_stopped_r_leader[-1]
            # 10032024updated: make sure all stop vehicle are not in acc space
            pos = self.data_recorder.dic_pos[newest_stop_r_leader] \
                if newest_stop_r_leader in self.traci.vehicle.getIDList() else None
            lane_id = self.data_recorder.dic_lane[newest_stop_r_leader] \
                if newest_stop_r_leader in self.traci.vehicle.getIDList() else None
            speed = self.data_recorder.dic_speed[newest_stop_r_leader] \
                if newest_stop_r_leader in self.traci.vehicle.getIDList() else None  # velocity_newest_stop_r_leader
            if (speed == 0 and pos < 210 and lane_id == 'inflow_merge_0'
                    and newest_stop_r_leader in dic_platoon_info): # new added 250615
                self.resume_state = False
                self.stop_state = True
            else:
                self.resume_state = True
                self.stop_state = False
        return

    def _get_remaining_t(self, id, type='follower'):
        '''
        get the remaining time to the weaving section
        :param id:
        :param type: 'leader', if its leader, max speed is faster
        :return: re_t (remaining time), dis (remaining dis)
        '''
        veh_info = self.data_recorder.get_vid_states(id)
        dis = veh_info['dis']
        v = veh_info['v']
        v = 0.0000001 if v == 0 else v
        if type == 'leader':
            re_t = self.vcfunc.estimate_travel_time(v, dis)
        else:
            re_t = dis / v
        return re_t, dis

    def _get_remaining_t2(self, id, type='follower'):
        '''
        use prediction model to get t
        get the remaining time to the weaving section
        :param id:
        :param type: 'leader', if its leader, max speed is faster
        :return: re_t (remaining time), dis (remaining dis)
        '''
        c_ts = self.traci.simulation.getTime()
        veh_info = self.data_recorder.get_vid_states(id)
        dis = veh_info['dis']
        v = veh_info['v']
        v = 0.0000001 if v== 0 else v

        if type == 'leader':
            re_t2 = self.vcfunc.estimate_travel_time(v, dis)
        else:
            # use prediction model
            # get its leader id
            leader_id = self.data_recorder.get_hv_leader(id, m=True)
            if leader_id == 'mavh310':
                pass
            if leader_id == None or id == leader_id or leader_id not in self.dic_mplatoon_et:
                re_t2 = dis / v
            else:
                re_t2 = self.dic_mplatoon_et[leader_id][2]-c_ts
        return re_t2, dis

    def _find_timing6(self, m_leader, action_m_leader, interval, rpt):
        """

        :param m_leader: choosing m_leader
        :param interval: biggest time gap between mainline platoons
        :param rpt: ramp platoons passing time
        :param ls_veh_onih: all veh on inflow_highway
        :return:
        """
        # t_merge = 7.5 # 100m acc
        # t_merge = 11 # 200m acc
        # self.length_ih = self.traci.lane.getLength('inflow_highway_0')
        c_ts = self.traci.simulation.getTime()
        ls_veh_onih = self.dic_vid_groups.get('ls_m_veh_up', None)  # ['mhv700', 'mhv690', 'mavh680'] all veh on inflow_highway
        ls_vonih_v = [self.dic_id_speed[id] for id in ls_veh_onih] if ls_veh_onih else None  # velocity of every veh
        self.timing = False
        # S1
        if (self.stop_state
                and len(ls_veh_onih) > 0
                and len(self.first_resume_recorded) == 0  # condition 5
                and min(ls_vonih_v) < 5):  # condition 11
            self.timing = False
            return self.timing

        # S4
        if self.stop_state and len(ls_veh_onih) == 0:
            self.timing = True
            return self.timing

        # S3
        # only one fleet on the inflow_highway, no follower of fleet's last veh
        if self.stop_state and len(ls_veh_onih) > 0 and m_leader is None:
            last_veh_onih = ls_veh_onih[0]  # last veh on the inflow_highway
            ret_pv, dis = self._get_remaining_t2(last_veh_onih)
            if ret_pv <= self.r_leader_acc_dur:  # condition 8
                self.timing = True
                return self.timing

        # S6: another common situation, m_leader need to take action, and m_leader has a leader on inflow_highway
        # if self.stop_state and len(ls_veh_onih) > 0 and m_leader and mavh_acting is True:
        if self.stop_state and len(ls_veh_onih) > 0 and action_m_leader and self.mavh_acting is True:
            pv_info = self.traci.vehicle.getLeader(action_m_leader, self.length_ih)
            if pv_info is not None:  # condition 6
                pv_id = pv_info[0]
                pv_dis = pv_info[1]  # the dis between m_leader and its pv
                pv_lane_id = self.traci.vehicle.getRoadID(pv_id)
                ret_pv, dis = self._get_remaining_t2(pv_id)
                v_a_mavh = self.dic_id_speed[action_m_leader]
                if v_a_mavh == 0:
                    v_a_mavh = 0.01
                thw = pv_dis / v_a_mavh  # m_leader' time headway (THW), speed
                buffer = 1.5
                desire_reaching_time = self.dic_disire_reach_ts[action_m_leader]
                rt_lrp = c_ts + rpt + self.r_leader_acc_dur  # last veh of ramp platoon reaching time

                if pv_lane_id == 'inflow_highway' and ret_pv <= self.r_leader_acc_dur:
                    if desire_reaching_time > rt_lrp:
                        self.timing = True
                    else:
                        self.ls_skip_stop = []
                    return self.timing

        # S2: Most common situation without m_leader action
        if self.stop_state and len(ls_veh_onih) > 0 and m_leader and self.mavh_acting is False:
            pv_info = self.traci.vehicle.getLeader(m_leader, self.length_ih)
            if pv_info is not None:  # condition 6
                pv_id = pv_info[0]
                pv_lane_id = self.traci.vehicle.getRoadID(pv_id)
                ret_pv, dis = self._get_remaining_t2(pv_id)
                buffer = 1.5
                diff = self.r_leader_acc_dur - ret_pv
                if (pv_lane_id == 'inflow_highway'  # codition 7
                        and ret_pv <= self.r_leader_acc_dur  # condition 8
                        and interval - diff - buffer > rpt):  # condition 9
                    self.timing = True  # S2
                else:
                    self.ls_skip_stop = []  # updated: 241203
                return self.timing

        # S5: 100624updated, m_leader has no leader on inflow_highway
        if self.stop_state and len(ls_veh_onih) > 0 and m_leader and self.mavh_acting is False:
            pv_info = self.traci.vehicle.getLeader(m_leader, self.length_ih)
            if pv_info is not None:  # condition 6
                pv_id = pv_info[0]
                pv_lane_id = self.traci.vehicle.getRoadID(pv_id)
                buffer = 1.5
                if (pv_lane_id != 'inflow_highway'
                        and interval - buffer > rpt + self.r_leader_acc_dur):  # condition 12
                    self.timing = True  # S5
                    return self.timing
        return self.timing

    def _get_max_interval8(self, ls_m_leader_up_asc, ls_m_veh_up):
        '''
        get the max interval on the mainline
        241203 updated, consider the acc time
        112624 updated, use prediction model
        :param ls_m_leader_up_asc: min => max
        :param dic_mplatoon_et: {m_leader:[platoon_type, ts_head, ts_tail, c_ts]}
            thw: time headway window
        :return:
        '''
        # 7.5 => the time needed for r_leader moving from stop point to the merging section
        c_ts = self.traci.simulation.getTime()
        if len(ls_m_veh_up) == 0:
            m_leader = None
            max_thw = self.length_ih/25 - self.r_leader_acc_dur # 7.5 is the acc time, need to consider in
        elif len(ls_m_leader_up_asc) == 0:
            m_leader = None
            max_thw = 0
        else:
        # case 1: between platoons
        # ls_m_leader_up_asc min => max
        # Dictionary to store the time differences for each head vehicle
            headway_differences = {}
            first_avhid = ls_m_leader_up_asc[0]
            first_veh = ls_m_veh_up[-1]
            for i, head_id in enumerate(ls_m_leader_up_asc):
                if head_id == 'mavh2140':
                    pass
                # If it is the first head vehicle, there is no preceding vehicle
                if head_id not in self.dic_mplatoon_et:
                    headway_differences[head_id] = 0
                elif i == 0 and head_id != first_veh:
                    ts_head_current = self.dic_mplatoon_et[head_id][1]
                    keys = list(self.dic_mplatoon_et.keys())
                    this_index = keys.index(head_id)
                    previous_index = this_index-1
                    previous_avh = keys[previous_index]
                    ts_tail_previous = self.dic_mplatoon_et[previous_avh][2]
                    headway_differences[head_id] = ts_head_current - ts_tail_previous
                elif i > 0:
                    # Get the arrival time of the current head vehicle
                    ts_head_current = self.dic_mplatoon_et[head_id][1]
                    # Get the arrival time of the preceding vehicle's tail
                    if ls_m_leader_up_asc[i - 1] == 'mav4690':
                        pass
                    ts_tail_previous = self.dic_mplatoon_et[ls_m_leader_up_asc[i - 1]][2]
                    # Get the remaining time of the preceding vehicle
                    ts_tail_remaining = ts_tail_previous - c_ts
                    if ts_tail_remaining >= self.r_leader_acc_dur:
                        # Calculate the time difference
                        headway_differences[head_id] = ts_head_current - ts_tail_previous
                    else:
                        headway_differences[head_id] = ts_head_current - ts_tail_previous - (self.r_leader_acc_dur - ts_tail_remaining)

            # case 2: between the first platoon and the weaving section
            headway_diff_1 = headway_differences.copy()
            if first_avhid == first_veh:
                pos = self.data_recorder.get_vid_states(first_avhid)['pos']
                first_veh_info = self.dic_mplatoon_et.get(first_avhid, [None, None]) # first vehicle arrival time
                arrive_time = first_veh_info[1]
                # Calculate real headway using a ternary expression
                real_headway = (self.length_ih - pos) / 24.5 - self.r_leader_acc_dur if arrive_time is None \
                    else arrive_time - c_ts - self.r_leader_acc_dur
                headway_diff_1[first_avhid] = real_headway
            # get the max thw of case 1 and 2
            m_leader, max_thw = max(headway_diff_1.items(), key=lambda x: x[1])

        # case 3: between the last platoon and the start point inflow_highway
        if ls_m_veh_up:
            last_mvb = ls_m_veh_up[0]
            veh_info = self.data_recorder.get_vid_states(last_mvb)
            pos = veh_info['pos']  # how far from the start point
            thw = pos / 24.5
            if thw > max_thw:
                m_leader = None
                max_thw = thw
        if m_leader == 'mavh3170':
            pass
        # last veh on inflow_highway
        dic_result = {m_leader: [max_thw]}
        return dic_result

    def _get_r_traveltime3(self, dic_platoon_info):
        """
        update: only r_leader before acc section //0928.2024
        5. get ramp fleet travel time, from stop to pass intersection
        :return: {'ravh880': ['AHHHHH', 'rhv930', 15.3], 'ravh680': ['AHH', 'rhv700', 8.8]}
        """
        # ls_ravhb = self.dic_vid_groups['ls_ravhb']
        ls_r_leader_proper = self._get_r_leader_proper()
        dic_ramp_platoon_info = \
            {ramp_avhid: dic_platoon_info[ramp_avhid] for ramp_avhid in ls_r_leader_proper if
             ramp_avhid in dic_platoon_info}
        # ramp_pinfo_simplified => rps, only keep id:[type,tail_id]
        # dic_ramp_platoon_basic = {key: value[:2] for key, value in dic_ramp_platoon_info.items()} # 20240929update: fixed length platoon info
        dic_ramp_platoon_basic = {key: value[0][:2] for key, value in dic_ramp_platoon_info.items()}  # 20240929update: fixed length platoon info
        dic_tt_by_size = {1: 3.75, 2: 6.17, 3: 8.3, 4: 10.6, 5: 12.67, 6: 14.73, 7: 16.91, 8: 19.0,
                 9: 21.02, 10: 23.98, 11: 26.3, 12: 28.99}
        # dic_tt_by_size = { 1: 4.75, 2: 7.17, 3: 9.3, 4: 11.6, 5: 13.67, 6: 15.73, 7: 17.91, 8: 20.0,
        #           9: 22.02, 10: 24.98, 11: 27.3, 12: 29.99}  # update250822
        dic_r_platoon_travel_time = {key: value + [dic_tt_by_size[len(value[0])]] for key, value in dic_ramp_platoon_basic.items()}
        self.dic_r_platoon_travel_time = dic_r_platoon_travel_time
        return dic_r_platoon_travel_time

    def _compare3(self, dic_max_interval, dic_r_platoon_travel_time, min_time_diff=1):
        """
        filter dic_r_platoon_travel_time after using _get_r_traveltime3 //0928.2024
        updated from find_suitable_interval2
        Targets: 2. determing how many ramp fleets can pass at this interval
        :param min_time_diff: minimal time difference
        :param dic_max_interval: {m_leader:[mpv_id, max_dis, max_thw]}
        :param dic_r_platoon_travel_time:
        :return: final_rpt: final ramp passing time (accumulate ramp passing time)
        """

        c_ts = self.traci.simulation.getTime()
        if c_ts == '250':
            pass
        if dic_max_interval and dic_r_platoon_travel_time and self.stop_state:
            # find the biggest interval and corresponding m_leader
            m_leader = list(dic_max_interval.keys())[0]
            if m_leader == 'mavh990':
                pass
            max_interval = dic_max_interval[m_leader][-1]
            rp_n = len(dic_r_platoon_travel_time) # number of ramp platoons
            cum_rpt = 0 # Cumulative time
            ls_pass_rid = [] # all r_leader_id can pass
            for i in range(rp_n): # 3; i=0, 1, 2
                rp_info = list(dic_r_platoon_travel_time.items())[i] # the ramp fleet info
                rp_leader = rp_info[0] # the leader if of ramp platoon
                rp_t = rp_info[1][2] # the ramp fleet passing time

                if rp_leader == 'ravh2350':
                    pass
                # judge if rp_leader is in stop state
                # if self.traci.vehicle.getSpeed(rp_leader) == 0: # use position instead of speed
                speed = self.data_recorder.dic_speed[rp_leader]
                if speed < 0.8:
                    cum_rpt += rp_t # accumulate ramp passing time
                    # if cum_rpt < max_interval + min_time_diff: # wrong
                    if cum_rpt+ min_time_diff < max_interval :
                        ls_pass_rid.append(rp_leader)
                        if rp_leader not in self.ls_skip_stop and i != 0:
                            if rp_leader == 'ravh990' or rp_leader == 'ravh2610':
                                pass
                            self.ls_skip_stop.append(rp_leader)
                        final_rpt = cum_rpt # final ramp passing time
                    else:
                        break  # jump out this loop
                else:
                    break

            if len(ls_pass_rid) > 0:
                # print(f'ramp platoons number passed this time:{len(ls_pass_rid)}')
                return m_leader, max_interval, final_rpt
            else:
                return m_leader, max_interval, cum_rpt
        else:
            return None, None, None

    def _restart_ramp_fleet(self, r_leader_f, timing):
        """
        250613 update, avoid r_leader stopped before stop point (203.5)
        :param dic_mavhb_hinfo:
               m_leader:
               r_leader_f: first ramp platoon leader (head)
        :return:
        """
        c_ts = self.traci.simulation.getTime()
        pos = self.data_recorder.dic_pos[r_leader_f] if r_leader_f else 0
        if (timing
                and r_leader_f not in self.first_resume_recorded
                and self.stop_state
                and not self.resume_state
                and r_leader_f
                and pos >= 203): # new
            # make sure stop_state=True, resume_state=False
            # (finished resume, whole fleet finished pass weaving section)
            if r_leader_f == 'ravh1470':
                pass
            v_r_leader_f = self.data_recorder.dic_speed[r_leader_f]
            if v_r_leader_f > 0:
                pass
            else:
                self.traci.vehicle.resume(r_leader_f)
                self.first_resume_recorded.append(r_leader_f)
                self.r_leader_stop = None
                self.resume_times[r_leader_f] = c_ts

    def _get_m_leader_action(self, step, r_leader_f, rp_pass_dur, m_leader, max_interval,
                            interval, delta_t, buffer=3):
        """
        _get_mavh_action => _get_m_leader_action
        Decide whether a MAVH (mainline leader) should take action to match the desired merging time.
    
        Args:
            step: current simulation step
            r_leader_f: ramp AV front vehicle ID
            rp_pass_dur: ramp platoon passing duration
            m_leader: candidate mainline vehicle ID
            max_interval: max time gap between ramp platoon and MAVH
            dic_mplatoon_et: estimated arrival time dict for platoon
            delta_t: allowable timing error
            buffer: safety buffer after platoon
            interval: frequency of evaluation

            ts: timestamp
            dur: duration (time period)
    
        Returns:
            self.dic_mavh_actionP: dict of MAVH (m_leader) and its action parameters
            => self.dic_m_leader_action_params
        """
        allowable_error = delta_t  # 0, 2, 4, 6, 8, 10
        last_stop_ts = list(self.stop_times.items())[-1][-1] if self.stop_times else None

        if (step % interval == 0
                or (last_stop_ts is not None and step == last_stop_ts * 10)):  # *10 because sim_step=0.1
            if not (step % interval == 0 or (last_stop_ts is not None and step == last_stop_ts * 10)):
                return self.dic_m_leader_action_params

        c_ts = self.traci.simulation.getTime()
        if not m_leader or m_leader in self.mavh_action_dic:
            return self.dic_m_leader_action_params

        pv_info = self.traci.vehicle.getLeader(m_leader)
        pv_id, pv_dis = pv_info if pv_info else (None, None)
        if not pv_id:
            return self.dic_m_leader_action_params

        pv_lane = self.data_recorder.dic_lane[pv_id]
        if not self.stop_state or pv_lane != 'inflow_highway_0':
            return self.dic_m_leader_action_params

        dic_vid_groups = self.data_recorder.record_vehinfo()
        ls_m_veh_up = dic_vid_groups.get('ls_m_veh_up', [])
        has_zero_speed = any(self.data_recorder.dic_speed[veh_id] == 0 for veh_id in ls_m_veh_up)

        r_leader_waiting_dur = c_ts - self.stop_times[r_leader_f]
        diff = rp_pass_dur - max_interval
        dic_m_leader_info = self.data_recorder.get_vid_states(m_leader)
        m_dis = dic_m_leader_info['dis']  # what's the difference between m_dis and mavh_dis
        m_v0 = dic_m_leader_info['v']

        pv_rem_dur, _ = self._get_remaining_t2(pv_id)  # remaining time of preceding vehicle
        pv_reach_ts = c_ts + pv_rem_dur  # reaching time of preceding vehicle

        r_leader_pv_differ = max(0, self.r_leader_acc_dur - pv_rem_dur)
        desired_reach_ts = pv_reach_ts + rp_pass_dur + buffer + r_leader_pv_differ
        self.dic_disire_reach_ts[m_leader] = desired_reach_ts  # dic_drt => dic_desire_reach_ts

        real_interval = max_interval - r_leader_pv_differ  # pv_rem_dur, remaining time of preceding vehicle to weaving section
        real_error = rp_pass_dur - real_interval  # the real difference between rp passing time needed and intervals

        # estimate reaching_time, with current speed
        estimated_reach_ts = pv_reach_ts + max_interval
        mavh_rem_dur = estimated_reach_ts - c_ts

        if (estimated_reach_ts >= desired_reach_ts or pv_rem_dur <= 0 or has_zero_speed):
            self.dic_m_leader_action_params = {m_leader: []}
            return self.dic_m_leader_action_params
        action_params = []  # get action parameters/ls_action

        # Case 1: If r_leader has been waiting too long, allow looser error margin to avoid indefinite waiting
        mavh_des_reach_dur = None
        if r_leader_waiting_dur > 30 and real_error < allowable_error + 10:
            # Looser threshold due to long waiting time
            mavh_des_reach_dur = mavh_rem_dur + allowable_error + buffer
        # Case 2: Otherwise, allow only if within strict allowable error
        elif real_error < allowable_error:
            # Strict error control
            mavh_des_reach_dur = mavh_rem_dur + real_error + buffer
        if mavh_des_reach_dur is not None:
            action_params = list(self.vcfunc.get_action_params(mavh_des_reach_dur, m_dis, m_v0))
            action_params.append(c_ts)
            self.mavh_action_dic[m_leader] = action_params

        self.dic_m_leader_action_params = {m_leader: action_params}
        return self.dic_m_leader_action_params

    def _apply_m_leader_control(self, step, dic_m_leader_action_params):
        '''
        Apply m_leader action params
        :param dic_m_leader_action_params: action params,
            format => {'mavh1630': [14.3, -1.1, 14.6, 1.1, 24.9, 168.1]}
        :return: the list of action_m_leader
        '''
        # apply action
        ls_m_leader_up = self.dic_vid_groups.get('ls_m_leader_up', None)
        action_m_leader = next(iter(dic_m_leader_action_params or {}), None)
        self.mavh_acting = False # should be mavh_acting
        if (action_m_leader in self.mavh_action_dic
                and action_m_leader in ls_m_leader_up):
            # apply action
            self.vcfunc.apply_leader_action(dic_m_leader_action_params)
            # flash
            self.vcfunc.flashing_merging(step, [action_m_leader])
            self.mavh_acting = True
        return action_m_leader

    def _jam_control_clean(self, step, dic_platoon_info, ls_m_leader_up_asc, ls_m_veh_up,
                           dic_vid_groups, ls_r_dep_times, mpc_interval, delta_t):
        '''
        update 25.1.30
        :param step:
        :param dic_platoon_info: {'ravh40': [['AHHHHHHHHH', 'rhv130'], deque([327.5, 328.6], maxlen=10)],
                                    'mavh40': [['AHHHHHHHHH', 'mhv130'], deque([274.9, 275.5], maxlen=10)]}
        :param ls_m_leader_up_asc: ['mavh1920']
        :param ls_m_veh_up: ['mhv2000', 'mhv1990', 'mhv1980', 'mhv1970', 'mhv1960', 'mhv1950',
                            'mhv1940', 'mhv1930', 'mavh1920', 'mhv1670', 'mhv1660', 'mhv1650',
                            'mhv1640', 'mhv1630', 'mhv1620', 'mhv1610', 'mhv1600', 'mhv1590',
                            'mhv1580', 'mhv1570']
        :param dic_mplatoon_et: {'mavh40': ['AHHHHHHHHH', 44.57387499999983, 64.42628849831135],
                        'mavh310': ['AHHHHHHHHHHH', 63.3397847329901, 87.53000000000002, 63.1]}
        :param dic_vid_groups: all veh info
        :param dic_id_speed: {id1 : [speed1, speed2, ...], id2: [speed1, speed2, ...]}
        :param ls_r_dep_times: [4, 5, 6, ....]
        :return:
        '''
        self.dic_vid_groups = dic_vid_groups
        prc.print_message('**in jam mode**')
        # Stop r_leader (the first one)
        r_leader_f = self._stop_ramp_fleet3()  # first r_leader id
        self._check_resume_state4(dic_platoon_info) # 241003update
        # Get ramp fleet travel time (from stop to pass intersection)
        dic_r_platoon_travel_time = self._get_r_traveltime3(dic_platoon_info)
        dic_max_interval = self._get_max_interval8(ls_m_leader_up_asc, ls_m_veh_up)
        m_leader, max_interval, final_rpt = self._compare3(dic_max_interval, dic_r_platoon_travel_time)
        # m_leader take action; mavh_acting = True/False
        dic_m_leader_action_params = self._get_m_leader_action(step, r_leader_f, final_rpt, m_leader,
                                                   max_interval, mpc_interval, delta_t)
        action_m_leader = self._apply_m_leader_control(step, dic_m_leader_action_params)
        timing = self._find_timing6(m_leader, action_m_leader, max_interval, final_rpt)
        ls_r_proper = self._get_r_proper()
        # record ramp queue length
        queue_log = self.data_recorder.get_queue_length(step, ls_r_proper, ls_r_dep_times)
        # Resume r_leader
        self._restart_ramp_fleet(r_leader_f, timing)
        return queue_log

    def _push_if_not_redundant(self, step, value, update_queue, last_value_attr: str):
        """
        Push a value into the specified delay update_queue if it's not redundant.

        :param step: current time step
        :param value: payload to be pushed
        :param update_queue: which delay buffer to use (e.g., self.action_buffer or self.timing_buffer)
        :param last_value_attr: name of attribute to store the last pushed value (e.g., 'last_action_payload')
        :return: maybe_released value from the update_queue
        """
        if value == True:
            pass
        last_value = getattr(self, last_value_attr, None)
        # is_redundant = (value == last_value or all(not x for x in value))
        if isinstance(value, (list, tuple)):
            is_redundant = (value == last_value or all(not x for x in value))
        else:
            if step == 493:
                pass
            last_push_step = update_queue.buffer[0][1] if update_queue.buffer else None
            is_redundant = (value is False or step == last_push_step)

        if not is_redundant:
            setattr(self, last_value_attr, value)
            update_queue.push2(step, value)
        return update_queue.maybe_release(step)

    def _jam_control_disturbed(self, step, dic_platoon_info, ls_m_leader_up_asc, ls_m_veh_up,
                     dic_vid_groups, ls_r_dep_times, mpc_interval, delta_t):
        '''
        add v2x disturbance
        update 25.1.30
        :param step:
        :param dic_platoon_info:
        :param ls_m_leader_up_asc: dic_m_leader_action_params
        :param ls_m_veh_up:
        :param dic_vid_groups:
        :param dic_id_speed:
        :param ls_r_dep_times:
        :param mpc_interval:
        :return:
        '''
        self.dic_vid_groups = dic_vid_groups
        prc.print_message('**in jam mode**')
        # Stop r_leader (the first one)
        r_leader_f = self._stop_ramp_fleet3()  # first r_leader id
        self._check_resume_state4(dic_platoon_info) # 241003update
        # Get ramp fleet travel time (from stop to pass intersection)
        dic_r_platoon_travel_time = self._get_r_traveltime3(dic_platoon_info)
        dic_max_interval = self._get_max_interval8(ls_m_leader_up_asc, ls_m_veh_up)
        m_leader, max_interval, final_rpt = self._compare3(dic_max_interval, dic_r_platoon_travel_time)

        # m_leader take action; mavh_acting = True/False
        action_params = self._get_mavh_action(step, r_leader_f, final_rpt, m_leader,
                                                 max_interval, mpc_interval, delta_t) # MPC interval = 7s
        action_pay_load = self._push_if_not_redundant(step, action_params,
                                                     self.action_buffer, 'last_action_payload')
        action_m_leader = self._apply_m_leader_control(step, action_pay_load) # pay_load = dic_m_leader_action_params

        timing = self._find_timing6(m_leader, action_m_leader, max_interval, final_rpt)
        timing_pay_load = self._push_if_not_redundant(step, timing,
                                                     self.timing_buffer, 'last_timing_payload')

        # record ramp queue length
        ls_r_proper = self._get_r_proper()
        queue_log = self.data_recorder.get_queue_length(step, ls_r_proper, ls_r_dep_times)
        # Resume r_leader
        self._restart_ramp_fleet(r_leader_f, timing)
        return queue_log

class ShiftMode:
    def __init__(self, traci, instance_dr, av_p):
        self.regular_mode = True
        self.jam_mode = False
        self.traci = traci
        self.data_recorder = instance_dr
        self.av_p = av_p

    def determine_mode4(self, ls_m_veh_up, ls_r_veh_up, ls_r_leader_up):
        '''
        241122 update: as platoon become longer, 1 lead 7
        params:
            ls_m_veh_up: mainline veh before merging
            ls_r_veh_up: ramp veh before merging
            ls_r_leader_up: list of rav leader (head) before merging

            min_plength:

        :return:
        '''
        rho_jam = 90 # 90 veh/km.lane
        check_length = 100 # the last 100m on mainline
        jam_threshold = rho_jam * check_length / 1000  # → 9 vehicles

        length_ih = self.traci.lane.getLength('inflow_highway_0')
        length_ramp = self.traci.lane.getLength('inflow_merge_0')

        # Count vehicles near the end of each lane
        ls_Mjam_veh = [
            vid for vid in ls_m_veh_up
            if self.data_recorder.dic_pos[vid] >= length_ih - check_length
        ]
        ls_Rjam_veh = [
            vid for vid in ls_r_veh_up
            if self.data_recorder.dic_pos[vid] >= length_ramp - check_length
        ]

        if self.regular_mode and (len(ls_Mjam_veh) >= jam_threshold or len(ls_Rjam_veh) >= jam_threshold):
            # on jam condition
            self.regular_mode = False
            self.jam_mode = True

        if self.jam_mode and len(ls_r_leader_up) < 1 and len(ls_Mjam_veh) < jam_threshold and len(ls_Rjam_veh) < jam_threshold: # new condtion: len(ls_veh_f) < max_jam_vnum
            self.regular_mode = True
            self.jam_mode = False

        return self.regular_mode, self.jam_mode