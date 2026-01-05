import print_control as prc
import joblib # model prediction
import pandas as pd

class Func:
    def __init__(self, traci, instance_dr, instance_vcfunc):
        self.traci = traci
        self.data_recorder = instance_dr
        self.vcfunc = instance_vcfunc
        self.dic_vehinfo = {}
        self.dic_id_speed = {}
        self.first_stop_recorded = [] # ramp av only stop once
        self.first_resume_recorded = []  # ramp av only resume once
        self.timing = False
        self.dic_mavhb_hinfo = None
        self.dic_tn = None
        self.temp_list = []
        self.state_jam = False # default state is non-jam
        self.stop_state = False
        self.resume_state = False
        self.stop_times = {}
        self.resume_times = {}
        self.ravh_stop = None # current stop ravh id; only one stop at the front is enough
        self.ls_skip_stop = [] # those ravh can pass with it's preceding ravh platoon doesn't need to stop
        # self.state_handling = True # if jam handling completed
        # for mavh_action2
        self.mavh_action_dic = {} # the action dic of mavh
        self.action_mavh = None
        self.cre_rps = None # coresponding ramp passing time
        self.dic_drt = {} # drt => desire reaching time
        self.dic_mavh_actionP = {'':[]} # mavh and its action parameters (P)
        self.ls_ravhb_acc = [] # list of ramp veh before acc
        self.ls_rvb_acc = [] # list of ramp veh before acc
        # prediction model
        self.mp_model2 = joblib.load(
            "/home/zzha/PycharmProjects/RampMerging3/Models/mr_arrival_prediction_model241122.pkl")


    def check_jam(self, ls_mavhb_sorted):
        # judge the state
        if len(ls_mavhb_sorted) >= 4:
            self.state_jam = True
        else:
            self.state_jam = False  # False
        return self.state_jam

    def stop_ramp_fleet(self):
        """
        3. Stop ravh (the first one)
        :return:
        """
        dic_vehinfo = self.data_recorder.record_vehinfo()
        ls_ravhb = dic_vehinfo["ls_ravhb"]
        ls_ravhb_r = ls_ravhb[::-1] # order reverse
        first_ravh = ls_ravhb[-1]
        if first_ravh not in self.first_stop_recorded:
            # self.traci.vehicle.setStop(first_ravh, 'inflow_merge', pos=283.5, laneIndex=0)  # duration
            # update0924, stop 80m before weaving section
            if first_ravh == 'ravh5820': # only for test
                print("****")
            if first_ravh == 'ravh1050': # only for test
                print("****")

            self.traci.vehicle.setStop(first_ravh, 'inflow_merge', pos=203.5, laneIndex=0)  # duration

            self.first_stop_recorded.append(first_ravh)
            return first_ravh
        # return self.first_stop_recorded[-1]
        return first_ravh

    def monitor_vehicle_stop(self, veh_id):
        """
        Monitors the vehicle and records the time when its speed becomes zero.
        :param veh_id: The vehicle ID to monitor
        :return: The time when the vehicle stops (speed = 0), or None if not stopped yet
        """
        # Get the current speed of the vehicle
        current_speed = self.traci.vehicle.getSpeed(veh_id)
        current_pos = self.traci.vehicle.getLanePosition(veh_id)
        if veh_id == 'ravh2340':
            pass
        # If the speed is zero, record the simulation time
        if (current_speed == 0 and veh_id not in self.stop_times and
                current_pos > 195):
            if veh_id == 'ravh2340':
                pass
            stop_time = self.traci.simulation.getTime()  # Get the simulation time
            self.stop_times[veh_id] = stop_time  # Record the time the vehicle stopped
            return stop_time  # Return the stop time
        else:
            return None  # If the speed is not zero, return None

    def get_ravhb_acc(self):
        '''
        get ls of ravh before acc space
        :return:
            ls_ravhb_acc//
            ls_ravhb_acc_r//['ravh1050', 'ravh920', 'ravh680', 'ravh440', 'ravh200', 'ravh70']
        '''
        ls_ravhb = self.dic_vehinfo["ls_ravhb"]  # ['ravh200', 'ravh70']
        self.ls_ravhb_acc = []
        for id in ls_ravhb[::-1]:
            pos = self.traci.vehicle.getLanePosition(id)
            if pos < 203.5:
                self.ls_ravhb_acc.append(id)
        return self.ls_ravhb_acc

    def get_rvb_acc(self):
        '''
        get ls of rv before acc space
        :return:
            ls_ravhb_acc//
            ls_ravhb_acc_r//['ravh1050', 'ravh920', 'ravh680', 'ravh440', 'ravh200', 'ravh70']
        '''
        ls_rvb = self.dic_vehinfo["ls_rvb"]
        self.ls_rvb_acc = []
        for id in ls_rvb[::-1]:
            pos = self.traci.vehicle.getLanePosition(id)
            if pos < 203.5:
                self.ls_rvb_acc.append(id)
        return self.ls_rvb_acc

    def stop_ramp_fleet3(self):
        '''
        stop the first fleet
        :return:
        '''
        ls_ravhb_acc = self.get_ravhb_acc()
        first_ravhb_acc = ls_ravhb_acc[0] if ls_ravhb_acc else None# first ravh before acc space
        # only dis < 203
        if first_ravhb_acc == 'ravh2890':
            pass

        if (self.ravh_stop is None
                and first_ravhb_acc is not None
                and first_ravhb_acc not in self.ls_skip_stop
                and first_ravhb_acc not in self.first_stop_recorded):
            # stop the first ravhb_acc and make sure it's not in skip list
            try:
                if first_ravhb_acc == 'ravh2340':
                    pass
                self.traci.vehicle.setStop(first_ravhb_acc, 'inflow_merge', pos=203.5, laneIndex=0)  # duration
                self.first_stop_recorded.append(first_ravhb_acc)
                self.ravh_stop = first_ravhb_acc
            except:
                pass

        # Monitor when the vehicle fully stops (speed becomes 0)
        stop_time = self.monitor_vehicle_stop(self.ravh_stop) if self.ravh_stop is not None else None
        return self.ravh_stop


    def check_resume_state4(self):
        '''
        default: False
        notation: sometimes newest_stop_ravh already gone, however the next ravh still in the process of stop
        between new stop and non-stop there is a gap where last stop id out of control area
        under this condition, it still under resume_state = True
        :return: resume_state
        '''

        ls_stopped_ravh = list(self.stop_times.keys())

        if len(ls_stopped_ravh) > 0:
            newest_stop_ravh = ls_stopped_ravh[-1]

            # 10032024updated: make sure all stop vehicle are not in acc space
            pos = self.traci.vehicle.getLanePosition(newest_stop_ravh) \
                if newest_stop_ravh in self.traci.vehicle.getIDList() else None
            lane_id = self.traci.vehicle.getLaneID(newest_stop_ravh) \
                if newest_stop_ravh in self.traci.vehicle.getIDList() else None
            v_nsr = self.traci.vehicle.getSpeed(newest_stop_ravh) \
                if newest_stop_ravh in self.traci.vehicle.getIDList() else None  # velocity_newest_stop_ravh
            if v_nsr == 0 and pos < 210 and lane_id == 'inflow_merge_0':
                self.resume_state = False
                self.stop_state = True
            else:
                self.resume_state = True
                self.stop_state = False
        return self.resume_state


    def find_timing(self, id):
        """
        Judge if it's time to run following codes,
        run following codes at the first second that ravh stopped
        1. the first ravh stopped
        2. there is a veh (preceding veh) in front of mavh at the end of scripts (before merging)

        :param id: mavh_id
        :return: timing, True or False

        Notation!!! sometimes there is a very big gap no following mainline AV appears
        but can't be detected, as no pv_id
        # time:10.5
        """
        if id is not None:
            pv_info = self.traci.vehicle.getLeader(id, 600) # preceeding veh
            if pv_info is not None:
                pv_id = pv_info[0]
            else:
                pv_id = None
            lane_pv = self.traci.vehicle.getLaneID(pv_id)
            if lane_pv != 'inflow_highway_0':
                # case1: id(mavh) is not None, and leader of id is not on 'inflow_highway_0'
                self.timing = True
                return self.timing
        # avoid there is no following MAV on inflow_highway
        elif len(self.traci.edge.getLastStepVehicleIDs('inflow_highway'))==0\
                and len(self.traci.edge.getLastStepVehicleIDs('center'))>0:
            # case2: id(mavh) is None, no vehicles on inflow_highway
            self.timing = True
        else:
            self.timing = False
        return self.timing

    def get_remaining_t(self, id, type='follower'):
        '''
        get the remaining time to the weaving section
        :param id:
        :param type: 'leader', if its leader, max speed is faster
        :return: re_t (remaining time), dis (remaining dis)
        '''
        veh_info = self.vcfunc.get_veh_info(id)
        dis = veh_info['dis']
        v = veh_info['v']
        v = 0.0000001 if v== 0 else v
        if type == 'leader':
            re_t = self.vcfunc.get_ts_a(v, dis)
        else:
            re_t = dis / v
        #
        return re_t, dis

    def get_remaining_t2(self, id, dic_mplatoon_et, type='follower'):
        '''
        use prediction model to get t
        get the remaining time to the weaving section
        :param id:
        :param type: 'leader', if its leader, max speed is faster
        :return: re_t (remaining time), dis (remaining dis)
        '''
        c_ts = self.traci.simulation.getTime()
        veh_info = self.vcfunc.get_veh_info(id)
        dis = veh_info['dis']
        v = veh_info['v']
        v = 0.0000001 if v== 0 else v
        ls_mavhb_sorted = self.dic_vehinfo['ls_mavhb_sorted']

        if type == 'leader':
            re_t = self.vcfunc.get_ts_a(v, dis)
        else:
            # origal method
            re_t = dis / v

            # use prediction model
            # get its leader id
            leader_id = self.data_recorder.get_hv_leader(id, m=True)
            re_t2 = dic_mplatoon_et[leader_id][2]-c_ts
        return re_t2, dis

    def find_timing6(self, mavh, action_mavh, interval, rpt, mavh_in_action = False, this_round = False):
        """

        :param mavh: choosing mavh
        :param interval: biggest time gap between mainline platoons
        :param rpt: ramp platoons passing time
        :param ls_veh_onih: all veh on inflow_highway
        :return:
        """
        c_ts = self.traci.simulation.getTime()
        ls_veh_onih = self.dic_vehinfo.get('ls_mvb', None) # ['mhv700', 'mhv690', 'mavh680'] all veh on inflow_highway
        ls_vonih_v = [self.dic_id_speed[id][-1] for id in ls_veh_onih] if ls_veh_onih else None # velocity of every veh
        self.timing = False
        if mavh == 'mavh3170':
            pass
        # S1
        if (self.stop_state and len(ls_veh_onih) > 0
                and len(self.first_resume_recorded) == 0 # condition 5
                and min(ls_vonih_v) < 5): # condition 11
            self.timing = False
            return self.timing

        # S4
        if self.stop_state and len(ls_veh_onih) == 0:
            self.timing = True
            return self.timing

        # S3
        # only one fleet on the inflow_highway, no follower of fleet's last veh
        if self.stop_state and len(ls_veh_onih) > 0 and mavh is None:
            last_veh_onih = ls_veh_onih[0]  # last veh on the inflow_highway
            ret_pv, dis = self.get_remaining_t(last_veh_onih)
            if ret_pv <= 7.5: # condition 8
                self.timing = True
                return self.timing

        # S6: another common situation, mavh need to take action, and mavh has a leader on inflow_highway
        # if self.stop_state and len(ls_veh_onih) > 0 and mavh and mavh_in_action is True and this_round:
        if self.stop_state and len(ls_veh_onih) > 0 and action_mavh and mavh_in_action is True:
            pv_info = self.traci.vehicle.getLeader(action_mavh, 600)
            if pv_info is not None: # condition 6
                pv_id = pv_info[0]
                pv_dis = pv_info[1] # the dis between mavh and its pv
                pv_lane_id = self.traci.vehicle.getRoadID(pv_id)
                ret_pv, dis = self.get_remaining_t(pv_id)
                v_a_mavh = self.dic_id_speed[action_mavh][-1]
                if v_a_mavh == 0:
                    v_a_mavh = 0.01
                thw = pv_dis / v_a_mavh  # mavh' time headway (THW), speed
                buffer = 1.5
                desire_reaching_time = self.dic_drt[action_mavh]
                rt_lrp = c_ts + rpt + 7.5 # last veh of ramp platoon reaching time
                # if (pv_lane_id == 'inflow_highway'  #  condition 7
                #         and ret_pv <= 7.5 # condition 8
                #         and desire_reaching_time > rt_lrp # condition 10 (updated)
                #         ):
                #     self.timing = True # S6

                if pv_lane_id == 'inflow_highway' and ret_pv <=7.5:
                    if desire_reaching_time > rt_lrp:
                        self.timing = True
                    else:
                        self.ls_skip_stop = []
                    return self.timing

        # S2: Most common situation without mavh action
        if self.stop_state and len(ls_veh_onih) > 0 and mavh and mavh_in_action is False:
            pv_info = self.traci.vehicle.getLeader(mavh, 600)
            if pv_info is not None:  # condition 6
                pv_id = pv_info[0]
                pv_lane_id = self.traci.vehicle.getRoadID(pv_id)
                ret_pv, dis = self.get_remaining_t(pv_id)
                buffer = 1.5
                diff = 7.5 - ret_pv
                if (pv_lane_id == 'inflow_highway'  # codition 7
                        and ret_pv <= 7.5 # condition 8
                        and interval - diff - buffer > rpt): # condition 9
                    self.timing = True  # S2
                else:
                    self.ls_skip_stop = [] # updated: 241203
                return self.timing

        # S5: 100624updated, mavh has no leader on inflow_highway
        if self.stop_state and len(ls_veh_onih) > 0 and mavh and mavh_in_action is False:
            pv_info = self.traci.vehicle.getLeader(mavh, 600)
            if pv_info is not None:  # condition 6
                pv_id = pv_info[0]
                pv_lane_id = self.traci.vehicle.getRoadID(pv_id)
                buffer = 1.5
                if (pv_lane_id != 'inflow_highway'
                        and interval - buffer > rpt + 7.5):  # condition 12
                    self.timing = True  # S5
                    return self.timing
        return self.timing

    def find_timing_old(self, mavh, interval, rpt, mavh_in_action = False, this_round = False):
        """

        :param mavh: choosing mavh
        :param interval: biggest time gap between mainline platoons
        :param rpt: ramp platoons passing time
        :param ls_veh_onih: all veh on inflow_highway
        :return:
        """
        c_ts = self.traci.simulation.getTime()
        ls_veh_onih = self.dic_vehinfo.get('ls_mvb', None) # ['mhv700', 'mhv690', 'mavh680'] all veh on inflow_highway
        ls_vonih_v = [self.dic_id_speed[id][-1] for id in ls_veh_onih] if ls_veh_onih else None # velocity of every veh
        self.timing = False
        if mavh == 'mavh2030':
            pass
        # S1
        if (self.stop_state and len(ls_veh_onih) > 0
                and len(self.first_resume_recorded) == 0 # condition 5
                and min(ls_vonih_v) < 5): # condition 11
            self.timing = False
            return self.timing

        # S4
        if self.stop_state and len(ls_veh_onih) == 0:
            self.timing = True
            return self.timing

        # S3
        # only one fleet on the inflow_highway, no follower of fleet's last veh
        if self.stop_state and len(ls_veh_onih) > 0 and mavh is None:
            last_veh_onih = ls_veh_onih[0]  # last veh on the inflow_highway
            ret_pv, dis = self.get_remaining_t(last_veh_onih)
            if ret_pv <= 7.5: # condition 8
                self.timing = True
                return self.timing

        # S6: another common situation, mavh need to take action, and mavh has a leader on inflow_highway
        if self.stop_state and len(ls_veh_onih) > 0 and mavh and mavh_in_action is True and this_round:
            pv_info = self.traci.vehicle.getLeader(mavh, 600)
            if pv_info is not None: # condition 6
                pv_id = pv_info[0]
                pv_dis = pv_info[1] # the dis between mavh and its pv
                pv_lane_id = self.traci.vehicle.getRoadID(pv_id)
                ret_pv, dis = self.get_remaining_t(pv_id)
                v_mavh = self.dic_id_speed[mavh][-1]
                thw = pv_dis / v_mavh  # mavh' time headway (THW), speed
                buffer = 1.5
                desire_reaching_time = self.dic_drt[mavh]
                rt_lrp = c_ts + rpt + 7.5 # last veh of ramp platoon reaching time
                # if (pv_lane_id == 'inflow_highway'  #  condition 7
                #         and ret_pv <= 7.5 # condition 8
                #         and desire_reaching_time > rt_lrp # condition 10 (updated)
                #         ):
                #     self.timing = True # S6

                if pv_lane_id == 'inflow_highway' and ret_pv <=7.5:
                    if desire_reaching_time > rt_lrp:
                        self.timing = True
                    else:
                        self.ls_skip_stop = []
                    return self.timing

        # S2: Most common situation without mavh action
        if self.stop_state and len(ls_veh_onih) > 0 and mavh and mavh_in_action is False:
            pv_info = self.traci.vehicle.getLeader(mavh, 600)
            if pv_info is not None:  # condition 6
                pv_id = pv_info[0]
                pv_lane_id = self.traci.vehicle.getRoadID(pv_id)
                ret_pv, dis = self.get_remaining_t(pv_id)
                buffer = 1.5
                diff = 7.5 - ret_pv
                if (pv_lane_id == 'inflow_highway'  # codition 7
                        and ret_pv <= 7.5 # condition 8
                        and interval - diff - buffer > rpt): # condition 9
                    self.timing = True  # S2
                else:
                    self.ls_skip_stop = [] # updated: 241203
                return self.timing

        # S5: 100624updated, mavh has no leader on inflow_highway
        if self.stop_state and len(ls_veh_onih) > 0 and mavh and mavh_in_action is False:
            pv_info = self.traci.vehicle.getLeader(mavh, 600)
            if pv_info is not None:  # condition 6
                pv_id = pv_info[0]
                pv_lane_id = self.traci.vehicle.getRoadID(pv_id)
                buffer = 1.5
                if (pv_lane_id != 'inflow_highway'
                        and interval - buffer > rpt + 7.5):  # condition 12
                    self.timing = True  # S5
                    return self.timing
        return self.timing

    def get_max_interval8(self, ls_mavhb_sorted, ls_mvb, dic_mplatoon_et):
        '''
        241203 updated, consider the acc time
        112624 updated, use prediction model
        :param ls_mavhb_sorted: min => max
        :param dic_mplatoon_et: {id:[platoon_type, ts_head, ts_tail, c_ts]}
        :return:
        '''

        c_ts = self.traci.simulation.getTime()
        if len(ls_mvb) == 0:
            mavh_id = None
            max_thw = 600/25 - 7.5 # 7.5 is the acc time, need to consider in
        elif len(ls_mavhb_sorted) == 0:
            mavh_id = None
            max_thw = 0
        else:
        # case 1: between platoons
        # ls_mavhb_sorted min => max
        # Dictionary to store the time differences for each head vehicle
            headway_differences = {}
            first_avhid = ls_mavhb_sorted[0]
            first_veh = ls_mvb[-1]
            for i, head_id in enumerate(ls_mavhb_sorted):
                if head_id == 'mavh2140':
                    pass
                # If it is the first head vehicle, there is no preceding vehicle
                if head_id not in dic_mplatoon_et:
                    headway_differences[head_id] = 0
                elif i == 0 and head_id != first_veh:
                    ts_head_current = dic_mplatoon_et[head_id][1]
                    keys = list(dic_mplatoon_et.keys())
                    this_index = keys.index(head_id)
                    previous_index = this_index-1
                    previous_avh = keys[previous_index]
                    ts_tail_previous = dic_mplatoon_et[previous_avh][2]
                    headway_differences[head_id] = ts_head_current - ts_tail_previous
                elif i > 0:
                    # Get the arrival time of the current head vehicle
                    ts_head_current = dic_mplatoon_et[head_id][1]
                    # Get the arrival time of the preceding vehicle's tail
                    ts_tail_previous = dic_mplatoon_et[ls_mavhb_sorted[i - 1]][2]
                    # Get the remaining time of the preceding vehicle
                    ts_tail_remaining = ts_tail_previous - c_ts
                    if ts_tail_remaining >= 7.5:
                        # Calculate the time difference
                        headway_differences[head_id] = ts_head_current - ts_tail_previous
                    else:
                        headway_differences[head_id] = ts_head_current - ts_tail_previous - (7.5 - ts_tail_remaining)

            # case 2: between the first platoon and the weaving section
            headway_diff_1 = headway_differences.copy()
            if first_avhid == first_veh:
                pos = self.vcfunc.get_veh_info(first_avhid)['pos']
                first_veh_info = dic_mplatoon_et.get(first_avhid, [None, None]) # first vehicle arrival time
                arrive_time = first_veh_info[1]
                # Calculate real headway using a ternary expression
                real_headway = (600 - pos) / 24.5 - 7.5 if arrive_time is None else arrive_time - c_ts - 7.5
                headway_diff_1[first_avhid] = real_headway
            # get the max thw of case 1 and 2
            mavh_id, max_thw = max(headway_diff_1.items(), key=lambda x: x[1])

        # case 3: between the last platoon and the start point inflow_highway
        if ls_mvb:
            last_mvb = ls_mvb[0]
            veh_info = self.vcfunc.get_veh_info(last_mvb)
            pos = veh_info['pos']  # how far from the start point
            thw = pos / 24.5
            if thw > max_thw:
                mavh_id = None
                max_thw = thw
        if mavh_id == 'mavh3170':
            pass
        # last veh on inflow_highway
        dic_result = {mavh_id: [max_thw]}
        return dic_result

    def get_max_interval7(self, ls_mavhb_sorted, ls_mvb, dic_mplatoon_et):
        '''
        112624 updated, use prediction model
        :param ls_mavhb_sorted: min => max
        :param dic_mplatoon_et: {id:[platoon_type, ts_head, ts_tail, c_ts]}
        :return:
        '''

        c_ts = self.traci.simulation.getTime()
        if len(ls_mvb) == 0:
            mavh_id = None
            max_thw = 600/25
        elif len(ls_mavhb_sorted) == 0:
            mavh_id = None
            max_thw = 0
        else:
        # case 1: between platoons
        # ls_mavhb_sorted min => max
        # Dictionary to store the time differences for each head vehicle
            headway_differences = {}
            first_avhid = ls_mavhb_sorted[0]
            first_veh = ls_mvb[-1]
            for i, head_id in enumerate(ls_mavhb_sorted):
                # If it is the first head vehicle, there is no preceding vehicle
                if head_id not in dic_mplatoon_et:
                    headway_differences[head_id] = 0
                elif i == 0 and head_id != first_veh:
                    ts_head_current = dic_mplatoon_et[head_id][1]
                    keys = list(dic_mplatoon_et.keys())
                    this_index = keys.index(head_id)
                    previous_index = this_index-1
                    previous_avh = keys[previous_index]
                    ts_tail_previous = dic_mplatoon_et[previous_avh][2]
                    headway_differences[head_id] = ts_head_current - ts_tail_previous
                elif i > 0:
                    # Get the arrival time of the current head vehicle
                    ts_head_current = dic_mplatoon_et[head_id][1]
                    # Get the arrival time of the preceding vehicle's tail
                    ts_tail_previous = dic_mplatoon_et[ls_mavhb_sorted[i - 1]][2]
                    # Calculate the time difference
                    headway_differences[head_id] = ts_head_current - ts_tail_previous

            # case 2: between the first platoon and the weaving section
            headway_diff_1 = headway_differences.copy()
            if first_avhid == first_veh:
                pos = self.vcfunc.get_veh_info(first_avhid)['pos']
                first_veh_info = dic_mplatoon_et.get(first_avhid, [None, None]) # first vehicle arrival time
                arrive_time = first_veh_info[1]
                # Calculate real headway using a ternary expression
                real_headway = (600 - pos) / 24.5 if arrive_time is None else arrive_time - c_ts
                headway_diff_1[first_avhid] = real_headway
            # get the max thw of case 1 and 2
            mavh_id, max_thw = max(headway_diff_1.items(), key=lambda x: x[1])

        # case 3: between the last platoon and the start point inflow_highway
        if ls_mvb:
            last_mvb = ls_mvb[0]
            veh_info = self.vcfunc.get_veh_info(last_mvb)
            pos = veh_info['pos'] # how far from the start point
            thw = pos/24.5
            if thw > max_thw:
                mavh_id = None
                max_thw = thw

        # last veh on inflow_highway
        dic_result = {mavh_id:[max_thw]}
        return dic_result

    def find_max_info(self, dic_mavhb_hinfo_new, ls_mavhb_sorted):
        '''
        find the max gaps on the inflow_highway, including related information
        :param dic_mavhb_hinfo_new:
        :return:
        '''
        values = dic_mavhb_hinfo_new.values()  # [p_id, dis, time_headway]
        ls_thw = [item[-1] for item in values]
        if len(ls_thw) == 0:
            max_dis = 600
            max_thw = max_dis/24
            mavh_id = None
            mpv_id = None
        else:
            max_thw = max(ls_thw)
            max_index = ls_thw.index(max_thw)
            mavh_id = ls_mavhb_sorted[max_index]
            mpv_id = dic_mavhb_hinfo_new[mavh_id][0]
            max_dis = dic_mavhb_hinfo_new[mavh_id][1]
        return mavh_id, max_dis, max_thw, mpv_id

    def get_r_traveltime3(self, dic_platoon_info):
        """
        update: only ravh before acc space //0928.2024
        5. get ramp fleet travel time, from stop to pass intersection
        :return: {'ravh880': ['AHHHHH', 'rhv930', 15.3], 'ravh680': ['AHH', 'rhv700', 8.8]}
        """
        # ls_ravhb = self.dic_vehinfo['ls_ravhb']
        ls_ravhb_acc = self.get_ravhb_acc()
        dic_ramp_pinfo = \
            {ramp_avhid: dic_platoon_info[ramp_avhid] for ramp_avhid in ls_ravhb_acc if
             ramp_avhid in dic_platoon_info}
        # ramp_pinfo_simplified => rps, only keep id:[type,tail_id]
        # dic_rps = {key: value[:2] for key, value in dic_ramp_pinfo.items()} # 20240929update: fixed length platoon info
        dic_rps = {key: value[0][:2] for key, value in dic_ramp_pinfo.items()}  # 20240929update: fixed length platoon info
        # dic_t = {1: 3.9, 2: 6.4, 3: 8.8, 4: 11, 5: 13.8, 6: 15.3}  # length : travel_time
        dic_t = {1: 1.25, 2: 3.67, 3: 5.80, 4: 8.10, 5: 10.17, 6: 12.23, 7: 14.41, 8: 16.50,
                 9: 18.52, 10: 21.48, 11: 23.80, 12: 26.49}  # update241201
        # dic_t = {1: 1.25, 2: 3.67, 3: 5.80, 4: 8.10, 5: 10.13, 6: 12.23, 7: 14.21}  # update0924
        # dic_t = {1: 1.25, 2: 3.67, 3: 5.80, 4: 8.10, 5: 10.13, 6: 12.83}  # update0924, with acc faster around 3s
        dic_tn = {key: value + [dic_t[len(value[0])]] for key, value in dic_rps.items()}
        self.dic_tn = dic_tn
        return dic_tn

    def compare3(self, dic_r_max, dic_tn, min_time_diff=1):
        """
        filter dic_tn after using get_r_traveltime3 //0928.2024
        updated from find_suitable_interval2
        Targets: 2. determing how many ramp fleets can pass at this interval
        :param min_time_diff: minimal time differennce
        :param dic_r_max: {mavh_id:[mpv_id, max_dis, max_thw]}
        :param dic_tn:
        :return: final_rpt: final ramp passing time (accumulate ramp passing time)
        """

        c_ts = self.traci.simulation.getTime()
        if dic_r_max and dic_tn and self.stop_state:
            # find the biggest interval and corresponding mavh
            mavh_id = list(dic_r_max.keys())[0]
            if mavh_id == 'mavh990':
                pass
            max_interval = dic_r_max[mavh_id][-1]
            rp_n = len(dic_tn) # number of ramp platoons
            cum_rpt = 0 # Cumulative time
            ls_pass_rid = [] # all ravh_id can pass
            for i in range(rp_n): # 3; i=0, 1, 2
                rp_info = list(dic_tn.items())[i] # the ramp fleet info
                rp_leader = rp_info[0] # the leader if of ramp platoon
                rp_t = rp_info[1][2] # the ramp fleet passing time
                # judge if rp_leader is in stop state
                # if self.traci.vehicle.getSpeed(rp_leader) == 0: # use position instead of speed
                # re_t, dis = self.get_remaining_t(rp_leader)
                speed = self.traci.vehicle.getSpeed(rp_leader)
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
                return mavh_id, max_interval, final_rpt
            else:
                return mavh_id, max_interval, cum_rpt
        else:
            return None, None, None

    def restart_ramp_fleet(self, ravh_f):
        """
        8. restart ravh after mpv passed the intersection
        :param dic_mavhb_hinfo:
        :param mavh_id:
        :return:
        """
        c_ts = self.traci.simulation.getTime()
        if (self.timing
                and ravh_f not in self.first_resume_recorded
                and self.stop_state
                and not self.resume_state
                and ravh_f):
            # make sure stop_state=True, resume_state=False
            # (finished resume, whole fleet finished pass wearving section)
            v_ravh_f = self.traci.vehicle.getSpeed(ravh_f)
            if v_ravh_f > 0:
                pass
            else:
                if ravh_f == 'ravh2890':
                    pass
                self.traci.vehicle.resume(ravh_f)
                self.first_resume_recorded.append(ravh_f)
                self.ravh_stop = None
                self.resume_times[ravh_f] = c_ts
                self.timing = False

    def mavh_action2(self, step, ravh_f, r_pass_time, mavh, max_interval, buffer=3):

        '''
        obtain mavh's action parameters and apply action
        input: step, ravh_f, r_pass_time, mavh, max_interval, set_waiting_time,
                tolerant_diff, buffer # r_pass_time > 10
        output: desire reaching time ???
        '''
        # allowable error and waiting time
        # 15 : wt<=10; 3 : 10<wt<=30; 5 : 30<wt<=60; 7 : 60<wt
        ls_allowable_error = [4, 7, 10, 13]  # [1.5, 3, 5, 7]
        c_ts = self.traci.simulation.getTime()
        ls_mavhb = self.dic_vehinfo.get('ls_mavhb', None)
        desire_reaching_time = None
        pv_info = self.traci.vehicle.getLeader(mavh) if mavh else None
        mavh_in_action = False # default

        if mavh is not None and mavh not in self.mavh_action_dic and self.stop_state and pv_info is not None:
            if mavh == 'mavh4970':
                pass
            waiting_time = c_ts - self.stop_times[ravh_f]
            diff = r_pass_time - max_interval
            dic_mavh_info = self.vcfunc.get_veh_info(mavh)
            m_dis = dic_mavh_info['dis']
            m_v0 = dic_mavh_info['v']

            pv_id, pv_dis = pv_info
            ret_pv, dis = self.get_remaining_t(pv_id)  # remaining time of preceding vehicle

            reaching_time_pv = c_ts + ret_pv  # reaching time of preceding vehicle
            desire_reaching_time = c_ts + ret_pv + r_pass_time + buffer  # mavh's reaching time to the weaving section
            # because the interval between pv and mavh should much larger, as ramp leader will has a little delay to the pv
            self.dic_drt[mavh] = desire_reaching_time # dic_drt => dic_desire_reaching_time
            # get real error
            # get real interval
            real_interval = max_interval-(7.5-ret_pv) # ret_pv, remaining time of preceding vehicle to weaving section
            real_error = r_pass_time - real_interval

            # estimate reaching_time, with current speed
            # if ert<drt, no need to take action
            ret_mavh, dis_mavh = self.get_remaining_t(mavh)
            estimate_reaching_time = c_ts + ret_mavh
            if estimate_reaching_time >= desire_reaching_time:
                mavh_in_action = False
                return mavh_in_action

            # else, mavh takes action. And get action parameters
            if waiting_time <= 10:
                allowable_error = ls_allowable_error[0]
                if real_error < allowable_error:
                    # get desired reaching time and apply action and flashing
                    # get acc parameter
                    # ls_action = (t1, a1, t3, a3, v_rem)
                    ls_action = list(self.vcfunc.get_action_params(desire_reaching_time - c_ts, m_dis, m_v0))
                    ls_action2 = ls_action.append(c_ts)
                    self.mavh_action_dic[mavh] = ls_action
                    self.action_mavh = mavh
                    self.cre_rps = r_pass_time
            elif 10 < waiting_time <= 30:
                allowable_error = ls_allowable_error[1]
                if real_error < allowable_error:
                    # get desired reaching time and apply action and flashing
                    # get acc parameter
                    # ls_action = (t1, a1, t3, a3, v_rem)
                    ls_action = list(self.vcfunc.get_action_params(desire_reaching_time - c_ts, m_dis, m_v0))
                    ls_action2 = ls_action.append(c_ts)
                    self.mavh_action_dic[mavh] = ls_action
                    self.action_mavh = mavh
                    self.cre_rps = r_pass_time
                    # dic = {mavh: ls_action}
            elif 30 < waiting_time <= 60:
                allowable_error = ls_allowable_error[2]
                # if r_pass_time - max_interval < allowable_error:
                if real_error < allowable_error:
                    # ls_action = (t1, a1, t3, a3, v_rem)
                    ls_action = list(self.vcfunc.get_action_params(desire_reaching_time - c_ts, m_dis, m_v0))
                    ls_action2 = ls_action.append(c_ts)
                    self.mavh_action_dic[mavh] = ls_action
                    self.action_mavh = mavh
                    self.cre_rps = r_pass_time
                    # dic = {mavh: ls_action}
            elif waiting_time > 60:
                allowable_error = ls_allowable_error[3]
                # if r_pass_time - max_interval < allowable_error:
                if real_error < allowable_error:
                    # ls_action = (t1, a1, t3, a3, v_rem)
                    ls_action = list(self.vcfunc.get_action_params(desire_reaching_time - c_ts, m_dis, m_v0))
                    ls_action2 = ls_action.append(c_ts)
                    self.mavh_action_dic[mavh] = ls_action
                    self.action_mavh = mavh
                    self.cre_rps = r_pass_time
                    # dic = {mavh: ls_action}
            else:
                pass
        # apply action
        if (self.action_mavh in self.mavh_action_dic
                and self.action_mavh in ls_mavhb):
        # if self.action_mavh in self.mavh_action_dic and self.action_mavh in ls_mavhb and mavh==self.action_mavh: # updated 241122
            if self.action_mavh == 'mavh4430':
                pass
            dic = {self.action_mavh: self.mavh_action_dic[self.action_mavh]}
            # apply action
            self.vcfunc.apply_avh_action(dic)
            # flash
            self.vcfunc.flashing2(step, [self.action_mavh])
            mavh_in_action = True
        return mavh_in_action

    def get_mavh_action_sp(self, step, ravh_f, r_pass_time, mavh, max_interval, dic_mplatoon_et, delta_t, buffer=3, interval=70):
        '''
        test single parameter (sp/delta_t_tolerable)
        activate at the set interval and the first time when ravh stopped
        get the mavh' action parameters
        :return: self.dic_mavh_actionP (mavh and its action parameters (P))
        '''
        allowable_error = delta_t  # 0, 2, 4, 6, 8, 10
        last_stop_time = list(self.stop_times.items())[-1][-1] if self.stop_times else None
        if step % interval == 0 or (last_stop_time is not None and step == last_stop_time * 10):
            c_ts = self.traci.simulation.getTime()
            ls_mavhb = self.dic_vehinfo.get('ls_mavhb', None)
            desire_reaching_time = None  # desire arrival time of mavh
            pv_info = self.traci.vehicle.getLeader(mavh) if mavh else None
            pv_id, pv_dis = pv_info if pv_info is not None else (None, None)
            pv_lane = self.traci.vehicle.getLaneID(pv_id) if pv_id else None

            if (mavh is not None and
                    mavh not in self.mavh_action_dic and
                    self.stop_state and
                    pv_lane == 'inflow_highway_0'
            ):

                dic_vehinfo = self.data_recorder.record_vehinfo()
                ls_mvb = dic_vehinfo.get('ls_mvb', [])
                has_zero_speed = any(self.traci.vehicle.getSpeed(veh_id) == 0 for veh_id in ls_mvb)

                waiting_time = c_ts - self.stop_times[ravh_f]
                diff = r_pass_time - max_interval
                dic_mavh_info = self.vcfunc.get_veh_info(mavh)
                m_dis = dic_mavh_info['dis']
                m_v0 = dic_mavh_info['v']

                pv_id, pv_dis = pv_info
                pv_lane = self.traci.vehicle.getLaneID(pv_id)

                # ret_pv, dis = self.get_remaining_t(pv_id)  # remaining time of preceding vehicle
                ret_pv, dis = self.get_remaining_t2(pv_id, dic_mplatoon_et)  # remaining time of preceding vehicle
                reaching_time_pv = c_ts + ret_pv  # reaching time of preceding vehicle
                # 250122 update: because ignore 7.5 before!
                if ret_pv < 7.5:
                    time_compensation = 7.5 - ret_pv
                    desire_reaching_time = reaching_time_pv + r_pass_time + buffer + time_compensation
                else:
                    desire_reaching_time = reaching_time_pv + r_pass_time + buffer  # mavh's reaching time to the weaving section
                # because the interval between pv and mavh should much larger, as ramp leader will has a little delay to the pv
                self.dic_drt[mavh] = desire_reaching_time  # dic_drt => dic_desire_reaching_time
                # get real error
                # get real interval
                if ret_pv < 7.5:
                    real_interval = max_interval - (
                            7.5 - ret_pv)  # ret_pv, remaining time of preceding vehicle to weaving section
                else:
                    real_interval = max_interval
                real_error = r_pass_time - real_interval  # the real difference between rp passing time needed and intervals
                # estimate reaching_time, with current speed
                # if ert<drt, no need to take action
                ret_mavh, dis_mavh = self.get_remaining_t(mavh)
                estimate_reaching_time = c_ts + ret_mavh
                if estimate_reaching_time >= desire_reaching_time:
                    self.dic_mavh_actionP = {mavh: []}
                elif ret_pv <= 0:  # 250122 updated: avoid leading_hv too close to the weaving section
                    self.dic_mavh_actionP = {mavh: []}
                elif has_zero_speed:
                    # 0122update: if there is one mainline vehicle speed = 0, no action
                    self.dic_mavh_actionP = {mavh: []}
                else:
                    # get action parameters/ls_action
                    ls_action = []

                    if real_error < allowable_error:
                        ls_action = list(self.vcfunc.get_action_params(desire_reaching_time - c_ts, m_dis, m_v0))
                        ls_action.append(c_ts)
                        self.mavh_action_dic[mavh] = ls_action
                        self.cre_rps = r_pass_time

                    self.dic_mavh_actionP = {mavh: ls_action}
        return self.dic_mavh_actionP


    def get_mavh_action(self, step, ravh_f, r_pass_time, mavh, max_interval, dic_mplatoon_et, par_group, buffer=3, interval=70):
        '''
        activate at the set interval and the first time when ravh stopped
        get the mavh' action parameters
        :return: self.dic_mavh_actionP (mavh and its action parameters (P))
        '''
        g_ori = [9, 12, 15, 18]
        # interval = 4
        g1_4 = [2, 6, 10, 14]
        g2_4 = [6, 10, 14, 18]
        g3_4 = [10, 14, 18, 22]
        g4_4 = [14, 18, 22, 26]
        # interval = 3
        g1_3 = [5, 8, 11, 14]
        g2_3 = [8, 11, 14, 17]
        g3_3 = [11, 14, 17, 20]
        g4_3 = [14, 17, 20, 23]
        # interval = 2
        g1_2 = [8, 10, 12, 14]
        g2_2 = [10, 12, 14, 16]
        g3_2 = [12, 14, 16, 18]
        g4_2 = [14, 16, 18, 20]

        # interval = 1
        g1_1 = [11, 12, 13, 14]
        g2_1 = [12, 13, 14, 15]
        g3_1 = [13, 14, 15, 16]
        g4_1 = [14, 15, 16, 17]
        # final candidates
        g_f1 = [8, 11, 14, 17] # better, 12.382357
        g_f2 = [8, 10, 12, 14]

        ls_allowable_error = eval(par_group)
        last_stop_time = list(self.stop_times.items())[-1][-1] if self.stop_times else None
        if step % interval == 0 or (last_stop_time is not None and step == last_stop_time*10):
            c_ts = self.traci.simulation.getTime()
            ls_mavhb = self.dic_vehinfo.get('ls_mavhb', None)
            desire_reaching_time = None # desire arrival time of mavh
            pv_info = self.traci.vehicle.getLeader(mavh) if mavh else None
            pv_id, pv_dis = pv_info if pv_info is not None else (None, None)
            pv_lane = self.traci.vehicle.getLaneID(pv_id) if pv_id else None

            if (mavh is not None and
                    mavh not in self.mavh_action_dic and
                    self.stop_state and
                    pv_lane == 'inflow_highway_0'
                    ):
                if mavh == 'mavh3170':
                    pass

                dic_vehinfo = self.data_recorder.record_vehinfo()
                ls_mvb = dic_vehinfo.get('ls_mvb', [])
                has_zero_speed = any(self.traci.vehicle.getSpeed(veh_id) == 0 for veh_id in ls_mvb)

                waiting_time = c_ts - self.stop_times[ravh_f]
                diff = r_pass_time - max_interval
                dic_mavh_info = self.vcfunc.get_veh_info(mavh)
                m_dis = dic_mavh_info['dis']
                m_v0 = dic_mavh_info['v']

                pv_id, pv_dis = pv_info
                pv_lane = self.traci.vehicle.getLaneID(pv_id)

                # ret_pv, dis = self.get_remaining_t(pv_id)  # remaining time of preceding vehicle
                ret_pv, dis = self.get_remaining_t2(pv_id, dic_mplatoon_et)  # remaining time of preceding vehicle
                reaching_time_pv = c_ts + ret_pv  # reaching time of preceding vehicle
                # 250122 update: because ignore 7.5 before!
                if ret_pv < 7.5:
                    time_compensation = 7.5 - ret_pv
                    desire_reaching_time = reaching_time_pv + r_pass_time + buffer + time_compensation
                else:
                    desire_reaching_time = reaching_time_pv + r_pass_time + buffer  # mavh's reaching time to the weaving section
                # because the interval between pv and mavh should much larger, as ramp leader will has a little delay to the pv
                self.dic_drt[mavh] = desire_reaching_time  # dic_drt => dic_desire_reaching_time
                # get real error
                # get real interval
                if ret_pv < 7.5:
                    real_interval = max_interval - (7.5 - ret_pv)  # ret_pv, remaining time of preceding vehicle to weaving section
                else:
                    real_interval = max_interval
                real_error = r_pass_time - real_interval # the real difference between rp passing time needed and intervals
                # estimate reaching_time, with current speed
                # if ert<drt, no need to take action
                ret_mavh, dis_mavh = self.get_remaining_t(mavh)
                estimate_reaching_time = c_ts + ret_mavh
                if estimate_reaching_time >= desire_reaching_time:
                    self.dic_mavh_actionP = {mavh:[]}
                elif ret_pv <= 0: # 250122 updated: avoid leading_hv too close to the weaving section
                    self.dic_mavh_actionP = {mavh: []}
                elif has_zero_speed:
                    # 0122update: if there is one mainline vehicle speed = 0, no action
                    self.dic_mavh_actionP = {mavh: []}
                else:
                    # get action parameters/ls_action
                    ls_action = []
                    if waiting_time <= 10:
                        allowable_error = ls_allowable_error[0] # 1.5, 3
                        if real_error < allowable_error:
                            # get desired reaching time and apply action and flashing
                            # get acc parameter
                            # ls_action = (t1, a1, t3, a3, v_rem)
                            ls_action = list(self.vcfunc.get_action_params(desire_reaching_time - c_ts, m_dis, m_v0))
                            ls_action.append(c_ts)
                            self.mavh_action_dic[mavh] = ls_action
                            self.cre_rps = r_pass_time
                    elif 10 < waiting_time <= 30:
                        allowable_error = ls_allowable_error[1]  # 3, 5
                        if real_error < allowable_error:
                            # get desired reaching time and apply action and flashing
                            # get acc parameter
                            # ls_action = (t1, a1, t3, a3, v_rem)
                            ls_action = list(self.vcfunc.get_action_params(desire_reaching_time - c_ts, m_dis, m_v0))
                            ls_action.append(c_ts)
                            self.mavh_action_dic[mavh] = ls_action
                            self.cre_rps = r_pass_time
                            # dic = {mavh: ls_action}
                    elif 30 < waiting_time <= 60:
                        allowable_error = ls_allowable_error[2]  # 5, 7
                        # if r_pass_time - max_interval < allowable_error:
                        if real_error < allowable_error:
                            # ls_action = (t1, a1, t3, a3, v_rem)
                            ls_action = list(self.vcfunc.get_action_params(desire_reaching_time - c_ts, m_dis, m_v0))
                            ls_action.append(c_ts)
                            self.mavh_action_dic[mavh] = ls_action
                            self.cre_rps = r_pass_time
                            # dic = {mavh: ls_action}
                    elif waiting_time > 60:
                        allowable_error = ls_allowable_error[3]  # 7, 9
                        # if r_pass_time - max_interval < allowable_error:
                        if real_error < allowable_error:
                            # ls_action = (t1, a1, t3, a3, v_rem)
                            ls_action = list(self.vcfunc.get_action_params(desire_reaching_time - c_ts, m_dis, m_v0))
                            ls_action.append(c_ts)
                            self.mavh_action_dic[mavh] = ls_action
                            self.cre_rps = r_pass_time
                            # dic = {mavh: ls_action}
                    self.dic_mavh_actionP  = {mavh:ls_action}
        return self.dic_mavh_actionP

    def apply_mavh_action(self, step, dic_mavh_actionP, mavh):
        '''

        :param dic_mavh_actionP:
                self.mavh_action_dic
        :return:
        '''
        # apply action
        ls_mavhb = self.dic_vehinfo.get('ls_mavhb', None)
        action_mavh = list(dic_mavh_actionP.keys())[0]
        mavh_in_action = False # should be mavh_in_action
        this_round = False
        if (action_mavh in self.mavh_action_dic
                and action_mavh in ls_mavhb):
            # apply action
            self.vcfunc.apply_avh_action(dic_mavh_actionP)
            # flash
            self.vcfunc.flashing2(step, [action_mavh])
            mavh_in_action = True
        if mavh == action_mavh:
            this_round = True # if not this round, no timing judgement
        return mavh_in_action, action_mavh, this_round

    def main_queue_length(self, step, dic_platoon_info, ls_mavhb_sorted, ls_mvb,
                          dic_mplatoon_et, dic_vehinfo, dic_id_speed, ls_rdpt, delta_t):
        '''
        update 25.1.30
        :param step:
        :param dic_platoon_info:
        :param ls_mavhb_sorted:
        :param ls_mvb:
        :param dic_mplatoon_et:
        :param dic_vehinfo:
        :param dic_id_speed:
        :param r_dpt_type:
        :return:
        '''
        self.dic_vehinfo = dic_vehinfo
        self.dic_id_speed = dic_id_speed
        prc.print_message('**in jam mode**')
        # Stop ravh (the first one)
        ravh_f = self.stop_ramp_fleet3()  # first ravh id
        if ravh_f == 'ravh2890':
            pass
        resume_sate = self.check_resume_state4() # 241003update
        # Get ramp fleet travel time (from stop to pass intersection)
        dic_tn = self.get_r_traveltime3(dic_platoon_info)
        dic_r_max = self.get_max_interval8(ls_mavhb_sorted, ls_mvb, dic_mplatoon_et)
        mavh_id, max_interval, final_rpt = self.compare3(dic_r_max, dic_tn)
        if mavh_id == 'mavh3170':
            pass
        # mavh take action; mavh_in_action = True/False
        dic_mavh_actionP = self.get_mavh_action_sp(step, ravh_f, final_rpt, mavh_id,
                                                   max_interval, dic_mplatoon_et, delta_t)
        # dic_mavh_actionP = self.get_mavh_action(step, ravh_f, final_rpt, mavh_id,
        #                                         max_interval, dic_mplatoon_et, par_group)
        mavh_in_action, action_mavh, this_round = self.apply_mavh_action(step, dic_mavh_actionP, mavh_id)
        timing = self.find_timing6(mavh_id, action_mavh, max_interval, final_rpt, mavh_in_action, this_round)

        # record ramp queue length
        ls_rvb_acc = self.get_rvb_acc()
        df_ql = self.data_recorder.get_queue_length(step, ls_rvb_acc, ls_rdpt)

        # Restart ravh after mpv passed the intersection
        self.restart_ramp_fleet(ravh_f)
        return df_ql

    def main(self, step, dic_platoon_info, ls_mavhb_sorted, ls_mvb, dic_mplatoon_et, dic_vehinfo, dic_id_speed):
        # High traffic volume state
        # print('in jam mode')
        self.dic_vehinfo = dic_vehinfo
        self.dic_id_speed = dic_id_speed
        prc.print_message('**in jam mode**')
        # Stop ravh (the first one)
        ravh_f = self.stop_ramp_fleet3()  # first ravh id
        if ravh_f == 'ravh2500':
            pass
        resume_sate = self.check_resume_state4() # 241003update
        # Get ramp fleet travel time (from stop to pass intersection)
        dic_tn = self.get_r_traveltime3(dic_platoon_info)
        dic_r_max = self.get_max_interval8(ls_mavhb_sorted, ls_mvb, dic_mplatoon_et)
        mavh_id, max_interval, final_rpt = self.compare3(dic_r_max, dic_tn)
        if mavh_id == 'mavh3170':
            pass
        # mavh take action; mavh_in_action = True/False
        # mavh_in_action = self.mavh_action2(step, ravh_f, final_rpt, mavh_id, max_interval)
        dic_mavh_actionP = self.get_mavh_action(step, ravh_f, final_rpt, mavh_id, max_interval, dic_mplatoon_et)
        mavh_in_action, action_mavh, this_round = self.apply_mavh_action(step, dic_mavh_actionP, mavh_id)
        timing = self.find_timing6(mavh_id, action_mavh, max_interval, final_rpt, mavh_in_action, this_round)
        # Restart ravh after mpv passed the intersection
        self.restart_ramp_fleet(ravh_f)

class ShiftMode:
    def __init__(self, traci, av_p):
        self.regular_mode = True
        self.jam_mode = False
        self.traci = traci
        self.av_p = av_p

    def determine_mode2(self, ls_mvb, ls_ravhb):
        '''
        normal => jam mode
        :return:
        '''
        min_gap = 4.5
        vehicle_length = 5
        max_jam_vnum = 5
        min_plength = max_jam_vnum*vehicle_length + (max_jam_vnum-1)*min_gap # 5*5 +4*4.5 (47.5)
        # the veh num in min_plength (len(ls_veh_f)), if len(ls_veh_f) > max_jam_vnum, it's in jam condition

        ls_veh_f = [] # filtered
        # get veh number on the last min_plength (m) of inflow_highway
        ih_length = 600
        for veh_id in ls_mvb:
            vehicle_pos = self.traci.vehicle.getLanePosition(veh_id)
            if vehicle_pos >= ih_length-min_plength:
                ls_veh_f.append(veh_id)

        if self.regular_mode and len(ls_veh_f) >= max_jam_vnum:
            # on jam condition
            self.regular_mode = False
            self.jam_mode = True

        if self.jam_mode and len(ls_ravhb) < 1:
            self.regular_mode = True
            self.jam_mode = False

        return self.regular_mode, self.jam_mode

    def determine_mode4(self, ls_mvb, ls_ravhb):
        '''
        241122 update: as platoon become longer, 1 lead 7
        unjam condition needs to update
        normal => jam mode
        :return:
        '''
        min_gap = 4.5
        vehicle_length = 5
        max_jam_vnum = 5
        min_plength = max_jam_vnum*vehicle_length + (max_jam_vnum-1)*min_gap # 5*5 +4*4.5 (47.5)
        # the veh num in min_plength (len(ls_veh_f)), if len(ls_veh_f) > max_jam_vnum, it's in jam condition

        ls_veh_f = [] # filtered
        # get veh number on the last min_plength (m) of inflow_highway
        ih_length = 600
        for veh_id in ls_mvb:
            vehicle_pos = self.traci.vehicle.getLanePosition(veh_id)
            if vehicle_pos >= ih_length-min_plength:
                ls_veh_f.append(veh_id)

        if self.regular_mode and len(ls_veh_f) >= max_jam_vnum:
            # on jam condition
            self.regular_mode = False
            self.jam_mode = True

        if self.jam_mode and len(ls_ravhb) < 1 and len(ls_veh_f) < max_jam_vnum-1: # new condtion: len(ls_veh_f) < max_jam_vnum
            self.regular_mode = True
            self.jam_mode = False

        return self.regular_mode, self.jam_mode