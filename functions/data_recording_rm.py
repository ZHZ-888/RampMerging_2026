import pandas as pd
class DataRecording:
    def __init__(self, traci, sim_step):
        self.traci = traci
        self.sim_step = sim_step
        self.ls_rve = [] # rv emerged
        self.ls_mve = [] # mv emerged
        self.ls_vehid_pre = [] # vehid of last step

        # default information
        self.length_ramp = self.traci.lane.getLength('inflow_merge_0')
        self.length_ih = self.traci.lane.getLength('inflow_highway_0')
        self.dic_vehinfo = {}
        self.dic_avhid_ptype = {} # platoon type => {avhid:"AHH", ...}
        self.dic_veh_hinfo = {}
        self.dic_id_speed = {} # record veh_id and its [speed list]

        # ["veh_id", "time", "dis", "speed"]
        self.data_vehinfo = [] # record veh_id and its speed and corresponding time
        self.ls_hinfo = [] # headway info
        self.throughput_count = 0
        self.counted_vehicles = set()  # 用于记录已经计数过的车辆

    def record_vehinfo(self):
        ls_vehid = self.traci.vehicle.getIDList() # tuple, all vehicles in this step
        # position and types
        ls_mavh = [veh_id for veh_id in ls_vehid if 'mavh' in veh_id]  # scripts road head av
        ls_mav = [veh_id for veh_id in ls_vehid if 'mav' in veh_id]  # scripts road av
        ls_mhv = [veh_id for veh_id in ls_vehid if 'mhv' in veh_id]  # scripts road hv
        ls_ravh = [veh_id for veh_id in ls_vehid if 'ravh' in veh_id]  # ramp road head av
        ls_rav = [veh_id for veh_id in ls_vehid if 'rav' in veh_id] # ramp road av
        ls_rhv = [veh_id for veh_id in ls_vehid if 'rhv' in veh_id]  # ramp road hv
        # rav+rhv, sequence => [rav 120, rav 100, ravh 110]
        ls_rv = [veh_id for veh_id in ls_vehid if 'r' in veh_id]
        ls_rv_s = sorted(ls_rv, key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=False)
        ls_mv = [veh_id for veh_id in ls_vehid if 'm' in veh_id]
        ls_mv_s = sorted(ls_mv, key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=False)

        # ls_rv = sorted(ls_rav + ls_rhv, key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=False)
        # ls_mv = sorted(ls_mav + ls_mhv, key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=False)
        ls_rv_sortedR = sorted(ls_rv, key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=True)

        # record every veh that emerged
        # SET drop duplicates
        self.ls_rve = sorted(list(set(self.ls_rve + ls_rv)), key=lambda x: int(''.join(filter(str.isdigit, x))),
                             reverse=False)
        self.ls_mve = sorted(list(set(self.ls_mve + ls_mv)), key=lambda x: int(''.join(filter(str.isdigit, x))),
                             reverse=False)

        # head MAV before merging
        tup_ihmv = self.traci.edge.getLastStepVehicleIDs('inflow_highway') # mainline mv
        ls_mvb = list(tup_ihmv) # mainline veh before merging
        ls_mavhb = [id for id in tup_ihmv if 'mavh' in id]  # Head mav Before merging
        # ls_BeforeMerging_mav_sorted_Reversed; decreasing sequence => ['mav60', 'mav0']
        ls_mavhb_sortedR = sorted(ls_mavhb, key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=True)
        ls_mavhb_sorted = sorted(ls_mavhb, key=lambda x: int(''.join(filter(str.isdigit, x)))) # min=>max

        # head RV before merging
        tup_rvb = self.traci.edge.getLastStepVehicleIDs('inflow_merge')  # merge road part B, rv Before merging
        ls_rvb = list(tup_rvb) # ramp veh before merging
        ls_ravhb = [id for id in tup_rvb if 'avh' in id]  # list of Head rav before merging (large=>small/new=>old)

        # head AV before merging
        ls_avhb = ls_mavhb+ls_ravhb

        # info of last step
        ls_rv_p = [veh_id for veh_id in self.ls_vehid_pre if 'r' in veh_id] # ramp veh previous step
        ls_rv_ps = sorted(ls_rv_p, key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=False) # sorted

        # update ls_vehid_pre
        self.ls_vehid_pre = ls_vehid
        # through result in dic
        self.dic_vehinfo['ls_vehid'] = ls_vehid
        self.dic_vehinfo['ls_rv_s'] = ls_rv_s
        self.dic_vehinfo['ls_rv_ps'] = ls_rv_ps
        self.dic_vehinfo['ls_mavhb'] = ls_mavhb
        self.dic_vehinfo['ls_mavhb_sorted'] = ls_mavhb_sorted
        self.dic_vehinfo['ls_ravhb'] = ls_ravhb
        self.dic_vehinfo['ls_avhb'] = ls_avhb
        self.dic_vehinfo['ls_rvb'] = ls_rvb # ramp veh before merging
        self.dic_vehinfo['ls_mvb'] = ls_mvb
        self.dic_vehinfo['ls_ravh'] = ls_ravh

        return self.dic_vehinfo

    def get_avhid_ptype(self, m_dpt_type=None, r_dpt_type=None):
        if m_dpt_type:
            for key, value in m_dpt_type.items():
                id1 = 'mavh' + str(key*10)
                self.dic_avhid_ptype[id1] = value
        if r_dpt_type:
            for key, value in r_dpt_type.items():
                id2 = 'ravh' + str(key*10)
                self.dic_avhid_ptype[id2] = value
        return self.dic_avhid_ptype


    def get_veh_headwayinfo(self, ls_vehid):
        """
        get the preceding veh_id of veh, and corresponding spacing and time_headway
        GENERALLY only focus on avh before merging
        :return: dic_veh_hinfo (headway info)
        """
        # ls_mavhb_sorted
        # 1. get preceding veh_id
        dic_veh_hinfo = {}
        for id in ls_vehid:
            p_veh_info = self.traci.vehicle.getLeader(id, float('inf')) # preceding
            if p_veh_info is not None:
                # 2. get spacing
                p_id, dis = p_veh_info
                speed = self.traci.vehicle.getSpeed(id)
                if speed > 0:
                    # 3. get time_headway
                    time_headway = dis/speed
                else:
                    time_headway = None
            else:
                p_id = None
                dis = None
                time_headway = None
            dic_veh_hinfo[id] = [p_id, dis, time_headway]
            self.dic_veh_hinfo = dic_veh_hinfo
        return self.dic_veh_hinfo

    def get_veh_headwayinfo2(self, ls_vehid):
        """
        get the preceding veh_id of veh, and corresponding spacing and time_headway
        GENERALLY only focus on avh before merging
        :return: dic_veh_hinfo (headway info)
        """
        # ls_mavhb_sorted
        # 1. get preceding veh_id
        dic_veh_hinfo = {}
        for id in ls_vehid:
            if id == 'mavh6670':
                pass
            p_veh_info = self.traci.vehicle.getLeader(id, float('inf')) # preceding
            if p_veh_info is not None:
                # 2. get spacing
                p_id, dis = p_veh_info
                speed = self.traci.vehicle.getSpeed(id)
                if speed > 0:
                    # 3. get time_headway
                    time_headway = dis/speed
                else:
                    time_headway = None
            else:
                p_id = None
                dis = None
                time_headway = None
            dic_veh_hinfo[id] = [p_id, dis, time_headway]
        return dic_veh_hinfo

    def organize_veh_hinfo(self, c_ts, dic_veh_hinfo):
        """
        hinfo => headwayinfo
        turn dic_vhe_hinfo into list, and add c_ts
        :param c_ts:
        :return: record all [[veh_id, leader_id, headway, time_headway, time],[]]
        """
        flattened_list = [[key]+value+[c_ts] for key, value in dic_veh_hinfo.items()]
        if len(flattened_list) > 0:
            self.ls_hinfo.extend(flattened_list) # use extend instead of append!!! as append will cause redundant []
        return self.ls_hinfo

    def transform_ls_df(self, ls, ls_column):
        df = pd.DataFrame(ls, columns = ls_column)
        return df

    def record_throughput(self, st, vehicle_ids, edge_id):
        '''
        record throughput
        :param vehicle_ids:
        :param edge_id:
        :return:
        '''
        for veh_id in vehicle_ids:
            if self.traci.vehicle.getRoadID(veh_id) == edge_id and veh_id not in self.counted_vehicles:
                self.throughput_count += 1
                self.counted_vehicles.add(veh_id)
        return self.throughput_count*3600/st


    def record_vehSpeed(self, vehicle_ids):
        for veh_id in vehicle_ids:
            if veh_id not in self.dic_id_speed:
                self.dic_id_speed[veh_id] = []
            speed = self.traci.vehicle.getSpeed(veh_id)
            self.dic_id_speed[veh_id].append(speed)
        return self.dic_id_speed

    def record_vehData(self, vehicle_ids, c_ts):
        row = []
        for veh_id in vehicle_ids:
            speed = self.traci.vehicle.getSpeed(veh_id)
            dis = self.traci.vehicle.getDistance(veh_id) # running distance
            row = [veh_id, c_ts, speed, dis]
            self.data_vehinfo.append(row)
        return self.data_vehinfo

    def get_average_speed(self, step, jam_mode, vehicle_ids):
        """
        get average speed of all veh per seconds
        vehicle_ids: all vehicles in this step
        :return:
        """
        if len(vehicle_ids) > 0:
            total_speed = sum(self.traci.vehicle.getSpeed(vehID) for vehID in vehicle_ids)
            avg_speed = total_speed / len(vehicle_ids)
        else:
            avg_speed = None
        ls_r = [step, avg_speed, jam_mode]
        return ls_r

    def get_average_speed2(self, step, jam_mode, vehicle_ids):
        """
        split ramp speed and mainline speed, therefore every step has two values (ramp and scripts)
        get average speed of all veh per seconds
        vehicle_ids: all vehicles in this step
        :return:
        """
        # ls_ramp_veh = []
        ls_r_v = [] # ramp vehicles' velocity
        # ls_main_veh = [] # including mainline and center
        ls_m_v = [] # including mainline and center
        if len(vehicle_ids) > 0:
            for veh_id in vehicle_ids:
                lane_id = self.traci.vehicle.getLaneID(veh_id)
                if lane_id == 'inflow_merge_0':
                    ls_r_v.append(self.traci.vehicle.getSpeed(veh_id))
                else:
                    ls_m_v.append(self.traci.vehicle.getSpeed(veh_id))
            # ramp avg_speed
            if len(ls_r_v) > 0:
                ramp_avg_speed = sum(ls_r_v) / len(ls_r_v)
            else:
                ramp_avg_speed = None
            # scripts avg_speed
            if len(ls_m_v) > 0:
                main_avg_speed = sum(ls_m_v) / len(ls_m_v)
            else:
                main_avg_speed = None
            # overall avg_speed
            total_speed = sum(self.traci.vehicle.getSpeed(vehID) for vehID in vehicle_ids)
            avg_speed = total_speed / len(vehicle_ids)
        else:
            ramp_avg_speed = None
            main_avg_speed = None
            avg_speed = None
        ls_r = [step, ramp_avg_speed, main_avg_speed, avg_speed, jam_mode]
        return ls_r



