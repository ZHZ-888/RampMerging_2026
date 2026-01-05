# pf => platoon formation
import pandas as pd
import re
class DataRecording:
    def __init__(self, traci, sim_step):
        self.traci = traci
        self.sim_step = sim_step
        self.ls_rve = [] # rv emerged; all ramp veh that have emerged
        self.ls_mve = [] # mv emerged; all mainline veh that have emerged
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
        self.counted_vehicles = set()  # To record vehicles that have already been counted

        self.ls_step = []
        self.ls_queue_length = []

    def record_vehinfo(self):
        # default sequence: DECREASE
        ls_upA = [] # up => upstream, lane 0 (A)
        ls_upA_av = []
        ls_upA_hv = []
        ls_upB = [] # lane 1 (B)

        # tuple, all vehicles on simulation road in this step
        tup_vehid = self.traci.vehicle.getIDList()
        ls_vehid = sorted(list(tup_vehid), key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=True)

        # vehicles on inflow_highway
        ls_ihA = list(self.traci.lane.getLastStepVehicleIDs("inflow_highway_0")) # inflow_highway lane A, all veh
        ls_ihA_av = [veh_id for veh_id in ls_ihA if 'av' in veh_id]
        ls_ihA_hv = [veh_id for veh_id in ls_ihA if 'hv' in veh_id]

        # veh on upstream_0
        ls_upA = list(self.traci.lane.getLastStepVehicleIDs("upstream_0")) # vehicles on upstream; big=>small
        ls_upA_av = [veh_id for veh_id in ls_upA if 'av' in veh_id]
        ls_upA_hv = [veh_id for veh_id in ls_upA if 'hv' in veh_id] # decrease
        ls_upB = list(self.traci.lane.getLastStepVehicleIDs("upstream_1"))
        ls_upB_av = [veh_id for veh_id in ls_upB if 'av' in veh_id]

        # update ls_vehid_pre
        self.ls_vehid_pre = ls_vehid
        # through result in dic
        self.dic_vehinfo['ls_vehid'] = ls_vehid
        # veh on inflow_highway_0
        self.dic_vehinfo['ls_ihA'] = ls_ihA # decrease
        self.dic_vehinfo['ls_ihA_av'] = ls_ihA_av
        self.dic_vehinfo['ls_ihA_hv'] = ls_ihA_hv

        # veh on upstream_0
        self.dic_vehinfo['ls_upA'] = ls_upA
        self.dic_vehinfo['ls_upA_av'] = ls_upA_av
        self.dic_vehinfo['ls_upA_hv'] = ls_upA_hv
        self.dic_vehinfo['ls_upB'] = ls_upB  # big => small
        self.dic_vehinfo['ls_upB_av'] = ls_upB_av

        return self.dic_vehinfo

    def get_hv_leader(self, hv_id, m=True):
        '''
        find the leader of the platoon (within hv_id)
        :param hv_id:
        :param m:
        :return:
        '''
        # mavh1330
        # mhv1290
        # mav1340
        if 'avh' in hv_id:
            hv_time = int(hv_id[4:])
            avh_id = hv_id
        else:
            hv_time = int(hv_id[3:])
            if m:
                avh_id = max((car for car in self.ls_mavhe_r if int(car[4:]) < hv_time),
                           key=lambda x: int(x[4:]), default=None)
            else:
                avh_id = max((car for car in self.ls_ravhe_r if int(car[4:]) < hv_time),
                               key=lambda x: int(x[4:]), default=None)
        return avh_id

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

    def get_queue_length(self, step, ls_rvb_acc, ls_rdpt):
        '''
        sample once per second
        get queue length only employ during Jam Rule
        :return: number of vehicles in ramp queue (slow/no speed)
        
        step: 0.1 seconds
        r_dpt_type: Planned List of Ramp Vehicles to Generate (seconds)
        ravh_f: current first waited veh id

        '''
        if step % 10 ==0:
            if len(ls_rvb_acc) > 0:
                first_stop_veh = ls_rvb_acc[-1]
                first_stop_num = int(re.search(r'\d+', first_stop_veh).group())
                queue_length = sum(first_stop_num/10 <= x <= step/10 for x in ls_rdpt)
            else:
                queue_length = 0
            # print(f'step, queue length: {step, queue_length}')
            self.ls_step.append(step)
            self.ls_queue_length.append(queue_length)
        df_ql = pd.DataFrame({'step': self.ls_step, 'queue_length': self.ls_queue_length}) # ql: queue length
        return df_ql

