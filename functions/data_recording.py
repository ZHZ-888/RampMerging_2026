import pandas as pd
import re

'''
ascending (asc) or 
descending (desc)
'''

class DataRecording:
    def __init__(self, traci, sim_step=0.1):
        self.traci = traci
        self.sim_step = sim_step
        self.ls_r_veh_net_his = [] # rv emerged; all ramp veh that have emerged
        self.ls_m_veh_net_his = [] # mv emerged; all mainline veh that have emerged
        self.ls_vehid_last_step = [] # vehid of last step
        self.ls_m_leader_his_asc = [] # ls_m_leader_his_asc
        # default information
        try:
            self.length_ramp = self.traci.lane.getLength('inflow_merge_0')
            self.length_ih = self.traci.lane.getLength('inflow_highway_0')
        except:
            self.length_ramp = self.traci.lane.getLength('ramp_0')
            self.length_ih = self.traci.lane.getLength('upstream_0')

        self.dic_vid_groups = {} # ori: dic_vehinfo
        self.dic_leader_ptype = {} # platoon type => {avhid:"AHH", ...}
        self.dic_veh_hinfo = {}

        # ["vid", "time", "dis", "speed"]
        self.data_vehinfo = [] # record vid and its speed and corresponding time
        self.ls_hinfo = [] # headway info
        self.throughput_count = 0
        self.counted_vehicles = set()  # To record vehicles that have already been counted

        self.queue_log = [] # [[step, queue_length], []]

        self.dic_pos = {} # {id1: pos1, id2: pos2}
        self.dic_dis = {}
        self.dic_speed = {}
        self.dic_lane = {}

        '''
        dic_platoon_members = 
                    {'mav19': ['mav19', 'mhv46', 'mhv64', 'mhv74', 'mhv86'], 
                     'veh_2': ['veh_2', 'veh_4']}
        '''
        self.dic_member_to_leader = {} # only multi-lane scenario have this dic; leader => members
        self.merge_control_length = 800  # the length of merging control section


    def record_vehinfo(self): # for single_lane scenario
        '''
        ONE-LANE MERGING ROAD NETWORK
        :param self:
        :return:
        '''
        # tuple, all vehicles on simulation road on current step
        ls_vehid = self.traci.vehicle.getIDList()  # no order
        self._build_step_cache(ls_vehid)

        # head MAV before merging (ih => inflow_highway)
        ls_m_leader_net = [vid for vid in ls_vehid if 'mavh' in vid]  # scripts road head av (leader_AV)
        tup_m_veh_up = self.traci.lane.getLastStepVehicleIDs(
            'inflow_highway_0')  # current veh on merging section (mainlane)
        ls_m_veh_up = list(tup_m_veh_up)  # current veh on merging section (mainlane)
        ls_m_leader_up = [id for id in ls_m_veh_up if 'mavh' in id]  # Head mav Before merging
        ls_m_leader_up_asc = sorted(ls_m_leader_up,
                                     key=lambda x: int(''.join(filter(str.isdigit, x))))  # min=>max; asc (ascending)
        ls_m_veh_net = [vid for vid in ls_vehid if 'm' in vid]

        # r, ramp
        ls_r_leader_net = [vid for vid in ls_vehid if 'ravh' in vid]  # ramp AV leader (history)
        ls_r_veh_net = [vid for vid in ls_vehid if
                        'r' in vid]  # all veh id from ramp, [rav 120, rav 100, ravh 110]
        ls_r_veh_net_asc = sorted(ls_r_veh_net, key=lambda x: int(''.join(filter(str.isdigit, x))))  # asc
        # head RV before merging
        tup_r_veh_up = self.traci.edge.getLastStepVehicleIDs('inflow_merge')  # ramp vehicle current
        ls_r_veh_up = list(tup_r_veh_up)  # ramp veh before merging
        ls_r_leader_up = [id for id in tup_r_veh_up if
                          'avh' in id]  # list of rav leader (head) before merging (large=>small/new=>old/max=>min)
        ls_r_leader_up_asc = sorted(ls_r_leader_up, key=lambda x: int(''.join(filter(str.isdigit, x))))  # min=>max
        # head AV before merging (mainline and ramp)
        ls_mr_leader_up = ls_m_leader_up + ls_r_leader_up  # ls_mr_leader_up

        # info of last step
        ls_r_veh_net_last = [vid for vid in self.ls_vehid_last_step if 'r' in vid]  # ramp veh last step
        ls_r_veh_net_last_asc = sorted(ls_r_veh_net_last, key=lambda x: int(''.join(filter(str.isdigit, x))),
                                       reverse=False)  # sorted

        # Record historical vehicles that have ever appeared in the network (deduplicated)
        # Order of ls_*_veh_net_his is NOT used anywhere; it only serves as a unique container
        self.ls_r_veh_net_his.extend(ls_r_veh_net)
        self.ls_r_veh_net_his = list(dict.fromkeys(self.ls_r_veh_net_his))
        self.ls_m_veh_net_his.extend(ls_m_veh_net)
        self.ls_m_veh_net_his = list(dict.fromkeys(self.ls_m_veh_net_his))
        # Extract historical leaders and sort ONLY the final required result (descending by ID index)
        # ravh: ramp AV head vehicle, mavh: mainline AV head vehicle
        self.ls_r_leader_net_his_desc = sorted(
            (vid for vid in self.ls_r_veh_net_his if vid.startswith('ravh')),
            key=lambda vid: int(vid[4:]),
            reverse=True
        )
        self.ls_m_leader_net_his_desc = sorted(
            (vid for vid in self.ls_m_veh_net_his if vid.startswith('mavh')),
            key=lambda vid: int(vid[4:]),
            reverse=True
        )

        # update ls_vehid_last_step
        self.ls_vehid_last_step = ls_vehid

        # vehicle info for control
        self.dic_vid_groups['ls_vehid'] = ls_vehid
        self.dic_vid_groups['ls_m_leader_net'] = ls_m_leader_net  # small => big
        self.dic_vid_groups['ls_m_leader_up'] = ls_m_leader_up  # big => small
        self.dic_vid_groups['ls_m_leader_up_asc'] = ls_m_leader_up_asc
        self.dic_vid_groups['ls_m_veh_up'] = ls_m_veh_up

        self.dic_vid_groups['ls_r_leader_net'] = ls_r_leader_net
        self.dic_vid_groups['ls_r_leader_up'] = ls_r_leader_up
        self.dic_vid_groups['ls_r_leader_up_asc'] = ls_r_leader_up_asc
        self.dic_vid_groups['ls_r_veh_up'] = ls_r_veh_up  # ramp veh before merging

        self.dic_vid_groups['ls_r_veh_net_asc'] = ls_r_veh_net_asc
        self.dic_vid_groups['ls_r_veh_net_last_asc'] = ls_r_veh_net_last_asc
        self.dic_vid_groups['ls_mr_leader_up'] = ls_mr_leader_up
        return self.dic_vid_groups

    def record_multi_lane_info(self, length_ms=800):
        '''
        ms => on merging section
        m => from mainlane
        r => from ramp
        :param length_ms:
        :return:
        '''
        # tuple, all vehicles on simulation road on current step
        ls_vehid = self.traci.vehicle.getIDList()  # no order
        self._build_step_cache(ls_vehid)

        # current leader on net from mainlane
        # head MAV on mainlane merging control section; self.ls_m_leader_his_asc, ls_ms_veh_up
        ls_m_leader_net = [
            vid for vid in self.ls_m_leader_his_asc
            if vid in ls_vehid
        ]
        # current veh on mainlane (inflow_highway_0, ih)
        tup_ih_veh_up = self.traci.lane.getLastStepVehicleIDs('inflow_highway_0')
        # current veh on merging section; length_ms=800, self.length_ih
        ls_ms_veh_up = [
            vid for vid in tup_ih_veh_up
            if self.dic_pos.get(vid, -1.0) >= self.length_ih - length_ms
        ]
        # current veh on platoon formation section
        ls_pf_veh = [
            vid for vid in tup_ih_veh_up
            if self.dic_pos.get(vid, -1.0) < self.length_ih - length_ms
        ]
        # leader on platoon formation
        ls_pf_leader = [
            vid for vid in self.ls_m_leader_his_asc
            if vid in ls_pf_veh
        ]
        ls_ms_leader_net = [
                            vid for vid in ls_m_leader_net
                            if vid not in ls_pf_leader
                        ]
        # Head mav Before merging
        ls_ms_leader_up = [vid for vid in ls_ms_veh_up if vid in self.ls_m_leader_his_asc]
        ls_ms_leader_up_asc = sorted(ls_ms_leader_up, key=lambda x: int(''.join(filter(str.isdigit, x))))  # min=>max; asc (ascending)
        ls_m_veh_net = [vid for vid in ls_vehid if 'm' in vid]

        # r, ramp
        ls_r_leader_net = [vid for vid in ls_vehid if 'ravh' in vid]  # ramp AV leader (history)
        ls_r_veh_net = [vid for vid in ls_vehid if 'r' in vid]  # all veh id from ramp, [rav 120, rav 100, ravh 110]
        ls_r_veh_net_asc = sorted(ls_r_veh_net, key=lambda x: int(''.join(filter(str.isdigit, x)))) # asc
        # head RV before merging
        tup_r_veh_up = self.traci.edge.getLastStepVehicleIDs('inflow_merge')  # ramp vehicle current
        ls_r_veh_up = list(tup_r_veh_up)  # ramp veh before merging
        ls_r_leader_up = [id for id in tup_r_veh_up if
                           'avh' in id]  # list of rav leader (head) before merging (large=>small/new=>old/max=>min)
        ls_r_leader_up_asc = sorted(ls_r_leader_up, key=lambda x: int(''.join(filter(str.isdigit, x))))  # min=>max

        # head AV before merging (mainline and ramp)
        ls_mr_leader_up = ls_ms_leader_up + ls_r_leader_up  # ls_mr_leader_up

        # info of last step
        ls_r_veh_net_last = [vid for vid in self.ls_vehid_last_step if 'r' in vid]  # ramp veh last step
        ls_r_veh_net_last_asc = sorted(ls_r_veh_net_last, key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=False)  # sorted

        # Record historical vehicles that have ever appeared in the network (deduplicated)
        # Order of ls_*_veh_net_his is NOT used anywhere; it only serves as a unique container
        self.ls_r_veh_net_his.extend(ls_r_veh_net)
        self.ls_r_veh_net_his = list(dict.fromkeys(self.ls_r_veh_net_his))
        # ravh: ramp AV head vehicle, mavh: mainline AV head vehicle
        self.ls_r_leader_net_his_desc = sorted(
            (vid for vid in self.ls_r_veh_net_his if vid.startswith('ravh')),
            key=lambda vid: int(vid[4:]),
            reverse=True
        )

        self.ls_m_leader_net_his_desc = (
            sorted(self.ls_m_leader_his_asc, key=lambda x: int(''.join(filter(str.isdigit, x))),
                    reverse=True))  # sorted

        # update ls_vehid_last_step
        self.ls_vehid_last_step = ls_vehid

        # MULTI-LANE MERGING ROAD NETWORK (platoon formation control)
        # m, mainline (inflow_highway); A(0)
        ls_ihA = list(self.traci.lane.getLastStepVehicleIDs("inflow_highway_0"))  # inflow_highway lane A, all veh
        ls_ihA_av = [vid for vid in ls_ihA if 'av' in vid]
        ls_ihA_hv = [vid for vid in ls_ihA if 'hv' in vid]
        ls_ihA_ms = [vid for vid in ls_ihA
                        if self.dic_pos.get(vid, -1.0) > (self.length_ih - 800)
                        ]
        ls_ihA_av_ms = [vid for vid in ls_ihA_av
                        if self.dic_pos.get(vid, -1.0) > (self.length_ih - 800)
                        ] # m => merging section

        if 'inflow_highway_1' in self.traci.lane.getIDList():
            ls_ihB = list(self.traci.lane.getLastStepVehicleIDs("inflow_highway_1"))
            ls_ihB_av = [vid for vid in ls_ihB if 'av' in vid]
            ls_ihB_hv = [vid for vid in ls_ihB if 'hv' in vid]
            ls_ihAB_hv = ls_ihA_hv + ls_ihB_hv
            ls_ihAB_av = ls_ihA_av + ls_ihB_av
        else:
            ls_ihB = []
            ls_ihB_av = []
            ls_ihB_hv = []
            ls_ihAB_hv = []
            ls_ihAB_av = []

        # veh on upstream_0(A) & upstream_1(B)
        if 'upstream_0' in self.traci.lane.getIDList():
            ls_upA = list(self.traci.lane.getLastStepVehicleIDs("upstream_0")) # vehicles on upstream; big=>small
            ls_upA_av = [vid for vid in ls_upA if 'av' in vid]
            ls_upA_hv = [vid for vid in ls_upA if 'hv' in vid] # decrease
            ls_upB = list(self.traci.lane.getLastStepVehicleIDs("upstream_1"))
            ls_upB_av = [vid for vid in ls_upB if 'av' in vid]
        else:
            ls_upA = []
            ls_upA_av = []
            ls_upA_hv = []
            ls_upB = []
            ls_upB_av = []

        # weaving section
        if 'ws_1' in self.traci.lane.getIDList():
            ls_wsB = list(self.traci.lane.getLastStepVehicleIDs("ws_1"))
            ls_wsB_av = [vid for vid in ls_wsB if 'av' in vid]
            ls_wsB_hv = [vid for vid in ls_wsB if 'hv' in vid]  # decrease
        else:
            ls_wsB = []
            ls_wsB_av = []
            ls_wsB_hv = []

        if 'ws_2' in self.traci.lane.getIDList():
            ls_wsC = list(self.traci.lane.getLastStepVehicleIDs("ws_2"))
            ls_wsC_av = [vid for vid in ls_wsC if 'av' in vid]
            ls_wsC_hv = [vid for vid in ls_wsC if 'hv' in vid]  # decrease
        else:
            ls_wsC = []
            ls_wsC_av = []
            ls_wsC_hv = []
        ls_wsBC_hv = ls_wsB_hv + ls_wsC_hv

        # center lane (after merging)
        if 'center_0' in self.traci.lane.getIDList():
            ls_centerA = list(self.traci.lane.getLastStepVehicleIDs("center_0"))
            ls_centerA_av = [vid for vid in ls_centerA if 'av' in vid]
            ls_centerA_hv = [vid for vid in ls_centerA if 'hv' in vid]  # decrease
        else:
            ls_centerA = []
            ls_centerA_av = []
            ls_centerA_hv = []

        if 'center_1' in self.traci.lane.getIDList():
            ls_centerB = list(self.traci.lane.getLastStepVehicleIDs("center_1"))
            ls_centerB_av = [vid for vid in ls_centerB if 'av' in vid]
            ls_centerB_hv = [vid for vid in ls_centerB if 'hv' in vid]  # decrease
        else:
            ls_centerB = []
            ls_centerB_av = []
            ls_centerB_hv = []

        # update ls_vehid_last_step
        self.ls_vehid_last_step = ls_vehid

        # vehicle info for merging control
        self.dic_vid_groups['ls_vehid'] = ls_vehid
        self.dic_vid_groups['ls_m_leader_net'] = ls_ms_leader_net  # small => big
        self.dic_vid_groups['ls_m_leader_up'] = ls_ms_leader_up  # big => small
        self.dic_vid_groups['ls_m_leader_up_asc'] = ls_ms_leader_up_asc
        self.dic_vid_groups['ls_m_veh_up'] = ls_ms_veh_up

        self.dic_vid_groups['ls_r_leader_net'] = ls_r_leader_net
        self.dic_vid_groups['ls_r_leader_up'] = ls_r_leader_up
        self.dic_vid_groups['ls_r_leader_up_asc'] = ls_r_leader_up_asc
        self.dic_vid_groups['ls_r_veh_up'] = ls_r_veh_up  # ramp veh before merging

        self.dic_vid_groups['ls_r_veh_net_asc'] = ls_r_veh_net_asc
        self.dic_vid_groups['ls_r_veh_net_last_asc'] = ls_r_veh_net_last_asc
        # max => min (ls_mr_leader_up = ls_ms_leader_up+ls_r_leader_up)
        self.dic_vid_groups['ls_mr_leader_up'] = ls_mr_leader_up

        # veh on inflow_highway_0
        self.dic_vid_groups['ls_ihA'] = ls_ihA  # decrease
        self.dic_vid_groups['ls_ihA_av'] = ls_ihA_av
        self.dic_vid_groups['ls_ihA_hv'] = ls_ihA_hv
        self.dic_vid_groups['ls_ihA_ms'] = ls_ihA_ms
        self.dic_vid_groups['ls_ihA_av_ms'] = ls_ihA_av_ms  # ms => merging section
        self.dic_vid_groups['ls_ihB'] = ls_ihB
        self.dic_vid_groups['ls_ihB_av'] = ls_ihB_av
        self.dic_vid_groups['ls_ihB_hv'] = ls_ihB_hv
        self.dic_vid_groups['ls_ihAB_av'] = ls_ihAB_av
        self.dic_vid_groups['ls_ihAB_hv'] = ls_ihAB_hv

        # veh on upstream_A and upstream_B
        self.dic_vid_groups['ls_upA'] = ls_upA
        self.dic_vid_groups['ls_upA_av'] = ls_upA_av
        self.dic_vid_groups['ls_upA_hv'] = ls_upA_hv
        self.dic_vid_groups['ls_upB'] = ls_upB
        self.dic_vid_groups['ls_upB_av'] = ls_upB_av

        # veh on weaving section (ws)
        self.dic_vid_groups['ls_wsB_hv'] = ls_wsB_hv
        self.dic_vid_groups['ls_wsC_hv'] = ls_wsC_hv
        self.dic_vid_groups['ls_wsBC_hv'] = ls_wsBC_hv

        # veh on center
        self.dic_vid_groups['ls_centerA_av'] = ls_centerA_av
        self.dic_vid_groups['ls_centerA_hv'] = ls_centerA_hv
        self.dic_vid_groups['ls_centerB_av'] = ls_centerB_av
        return self.dic_vid_groups

    def get_hv_leader(self, hv_id, m=True):
        '''
        find the leader of hv_id
        :param hv_id:
        :param m:
        :return:
        '''
        if self.dic_member_to_leader:
            if hv_id == 'mhv83':
                pass
            leader_id = self.dic_member_to_leader[hv_id]
            return leader_id
        if 'avh' in hv_id:
            leader_id = hv_id
        else:
            hv_time = int(hv_id[3:])
            if m:
                leader_id = max((car for car in self.ls_m_leader_net_his_desc if int(car[4:]) < hv_time),
                           key=lambda x: int(x[4:]), default=None)
            else:
                leader_id = max((car for car in self.ls_r_leader_net_his_desc if int(car[4:]) < hv_time),
                               key=lambda x: int(x[4:]), default=None)
        return leader_id

    def get_avhid_ptype(self, m_dpt_type=None, r_dpt_type=None):
        '''
         get av head id and it's platoon type

        :param m_dpt_type: {4: 'AHHHHHHHHH', 31: 'AHHHHHHHHHHH'}
        :param r_dpt_type:
        :return:  {'mavh40': 'AHHHHHHHHH', 'mavh310': 'AHHHHHHHHHHH', 'mavh580': 'AHHHHHHHH'}
        '''
        if m_dpt_type:
            for key, value in m_dpt_type.items():
                if isinstance(key, str):
                    self.dic_leader_ptype[key] = value
                else:
                    id1 = 'mavh' + str(key*10)
                    self.dic_leader_ptype[id1] = value
        if r_dpt_type:
            for key, value in r_dpt_type.items():
                id2 = 'ravh' + str(key*10)
                self.dic_leader_ptype[id2] = value
        return

    def transform_ls_df(self, ls, ls_column):
        df = pd.DataFrame(ls, columns = ls_column)
        return df

    def record_throughput(self, st, vehicle_ids, edge_id):
        '''
        record throughput
        :param  st: simulation time
                vehicle_ids:
                edge_id:
        :return:
        '''
        for vid in vehicle_ids:
            if self.traci.vehicle.getRoadID(vid) == edge_id and vid not in self.counted_vehicles:
                self.throughput_count += 1
                self.counted_vehicles.add(vid)
        return self.throughput_count*3600/st

    def get_average_speed(self, step, vehicle_ids, jam_mode=None):
        """
        get average speed of all veh per seconds
        vehicle_ids: all vehicles in this step
        :return: speed_log
        """
        if len(vehicle_ids) > 0:
            total_speed = sum(self.dic_speed[vehID] for vehID in vehicle_ids)
            avg_speed = total_speed / len(vehicle_ids)
        else:
            avg_speed = None
        speed_log = [step, avg_speed, jam_mode]
        return speed_log

    def get_queue_length(self, step, ls_rvb_acc, ls_rdpt):
        '''
        sample once per second
        get queue length only employ during Jam Rule
        :return: number of vehicles in ramp queue (slow/no speed)
        
        step: 0.1 seconds
        r_dpt_type: Planned List of Ramp Vehicles to Generate (seconds)
        ravh_f: current first waited veh id

        '''
        if step % 10 == 0:
            if len(ls_rvb_acc) > 0:
                first_stop_veh = ls_rvb_acc[-1]
                first_stop_num = int(re.search(r'\d+', first_stop_veh).group())
                queue_length = sum(first_stop_num/10 <= x <= step/10 for x in ls_rdpt)
            else:
                queue_length = 0
            self.queue_log.append((step, queue_length))
        return self.queue_log

    def _build_step_cache(self, ls_vehid):
        """
           Cache per-step TraCI queries to reduce IPC overhead.
           all vehicle on traffic network
        """

        for vid in ls_vehid:
            # These TraCI calls are expensive; cache them once per step.
            self.dic_pos[vid] = self.traci.vehicle.getLanePosition(vid)
            self.dic_dis[vid] = self.traci.vehicle.getDistance(vid)
            self.dic_speed[vid] = self.traci.vehicle.getSpeed(vid)
            self.dic_lane[vid] = self.traci.vehicle.getLaneID(vid)
        return



