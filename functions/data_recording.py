import pandas as pd
import re
from collections import defaultdict

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
        self.throughput_count = 0
        self.counted_vehicles = set()  # To record vehicles that have already been counted

        self.queue_log = [] # [[step, queue_length], []]
        self.dic_pos = {} # {id1: pos1, id2: pos2}
        self.dic_dis = {}
        self.dic_speed = {}
        self.dic_lane = {}
        self.ls_tail_ids = []
        self.dic_lane_length = {} # {lane_id: lane_length}; record lane_id and it's length, save traci calls

        # Phase 1 optimization caches
        self.dic_leader_gap = {}  # {follower_id: gap_to_leader} - pre-computed gaps
        self.ls_lane0_veh = []  # Vehicles on lane 0 (main lane 0)
        self.ls_lane1_veh = []  # Vehicles on lane 1 (main lane 1)

        '''
        dic_platoon_members = 
                    {'mav19': ['mav19', 'mhv46', 'mhv64', 'mhv74', 'mhv86'], 
                     'veh_2': ['veh_2', 'veh_4']}
        dic_member_to_leader = {follower_id: leader_id}
        '''
        self.dic_follower_state = {} # {follower_id: [state, leader_id]}
        self.dic_member_to_leader = {} # only multi-lane scenario have this dic; from platoon_formation2.py
        self.merge_control_length = 800  # the length of merging control section

        # multi-lane scenario => (27.78 m/s => 100 km/h)
        self.max_speed = 27.78 if self.length_ih > 1500 else 25

        self.ls_features = []
        self.leader_record_counter = defaultdict(int)
        self.dic_tail_arrived_ws = {} # {leader_id: [tail_id, arrival_time]}
        self.dic_platoon_info = {} # {vid:[type, tail_id, length1, length2...]}

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
        self.dic_vid_groups['ls_m_veh_net'] = ls_m_veh_net

        self.dic_vid_groups['ls_r_leader_net'] = ls_r_leader_net
        self.dic_vid_groups['ls_r_leader_up'] = ls_r_leader_up
        self.dic_vid_groups['ls_r_leader_up_asc'] = ls_r_leader_up_asc
        self.dic_vid_groups['ls_r_veh_up'] = ls_r_veh_up  # ramp veh before merging
        self.dic_vid_groups['ls_r_veh_net'] = ls_r_veh_net

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
        ls_m_veh_net = sorted(ls_m_veh_net, key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=True)

        # r, ramp
        ls_r_leader_net = [vid for vid in ls_vehid if 'ravh' in vid]  # ramp AV leader (history)
        ls_r_veh_net = [vid for vid in ls_vehid if 'r' in vid]  # all veh id from ramp, [rav 120, rav 100, ravh 110]
        ls_r_veh_net = sorted(ls_r_veh_net, key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=True)
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
        if 'ws_0' in self.traci.lane.getIDList():
            ls_wsA = list(self.traci.lane.getLastStepVehicleIDs("ws_0"))
            ls_wsA_av = [vid for vid in ls_wsA if 'av' in vid]
            ls_wsA_hv = [vid for vid in ls_wsA if 'hv' in vid]  # decrease
        else:
            ls_wsB = []
            ls_wsB_av = []
            ls_wsB_hv = []

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
        self.dic_vid_groups['ls_m_veh_net'] = ls_m_veh_net

        self.dic_vid_groups['ls_r_leader_net'] = ls_r_leader_net
        self.dic_vid_groups['ls_r_leader_up'] = ls_r_leader_up
        self.dic_vid_groups['ls_r_leader_up_asc'] = ls_r_leader_up_asc
        self.dic_vid_groups['ls_r_veh_up'] = ls_r_veh_up  # ramp veh before merging
        self.dic_vid_groups['ls_r_veh_net'] = ls_r_veh_net
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
        self.dic_vid_groups['ls_wsA'] = ls_wsA
        self.dic_vid_groups['ls_wsB'] = ls_wsB
        self.dic_vid_groups['ls_wsB_av'] = ls_wsB_av
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
        if hv_id == 'rhv1000':
            pass
        if hv_id in self.dic_member_to_leader:
            leader_id = self.dic_member_to_leader[hv_id]
            return leader_id
        if 'avh' in hv_id:
            leader_id = hv_id
        else:
            hv_time = self._get_vid_digit(hv_id)
            vid_digit = self._get_vid_digit
            if m:
                # max(iterable, key=func)
                leader_id = max((vid for vid in self.ls_m_leader_net_his_desc if vid_digit(vid) < hv_time),
                                key=vid_digit, default=None)
                if leader_id is None:
                    pass
            else:
                leader_id = max((vid for vid in self.ls_r_leader_net_his_desc if vid_digit(vid) < hv_time),
                                key=vid_digit, default=None)
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

    def get_vid_states(self, vid):
        """
        get speed, lane, position, distance of this ID
        """
        dic_vid_states = {'v': None, 'lane': None, 'pos': None, 'dis': None}
        if not vid:
            return dic_vid_states
        try:
            v = self.dic_speed.get(vid)  # cached
            lane_id = self.dic_lane.get(vid) # lane_id = self.traci.vehicle.getLaneID(vid)
            pos = self.dic_pos.get(vid) # pos = self.traci.vehicle.getLanePosition(vid)
            dis = self.dic_dis.get(vid)
        except Exception:
            return dic_vid_states
        dic_vid_states['v'] = v
        dic_vid_states['lane'] = lane_id
        dic_vid_states['pos'] = pos
        dic_vid_states['dis'] = dis
        return dic_vid_states # v, lane, pos, dis

    def record_rf_at_features(self, leader_id, features, c_ts):
        '''
        record features
        two Random Forest model
        RF_ArrivalTime (RF_AT) & RF_FollowerStates (RF_FS)
        Returns

        features: [leader_id, record_index, prediction_ts
                    platoon_type, dis_to_pv, speed_leader, remain_dis_leader, m (1 or 0)]
        & targets
        -------
        '''
        self.leader_record_counter[leader_id] += 1
        record_index = self.leader_record_counter[leader_id]
        record = [leader_id, record_index, c_ts] + features
        self.ls_features.append(record)
        return self.ls_features

    def record_tail_arrival(self, step):
        '''
        call self._record_rf_at_target
        Params:
            self.ls_tail_ids
        '''
        ls_vehid = self.dic_vid_groups['ls_vehid']
        ls_tail_ids_net = [vid for vid in self.ls_tail_ids if vid in ls_vehid]
        for tail_id in ls_tail_ids_net:
            self._record_rf_at_target(step, tail_id)

    def disable_all_lane_changes(self):
        '''
        Disable lane-changing behavior for all currently active vehicles.
        '''
        for vid in self.traci.vehicle.getIDList():
            self.traci.vehicle.setLaneChangeMode(vid, 0)

    def update_leader_gap_cache(self, dic_platoon_members):
        """
        Update leader-follower gap cache after platoon members are identified.
        Called after identify_platoon_members() in formation_controller.

        Pre-compute gaps to avoid repeated position queries.
        Params:
            - dic_platoon_members: {leader_id: [leader_id, follower1_id, follower
            - dic_pos: lane position
        """
        self.dic_leader_gap.clear()

        for leader_id, members in dic_platoon_members.items():
            if leader_id not in self.dic_pos:
                continue
            leader_pos = self.dic_pos[leader_id]

            for follower_id in members[1:]:  # Skip leader itself
                if follower_id in self.dic_pos:
                    follower_pos = self.dic_pos[follower_id]
                    self.dic_leader_gap[follower_id] = leader_pos - follower_pos
        return


    def _record_rf_at_target(self, step, tail_id):
        '''
        record platoon tail arrival time
        enters WS for the first time
        dic_tail_arrived_ws = {leader_id: [tail_id, arrival_time]}
        '''
        if tail_id == 'rhv540':
            pass
        if tail_id.startswith('m'):
            leader_id = self.get_hv_leader(tail_id)
        else:
            leader_id = self.get_hv_leader(tail_id, m=False)
        dic_vid_states = self.get_vid_states(tail_id)
        lane = dic_vid_states['lane']
        c_ts = round(step/10 + 0.1, 1) # getTime() = c_ts + 0.1
        if lane in ('ws_0', 'ws_1', 'ws_2') and leader_id not in self.dic_tail_arrived_ws:
            # record only the first time
            self.dic_tail_arrived_ws[leader_id] = [tail_id, c_ts]
        return self.dic_tail_arrived_ws

    def _build_step_cache(self, ls_vehid):
        """
        Cache per-step TraCI queries to reduce IPC overhead.
        All vehicle on traffic network.

        - Build lane-specific vehicle lists
        - Note: Leader gaps are computed separately via update_leader_gap_cache()
        """
        # Clear previous cache
        self.dic_speed.clear()
        self.dic_lane.clear()
        self.dic_pos.clear()
        self.dic_dis.clear() # distance to end of lane
        self.ls_lane0_veh.clear()
        self.ls_lane1_veh.clear()

        # Cache basic vehicle properties
        for vid in ls_vehid:
            # These TraCI calls are expensive; cache them once per step.
            lane_id = self.traci.vehicle.getLaneID(vid)
            pos = self.traci.vehicle.getLanePosition(vid)
            lane_length = self.dic_lane_length.get(lane_id)
            if lane_length is None:
                lane_length = self.traci.lane.getLength(lane_id)
                self.dic_lane_length[lane_id] = lane_length

            self.dic_speed[vid] = self.traci.vehicle.getSpeed(vid)
            self.dic_lane[vid] = lane_id
            self.dic_pos[vid] = pos
            self.dic_dis[vid] = lane_length - pos
            # Build lane-specific lists (Phase 1 optimization)
            if 'inflow_highway_0' in lane_id:
                self.ls_lane0_veh.append(vid)
            elif 'inflow_highway_1' in lane_id:
                self.ls_lane1_veh.append(vid)
        return

    def _get_vid_digit(self, vid):
        '''
        Return the trailing number from a vehicle ID string
        '''
        i = len(vid) - 1
        while i >= 0 and vid[i].isdigit():
            i -= 1
        return int(vid[i + 1:])




