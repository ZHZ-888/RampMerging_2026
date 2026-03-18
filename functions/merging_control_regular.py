"""
merging_control_regular.py
-*- coding: utf-8 -*-
Created on Sun Feb 18 23:58:36 2024

@author: zzha
"""

import math
import numpy as np
import joblib # model prediction
import os
import warnings
from collections import deque # fixed length list

from functions import print_control as prc  # the shared fuction of print control
from functions.optimisation_algo import GetBVCurve, GetBVCurve2  # Optimiser

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

warnings.filterwarnings("ignore", category=FutureWarning,
                        message="is_sparse is deprecated and will be removed in a future version")

fomula1 = '2*v0*t+2*a*t*t1-a*t1**2-2*D'  # self.D
fomula2 = 'v0*t + 0.5*a*(t-t1)**2 - D'  # 先匀后减

class MergingControlRegular:
    def __init__(self, traci, instance_dr, ml, optimizer=True):
        self.traci = traci
        self.data_recorder = instance_dr  # Data_Recording
        self.sim_step = self.data_recorder.sim_step
        self.dic_id_speed = self.data_recorder.dic_speed
        self.merge_control_length = self.data_recorder.merge_control_length

        self.amax = 2.6
        self.max_speed = self.data_recorder.max_speed  # ??? 23
        self.ls_v0 = []
        self.ls_teR = []
        self.optimizer = optimizer

        self.dic_platoon_info = {} # {vid:[type, tail_id, length1, length2...]}
        self.dic_rm_leader_map = {} # the cor mavh of ravh
        self.dic_leader_action = {} # {leader: ls_action, ..., }
        self.dic_rm_leader_actor = {} # actor: selected r_leader or m_leader to take action, {(r_leader: m_leader): r_leader, ...}
        self.dic_m_leader_followup_action = {} # spacing creation action of mavh
        self.ls_action_leader = [] # leader need to take action
        self.ls_m_leaders_followup = [] # m_leader need to take Spacing control action

        self.dic_rplatoon_et = {}  # ramp platoon estimate time// {leader_id : [platoon_type, ts_head, ts_tail, update_time]}
        self.dic_mplatoon_et = {}  # mainline platoon estimate time//

        # random forest arrival time prediction model
        if ml:
            self.rf_at_model = joblib.load(
                os.path.join(project_root, 'rf_models', 'mr_arrival_prediction_model260125_ndarray.pkl'))
        else:
            # model241128 is more accurate compare with model260125 in single lane
            self.rf_at_model = joblib.load(
                os.path.join(project_root, 'rf_models', 'mr_arrival_prediction_model241128_ndarray.pkl'))


    def update_platoon_et(self, step, ls_leader_up, m=True, interval=60):
        '''
        calls self._get_features()
              self.rf_at_model
        update the estimate arrival time of platoon leader and tail
        ls_leader_up: list of pre merging (upstream) ramp av leader or mainlane av leader
        :return:
            dic_mplatoon_et ={m_vid, [platoon_type, ts_head, ts_tail, c_ts]}
        '''
        if step % interval == 0:
            c_ts = step/10 + 0.1 # c_ts = self.traci.simulation.getTime()
            for leader in ls_leader_up:
                platoon_type = self.data_recorder.dic_leader_ptype.get(leader, "A")
                if platoon_type is None:
                    continue  # pass
                leader_info = self.data_recorder.get_vid_states(leader)
                remain_dis_leader = leader_info['dis'] # remaining dis of this leader
                speed_leader = leader_info['v'] # speed of this leader
                # ts_head, the reaching timestamp of leader
                ts_head = self.estimate_travel_time(speed_leader, remain_dis_leader) + c_ts
                if platoon_type == 'A':
                    # updated 241207, consider in platoon_type = 'A'
                    ts_tail = ts_head
                else:
                    # ts_tail, obtain through prediction model
                    features = self._get_features2(leader)  # 5 features
                    ls_features = list(features)
                    # [platoon_type, leader_to_pv_dis, speed_leader, remain_dis_leader, m]
                    indices = [3, 2, 4, 0, 7]  # the indices of selected features
                    sel_features = [ls_features[i] for i in indices] # selected features
                    sel_features_c = sel_features.copy()
                    self.data_recorder.record_rf_at_features(leader, sel_features, c_ts)
                    sel_features_c[0] = len(sel_features[0])  # organise type 'AH' to length 2
                    travel_time = self.rf_at_model.predict([sel_features_c])[0]
                    ts_tail = travel_time + c_ts
                if m:
                    self.dic_mplatoon_et[leader] = [platoon_type, ts_head, ts_tail, c_ts]
                else:
                    self.dic_rplatoon_et[leader] = [platoon_type, ts_head, ts_tail, c_ts]
        return self.dic_mplatoon_et if m else self.dic_rplatoon_et

    def find_r_leader(self, ls_r_veh_net_asc, ls_r_veh_net_last_asc):
        '''
        240614:find_r_leader (find_ravh) instead of find rav
        Get new emerge ramp leader
        Parameters
        ----------
        tup_bmrv : TYPE
            vehs on the ramp road (inflow_merge) at current moment.
        Returns
        -------
        r_havid : new emerge rav id
        '''
        if len(ls_r_veh_net_asc) > 0 and 'avh' in ls_r_veh_net_asc[-1] and ls_r_veh_net_asc[-1] \
                not in ls_r_veh_net_last_asc:
            r_leader = ls_r_veh_net_asc[-1]
        else:
            r_leader = None
        return r_leader

    def get_platoon_info2(self, step, m_dpt_type={}, r_dpt_type={}):
        """
        IMPORTANT: recording platoon information
        240929update: fixed length of platoon info
        240622update: add tail_id

        :param
                ls_r_leader_up:
                m_dpt_type: scripts lane veh departure schedule; {4: 'AHHHHHHHHH', 31: 'AHHHHHHHHHHH'}
                r_dpt_type: ramp lane veh departure schedule; {4: 'AHHHHHHHHH', 58: 'AHHHHHHHHHHH'}
        :return: dic_platoon_info:
                {'mavh70': [['AHH', 'mhv90'], deque([platoon_length1, platoon_length2], maxlen=10)]}
        """
        # get info at this moment
        dic_vid_groups = self.data_recorder.dic_vid_groups

        # ls_m_leader_up_asc = dic_vid_groups['ls_m_leader_up_asc']
        # ls_r_leader_up = dic_vid_groups['ls_r_leader_up']
        # ls_mr_leader_up = ls_m_leader_up_asc + ls_r_leader_up

        ls_m_leader_net = dic_vid_groups['ls_m_leader_net']
        ls_r_leader_net = dic_vid_groups['ls_r_leader_net']
        ls_mr_leader_net = ls_m_leader_net + ls_r_leader_net

        for leader in ls_mr_leader_net:
            if leader == 'ravh900':
                pass
            platoon_type = self.data_recorder.dic_leader_ptype.get(leader)
            if platoon_type is None:
                continue  # jump
            if leader not in self.dic_platoon_info:
                # Part A: [platoon_type, tail_id];
                # Part B: deque(maxlen=10) (fixed length is 10)
                self.dic_platoon_info[leader] = [[platoon_type, None], deque(maxlen=10)]
            veh_num = len(platoon_type)
            # SPECIAL CASE: type 'A', no tail vehicle, no platoon length
            if veh_num == 1:
                self.dic_platoon_info[leader][0] = [platoon_type, None]
                continue
            tail_id = self._get_tail_id(dic_vid_groups, platoon_type, leader)
            if tail_id in dic_vid_groups['ls_vehid']:
                # update Tail ID
                self.dic_platoon_info[leader][0] = [platoon_type, tail_id]
                if tail_id not in self.data_recorder.ls_tail_ids:
                    self.data_recorder.ls_tail_ids.append(tail_id) # ls_tail_ids (purpose?)
                dic_head_states = self.data_recorder.get_vid_states(leader)
                dic_tail_states = self.data_recorder.get_vid_states(tail_id)
                pos_head = dic_head_states['pos']
                pos_tail = dic_tail_states['pos']
                platoon_length = pos_head-pos_tail
                # record length
                self.dic_platoon_info[leader][1].append(platoon_length)
        self.data_recorder.dic_platoon_info = self.dic_platoon_info
        return self.dic_platoon_info

    def find_rm_leader_map(self):
        """
        Params:
            self.dic_mplatoon_et: {m_vid, [platoon_type, ts_head, ts_tail, c_ts]}
            self.dic_rplatoon_et:
            dic_mplatoon_et_valid: {id, [platoon_type, ts_head, ts_tail]}
            dic_rplatoon_et_valid:
            dic_mplatoon_tail_et: {m_vid: ts_tail}
        Return:
            self.dic_rm_leader_map
        """
        c_ts = self.traci.simulation.getTime()

        dic_vid_groups = self.data_recorder.dic_vid_groups
        ls_r_leader_up = dic_vid_groups['ls_r_leader_up']

        # rav/mav information Before merging # mavhb => m_leader_up
        dic_mplatoon_et_valid = {m_leader: ls_ts for m_leader, ls_ts in self.dic_mplatoon_et.items() if ls_ts[2] > c_ts}
        dic_rplatoon_et_valid = {r_leader: self.dic_rplatoon_et[r_leader] for r_leader in ls_r_leader_up if r_leader in self.dic_rplatoon_et}
        dic_mplatoon_tail_et = {m_leader: ls_ts[2] for m_leader, ls_ts in dic_mplatoon_et_valid.items()}

        for r_leader in dic_rplatoon_et_valid.keys():
            if r_leader == 'ravh700':
                pass
            r_ts_head = dic_rplatoon_et_valid[r_leader][1] # ramp platoon head timestamp
            dic_mplatoon_tail_et_asc = sorted(dic_mplatoon_tail_et.items(),
                                              key=lambda x: x[1])  # min => max by tail time
            for index in range(len(dic_mplatoon_tail_et_asc)):
                m_leader, m_ts_tail = dic_mplatoon_tail_et_asc[index]
                if index == 0 and r_ts_head < m_ts_tail:
                    self.dic_rm_leader_map[r_leader] = m_leader
                    # prc.print_message(f'r_leader:{r_leader}, m_leader:{m_leader}')
                    break
                elif index > 0 and dic_mplatoon_tail_et_asc[index - 1][1] <= r_ts_head < m_ts_tail:
                    # r_ts_head is between m_ts_tail(index-1) and m_ts_tail(index)
                    self.dic_rm_leader_map[r_leader] = m_leader
                    # prc.print_message(f'r_leader:{r_leader}, m_leader:{m_leader}')
                    break
                else:
                    m_leader = None
                    self.dic_rm_leader_map[r_leader] = m_leader
                    # prc.print_message(f'r_leader:{r_leader}, m_leader:{m_leader}')

        # sorted dic_rm_leader_map, r_leader in increase sequence
        self.dic_rm_leader_map = {k: v for k, v in sorted(self.dic_rm_leader_map.items(), key=lambda item: int(item[0][4:]))}
        # once Mfleet or Rfleet completed pass through the intersection, delete r_leader:m_leader pair
        self._update_dic_rm_leader_map()
        # compare delay_loss of R_leader or M_leader, to decided which one taking action
        self.dic_rm_leader_actor = self._compare_delay_loss()
        return self.dic_rm_leader_map, self.dic_rm_leader_actor

    def get_action_params(self, t, dis, v0):
        '''

        Parameters
        ----------
        t : TYPE
            time require.
        dis : TYPE
            the remaining distance to weaving section of this leader.
        v0 : TYPE
            current speed.

        Returns
        -------
        ls_acc_profile : list
            the acc/dec strategy.
            (t1, a1, t3, a3, v_rem) or (T, a) v_rem => velocity of reach moment
        '''
        fomula = '2*v0*t+2*a*t*t1-a*t1**2-2*self.dis'

        t1 = (self.max_speed - v0) / self.amax  # duration to accelerate to peak velocity
        x1 = v0 * t1 + 0.5 * self.amax * t1 ** 2  # distance for speed increase to max_v

        t2 = t - t1
        x2 = t2 * self.max_speed
        xx = x1 + x2  # farthest dis could run in t

        if xx >= dis:  # TODO: this part should be replaced
            prc.print_message(f"S1: avh will arrive WS in {t} (eg: r_platoon will encounter m_platoon),\n "
                              f"max_travel_dis_in_t {xx} >= current_dis_to_WS {dis}")
            if self.optimizer:
                # new add min speed
                optm = GetBVCurve2(v0, t, dis=dis, min_speed=5)
                # v_rem: velocity of reach moment
                r, v_rem = optm.optimize()  # r.x = (t1, a1, t3, a3)
                # r.x[3] < 0.01 updated 110824, to avoid stop at the end of ramp caused by conflict
                if v_rem < 1 or r.x[3] < 0.01: # to avoid sudden stop; 241003update, avoid stop can start
                    prc.print_message("**Too big sacrifice, can't avoid encounter**")
                    return [None, self.amax]
                ls_acc_profile = list(np.append(r.x, v_rem))  # ls_acc_profile = (t1, a1, t3, a3, v_rem)
                return ls_acc_profile
            else:
                self.ls_v0.append(v0)
                self.ls_teR.append(t)
                optimal_a, optimal_vt, optimal_t1 = self.calculate_optimal_acceleration(v0, t, fomula)
                a = optimal_a
                vt = optimal_vt
                T = optimal_t1
                acc_dis = v0 * T + 0.5 * a * T * T
                cons_dis = vt * (t - T) # consistent speed distance
                ls_acc_profile = [T, a]
                return ls_acc_profile
            prc.print_message(f"action: apply_dec in {optimal_a} last {T}s, then keep vt {vt} for {t - T}s")
            prc.print_message(f'acc_dis {acc_dis}, cons_dis {avg_dis}')
        else:
            prc.print_message(f"S2: leader can't arrive WS in {t} (eg: no conflict), \n"
             f"max_travel_dis_in_t {xx} < current_dis_to_WS {dis}")
            ls_acc_profile = [0, self.amax]
            prc.print_message(f'action: apply_acc {self.amax}')
            return ls_acc_profile

    def get_leader_action(self, gap=2):
        """
        calls 'self.get_action_params()'
        both R_LEADER and M_LEADER may take action.
        Note: there may have special situation, one value with multiple keys
        :param
               gap:
               self.dic_rm_leader_actor: # actor: selected r_leader or m_leader to take action,
                                         # {(r_leader: m_leader): actor, ...}
               dic_rm_leader_actor_filtered
               self.dic_rm_leader_map

        :return: self.dic_leader_action = {leader: [ls_r, c_ts]}
        """
        c_ts = self.traci.simulation.getTime()

        # m_leader and r_leader before merging
        dic_vid_groups = self.data_recorder.dic_vid_groups
        ls_m_leader_up_asc = dic_vid_groups['ls_m_leader_up_asc']
        ls_r_leader_up_asc = dic_vid_groups['ls_r_leader_up_asc']

        ls_action_leader = self.dic_rm_leader_actor.values()
        # only keep action leaders in control area
        ls_action_leader_filtered = [
            v for v in ls_action_leader
            if ('r' in v and v in ls_r_leader_up_asc) or ('m' in v and v in ls_m_leader_up_asc)]
        dic_rm_leader_actor_filtered = {k: v for k, v in self.dic_rm_leader_actor.items()
                                        if v in ls_action_leader_filtered}

        for (r_leader, m_leader), action_leader in dic_rm_leader_actor_filtered.items():
            if 'm' in action_leader:
                ls_acc_profile = self._get_action_params_upper_level(c_ts, r_leader, m_leader, action_leader, gap)
                if ls_acc_profile and ls_acc_profile[0] is None and (r_leader in ls_r_leader_up_asc):
                    prc.print_message(f"m_leader {m_leader} action failed, trying R_leader {r_leader} instead")
                    # Fallback to r_leader
                    action_leader = r_leader
                    ls_acc_profile = self._get_action_params_upper_level(c_ts, r_leader, m_leader, action_leader, gap)
                    if ls_acc_profile and ls_acc_profile[0] is not None:
                        self.ls_m_leaders_followup.append(m_leader)
            else: # 'r' in action_leader
                ls_acc_profile = self._get_action_params_upper_level(c_ts, r_leader, m_leader, action_leader, gap)

        self.ls_action_leader = sorted(self.ls_action_leader, key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=True)
        # drop duplicate
        self.ls_action_leader = list(dict.fromkeys(self.ls_action_leader))
        return (self.dic_leader_action, self.ls_action_leader)

    def _get_action_params_upper_level(self, c_ts, r_leader, m_leader, action_leader, gap):
        # action leader info
        action_leader_info = self.data_recorder.get_vid_states(action_leader)
        dis_action_leader = action_leader_info['dis']
        if dis_action_leader is None:
            pass
        v_action_leader = action_leader_info['v']
        # get adjusted time
        if 'r' in action_leader:
            ts_tail = self.dic_mplatoon_et[m_leader][2]
        else:
            ts_tail = self.dic_rplatoon_et[r_leader][2]
            # 241210updated: if m_leader in ls_action_leader, then should remove m_leader from ls_m_leaders_followup
            self.ls_m_leaders_followup.remove(m_leader) if m_leader in self.ls_m_leaders_followup else None
            self.dic_m_leader_followup_action.pop(m_leader, None)
        ts_head_adj = ts_tail + gap
        ls_acc_profile = self.get_action_params(ts_head_adj - c_ts, dis_action_leader, v_action_leader)  # acc strategy
        if ls_acc_profile and len(ls_acc_profile) > 2: # only take action then update info
            # update head/tail reaching time information
            self._update_info(action_leader, ts_head_adj)
        ls_acc_profile.append(c_ts)  # ls_acc_profile  = [t1, a1, t3, a3, c_ts]
        self.dic_leader_action[action_leader] = ls_acc_profile
        self.ls_action_leader.append(action_leader)  # list of vid need to action
        return ls_acc_profile

    def get_m_leader_followup_action(self, gap=2):
        """
        when r_leader taking action, m_leader after cor m_leader may needs to take action to create space
        for this act ramp platoon
        :param gap:
               m_leader_next: the next m_leader of this m_leader
               self.ls_m_leaders_followup: m_leader need to take followup action for ramp action
        :return:
        """
        dic_vid_groups = self.data_recorder.dic_vid_groups
        c_ts = self.traci.simulation.getTime()
        ls_m_leader_up_asc = dic_vid_groups['ls_m_leader_up_asc']
        ls_mr_leader_up = dic_vid_groups['ls_mr_leader_up']  # all av id that before merging
        ls_action_r_leader = [vid for vid in self.ls_action_leader if 'r' in vid]
        ls_action_r_leader_up = [vid for vid in ls_action_r_leader if vid in ls_mr_leader_up]
        for r_leader in ls_action_r_leader_up:
            m_leader = self.dic_rm_leader_map.get(r_leader)
            if m_leader is None:
                continue
            m_leader_next = None # the next m_leader of this m_leader
            for vid in ls_m_leader_up_asc:
                if self._extract_number(vid) > self._extract_number(m_leader):
                    m_leader_next = vid
                    self.ls_m_leaders_followup.append(m_leader_next)
                    break
            if m_leader_next is None:
                continue
            # get new tail time of r_leader
            r_ts_tail = self.dic_rplatoon_et[r_leader][2]
            # get m_leader_next info
            dis = self.data_recorder.get_vid_states(m_leader_next)['dis']
            v = self.data_recorder.get_vid_states(m_leader_next)['v']
            # through into optimiser
            ls_acc_profile = self.get_action_params(r_ts_tail-c_ts+gap, dis, v)
            ls_acc_profile.append(c_ts)
            self.dic_m_leader_followup_action[m_leader_next] = ls_acc_profile
        self.ls_m_leaders_followup = sorted(self.ls_m_leaders_followup, key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=True)
        self.ls_m_leaders_followup = list(dict.fromkeys(self.ls_m_leaders_followup)) # remove duplicates
        return(self.dic_m_leader_followup_action, self.ls_m_leaders_followup)

    def apply_leader_action(self, dic_leader_action):
        c_ts = self.traci.simulation.getTime()
        dic_vid_groups = self.data_recorder.dic_vid_groups

        ls_mr_leader_up = dic_vid_groups['ls_mr_leader_up'] # all avid that before merging
        dic_leader_up_action = {leader: v for leader, v in dic_leader_action.items() if leader in ls_mr_leader_up}  # action_before Merging

        for leader, ls_action in dic_leader_up_action.items():
            if len(ls_action) > 3:
                t1 = ls_action[0]
                a1 = ls_action[1]
                t3 = ls_action[2]
                a3 = ls_action[3]
                if c_ts < t1 + ls_action[-1]: # ls_action[-1] => the action start time
                    dec_st = ls_action[-1] # dec start time
                    if c_ts % 1 == 0:
                        prc.print_message(f"{leader} in dec_phase!\n a1:{a1}, start_time:{dec_st}, current_time:{c_ts}")
                    if a1 < 1:
                        pass
                    self._apply_acceleration(leader, a1, smooth=True)
                else:
                    acc_st = t1 + ls_action[-1] # acc started time
                    if c_ts % 1 == 0:
                        prc.print_message(f"{leader} in acc_phase!\n a3:{a3}, start_time:{acc_st}, current_time:{c_ts}") #\n => line break
                    self._apply_acceleration(leader, a3, smooth=True)
            else:
                acc2 = 2.6
                if c_ts % 1 == 0:
                    prc.print_message(f"{leader} full speed up!\n acc:{acc2}, current_time:{c_ts}")
                self._apply_acceleration(leader, acc2, smooth=True)

    def flashing_lane_changing(self, step, dic_insertedAV, ls_vid):
        '''
        call self._flashing_base()
        lane changing action flashing (jump from inflow_highway_1//B to inflow_highway_0//A)
        flashing only on inflow_highway_1
        '''
        flashing_lane_id = 'inflow_highway_1'
        if not dic_insertedAV or not ls_vid:
            return
        ls_insertedAV_his = dic_insertedAV.keys()
        ls_insertedAV = [k for k in ls_insertedAV_his if k in ls_vid] # current lane change av
        for vid in ls_insertedAV:
            self._flashing_base(step, flashing_lane_id, vid)

    def flashing_merging(self, step, ls_id): # ls_id: veh under control
        '''
        call self._flashing_base()
        merging control action;
        flashing only on inflow_highway_0 and ramp (inflow_merge_0)
        '''
        mainlane = 'inflow_highway_0'
        ramp = 'inflow_merge_0'
        for vid in ls_id:
            # the order of list is important, if vid already not on-spot blow sentence will throw error
            if vid.startswith('r'):
                self._flashing_base(step, ramp, vid)
            else:
                self._flashing_base(step, mainlane, vid)

    def estimate_travel_time(self, v0, D): # get_ts_a
        """
        estimate travel time according to remaining distance and current speed
        :param v0: current speed
        :param D: distance
        :return: travel time
        """
        t_acc = (self.max_speed - v0)/self.amax # reach max_speed's spending time
        d_acc = v0*t_acc + 0.5*self.amax*t_acc**2 # reach max_speed's needing dis
        if D == d_acc:
            travel_time = t_acc
        elif D < d_acc:
            # a bug here already fixed: self.amax*d_acc => self.amax*D
            travel_time = (-v0 + math.sqrt(v0**2 + 2 * self.amax * D)) / self.amax
        else: # D > d_acc
            t_uni = (D-d_acc)/self.max_speed
            travel_time = t_uni+t_acc
        return travel_time

    def _get_tail_id(self, dic_vid_groups, platoon_type, leader_id):
        '''
        get platoon tail id
        Params:
            ls_m_veh_up:
            ls_r_veh_up: desc

            ls_r_veh_net: should be desc
        '''
        if leader_id.startswith('m'):
            ls_veh_net = dic_vid_groups['ls_m_veh_net']
        else:
            ls_veh_net = dic_vid_groups['ls_r_veh_net']
        idx_leader = ls_veh_net.index(leader_id)
        dev_idx = len(platoon_type) - 1
        idx_tail = idx_leader - dev_idx
        tail_id = ls_veh_net[idx_tail] if idx_tail >= 0 else None
        return tail_id

    def _apply_acceleration(self, vid, acc, smooth=False):
        if acc is not None:
            this_vel = self.traci.vehicle.getSpeed(vid)
            next_vel = max([this_vel + acc * self.sim_step, 0])
            if smooth:
                self.traci.vehicle.slowDown(vid, next_vel, 1e-3)  # 1e-3: 0.001
            else:
                self.traci.vehicle.setSpeed(vid, next_vel)

    def _flashing_base(self, step, lane_id, vid):  # ls_id: veh under control
        '''
        the foundation of flashing function
        :param step:
        :param lane_id: flashing only on this lane
        :param vid:
        :return:
        '''
        if self.traci.vehicle.getLaneID(vid) == lane_id:
            if step % 2 == 0:  # every 2 steps change color once
                self.traci.vehicle.setColor(vid, (255, 255, 0, 255))  # set yellow
            else:
                self.traci.vehicle.setColor(vid, (255, 0, 0, 255))  # set red
        else:
            self.traci.vehicle.setColor(vid, (255, 255, 0, 255)) # default yellow for AV_leader

    def _get_features2(self, leader):
        '''
        update from 4 features to 5 features (m or r)
        get features of the leader
        Features:
            platoon_type
            leader_to_pv_dis (dis_to_pv)
            leader_left_dis (remain_dis_leader)

        :param leader:
               dic_platoon_info: {head_id:[['AHHH', tail_id], [length1, length2, ...]]}
        :return:
        '''
        # 1. platoon type
        platoon_type = self.data_recorder.dic_leader_ptype[leader]
        # 2. distance between this leader and its previous vehicle
        p_veh_info = self.traci.vehicle.getLeader(leader, self.merge_control_length)  # p_veh_info = [p_id, dis]
        if p_veh_info is not None:
            leader_to_pv_dis = p_veh_info[1] # dis to pv (leader dis to pv)
        else:
            leader_to_pv_dis = self.merge_control_length # equal to the length of merging control section
        # 3. speed of platoon leader AV
        dic_head_states = self.data_recorder.get_vid_states(leader)
        speed_leader = dic_head_states['v'] # this_speed
        # 4. remain dis of platoon leader AV
        remain_dis_leader = dic_head_states['dis'] # this_remain_dis
        # 5. speed of tail vid
        if self.dic_platoon_info.get(leader, None) is not None:
            tail_id = self.dic_platoon_info[leader][0][1]
            if tail_id == None:
                pass
            dic_tail_states = self.data_recorder.get_vid_states(tail_id)
            speed_tail = dic_tail_states['v']
        # 6. dis of tail vid
            remaining_dis_tail = dic_tail_states['dis']
        else:
            tail_id = None
            speed_tail = None
            remaining_dis_tail = None
        # 7. mainline platoon or ramp platoon
        if 'm' in leader:
            m = 1 # True
        else:
            m = 0 # False
        return remain_dis_leader, remaining_dis_tail, leader_to_pv_dis, platoon_type, speed_leader, speed_tail, tail_id, m

    def _update_info(self, vid, ts_head_adj):
        """
        update dic_mavh_info/dic_ravh_info when take new action
        :param vid:
        :param ts_head_adj:
        :return:
        """
        if vid == 'mav239':
            pass
        platoon_type = self.dic_mplatoon_et[vid][0] if 'm' in vid else self.dic_rplatoon_et[vid][0]
        ts_head = self.dic_mplatoon_et[vid][1] if 'm' in vid else self.dic_rplatoon_et[vid][1]
        ts_tail = self.dic_mplatoon_et[vid][2] if 'm' in vid else self.dic_rplatoon_et[vid][2]
        dev = ts_head_adj - ts_head
        ts_tail_adj = ts_tail + dev
        if 'm' in vid:
            self.dic_mplatoon_et[vid] = [platoon_type, ts_head_adj, ts_tail_adj]
        else:
            self.dic_rplatoon_et[vid] = [platoon_type, ts_head_adj, ts_tail_adj]

    def _update_dic_rm_leader_map(self): # update dic_rm_leader_map
        # ramp veh list before merging
        dic_vid_groups = self.data_recorder.dic_vid_groups
        ls_vehid = dic_vid_groups['ls_vehid']
        ls_m_veh_up = dic_vid_groups['ls_m_veh_up']  # m_leader list before merging
        ls_r_veh_up = dic_vid_groups['ls_r_veh_up']
        # scripts veh list before merging
        keys_to_remove = []
        for r_leader, m_leader in self.dic_rm_leader_map.items():
            # check r_leader fleet state
            if r_leader in self.dic_platoon_info and m_leader in self.dic_platoon_info:
                # r_tail_id = self.dic_platoon_info[r_leader][1]
                # m_tail_id = self.dic_platoon_info[m_leader][1]
                r_tail_id = self.dic_platoon_info[r_leader][0][1]
                m_tail_id = self.dic_platoon_info[m_leader][0][1]
                if r_tail_id in ls_vehid and m_tail_id in ls_vehid:
                    if r_tail_id not in ls_r_veh_up or m_tail_id not in ls_m_veh_up:
                        keys_to_remove.append(r_leader)
            # check m_leader fleet state
        for key in keys_to_remove:
            del self.dic_rm_leader_map[key]

    def _compare_delay_loss(self):
        for r_leader, m_leader in self.dic_rm_leader_map.items():
            if r_leader is not None and m_leader is not None:  # make sure r_leader/m_leader is not None
                if r_leader == 'ravh2410':
                    pass
                ts_rp_head = self.dic_rplatoon_et[r_leader][1] # ramp platoon leader (head) estimate arrival timestamp (head_tr)
                ts_rp_tail = self.dic_rplatoon_et[r_leader][2] # tail time (tail_tr)
                ts_mp_head= self.dic_mplatoon_et[m_leader][1] # head_tm
                ts_mp_tail = self.dic_mplatoon_et[m_leader][2] # tail_tm
                m_give = ts_rp_tail - ts_mp_head
                r_give = ts_mp_tail - ts_rp_head
                if m_give >= r_give:
                    # r_leader take action
                    # dic_rm_leader_actor: # {(r_leader: m_leader): r_leader, ...}, which one as actor
                    self.dic_rm_leader_actor[(r_leader, m_leader)] = r_leader
                    # 241211update: if r_leader take action, corresponding m_leader should not take any action, drop it from ls_m_leaders_followup and dic_m_leader_followup_action
                    self.ls_m_leaders_followup.remove(m_leader) if m_leader in self.ls_m_leaders_followup else None
                    self.dic_m_leader_followup_action.pop(m_leader, None)
                else:
                    # m_leader take action
                    self.dic_rm_leader_actor[(r_leader, m_leader)] = m_leader
        return self.dic_rm_leader_actor

    def _extract_number(self, s):
        """
        input vid and get digital number (depature time)
        "mav110"
        :return: 110 int
        """
        return int(''.join(filter(str.isdigit, s)))




