#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
    def __init__(self, traci, instance_dr, optimizer=True):
        self.traci = traci
        self.amax = 2.6
        self.max_speed = 25  # ??? 23
        self.ls_v0 = []
        self.ls_teR = []
        self.optimizer = optimizer

        self.data_recorder = instance_dr # Data_Recording
        self.sim_step = self.data_recorder.sim_step
        self.dic_id_speed = self.data_recorder.dic_speed # new

        self.dic_platoon_info = {} # {id:[type, tail_id, length1, length2...]}

        self.dic_ravh_mavh = {} # the cor mavh of ravh
        self.dic_avh_action = {} # {avh_id:ls_action, ..., }
        self.dic_rm_c = {} # c: choosed to take action, {(ravh:mavh):ravh, ...}
        self.dic_mavh_scAction = {} # spacing creation action of mavh
        self.ls_avh_act = [] # avh need to take action
        self.ls_mavh_scAct = [] # mavh need to take Spacing control action

        self.dic_rplatoon_et = {}  # ramp platoon estimate time// {leader_id : [platoon_type, ts_head, ts_tail, update_time]}
        self.dic_mplatoon_et = {}  # mainline platoon estimate time//

        # self.mp_model = joblib.load("/home/zzha/PycharmProjects/RampMerging2/Models/m_arrival_prediction_model.pkl") # mainline prediction model
        # self.mp_model_old = joblib.load("/home/zzha/PycharmProjects/RampMerging3/Models/mr_arrival_prediction_model241122.pkl")
        # self.mp_model2 = joblib.load("/home/zzha/PycharmProjects/RampMerging3/Models/mr_arrival_prediction_model241128.pkl")

        # self.mp_model2 = joblib.load(
        #     os.path.join(project_root, 'rf_models', 'mr_arrival_prediction_model241128.pkl'))

        self.mp_model2 = joblib.load(
            os.path.join(project_root, 'rf_models', 'mr_arrival_prediction_model241128_ndarray.pkl'))

    def update_platoon_et(self, step, ls_leader_up, m=True, interval=70):
        '''
        update the estimate arrival time of platoon leader and tail
        ls_leader_up: list of pre merging (upstream) ramp av leader or mainlane av leader
        :return:
        '''
        if step % interval == 0:
            c_ts = self.traci.simulation.getTime()
            for id in ls_leader_up:
                if id == 'mav4690':
                    pass
                # platoon_type = self.data_recorder.dic_avhid_ptype[id]
                platoon_type = self.data_recorder.dic_avhid_ptype.get(id, "A")
                if platoon_type is None:
                    continue  # pass
                dic_id_info = self.get_veh_info(id)
                dis_head = dic_id_info['dis']
                speed_head = dic_id_info['v']
                # ts_head, the reaching time of head veh
                ts_head = self.get_ts_a(speed_head, dis_head) + c_ts
                if platoon_type == 'A':
                    # updated 241207, consider in platoon_type = 'A'
                    ts_tail = ts_head
                else:
                    # ts_tail, obtain through prediction model
                    features = self._get_features2(id)  # 5 features
                    ls_ft = list(features)
                    # [platoon_type, lead_dis, speed_head, dis_head, m]
                    indices = [3, 2, 4, 0, 7]  # the indices of selected features
                    selected_ft = [ls_ft[i] for i in indices]
                    selected_ft_c = selected_ft.copy()
                    selected_ft_c[0] = len(selected_ft[0])  # organize type 'AH' to length 2
                    selected_ft_cc = [600 if x is None else x for x in selected_ft_c]
                    # df_selected_ft_cc = pd.DataFrame([selected_ft_cc],
                    #                                  columns=['platoon_type', 'dis_leader_pv', 'leader_v', 'leader_r_dis',
                    #                                           'm'])
                    ts_tail0 = self.mp_model2.predict([selected_ft_cc])[0]
                    ts_tail = ts_tail0 + c_ts
                if m:
                    self.dic_mplatoon_et[id] = [platoon_type, ts_head, ts_tail, c_ts]
                else:
                    self.dic_rplatoon_et[id] = [platoon_type, ts_head, ts_tail, c_ts]
        return self.dic_mplatoon_et if m else self.dic_rplatoon_et

    def find_ravh(self, ls_r_veh_net_asc, ls_r_veh_net_last_asc):
        '''
        240614:find_ravh instead of find rav
        Get new emerge ravh id
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
            ravh_id = ls_r_veh_net_asc[-1]
        else:
            ravh_id = None
        return ravh_id

    def get_veh_info(self, id):
        dic_idinfo = {}
        ls_veh = self.traci.vehicle.getIDList()
        # v = self.traci.vehicle.getSpeed(id)
        if id == None or id not in ls_veh:
            dic_idinfo['v'] = None
            dic_idinfo['lane'] = None
            dic_idinfo['pos'] = None
            dic_idinfo['dis'] = None
            return dic_idinfo
        v = self.dic_id_speed[id]
        pos = self.traci.vehicle.getLanePosition(id)
        lane_id = self.traci.vehicle.getLaneID(id)
        lane_length = self.traci.lane.getLength(lane_id)

        dic_idinfo['v'] = v
        dic_idinfo['lane'] = lane_id
        dic_idinfo['pos'] = pos
        dic_idinfo['dis'] = lane_length-pos
        return dic_idinfo # v, lane, pos, dis

    def get_platoon_info2(self, m_dpt_type={}, r_dpt_type={}):
        """
        IMPORTANT: recording platoon information
        240929update: fixed length of platoon info
        240622update: add tail_id
        platoon type, platoon length, head reaching time, tail reaching time, head id, tail id

        :param m_dpt_type: scripts lane veh departure schedule; {4: 'AHHHHHHHHH', 31: 'AHHHHHHHHHHH'}
        :param r_dpt_type: ramp lane veh departure schedule; {4: 'AHHHHHHHHH', 58: 'AHHHHHHHHHHH'}
        :return: dic_platoon_info: {id:[type, tail_id, length1, length2...]}
                {'mavh70': [['AHH', 'mhv90'], deque([125.20400390170198, 125.383892989], maxlen=10)]}
        """
        # 1. get all avh at this moment
        dic_vehinfo = self.data_recorder.dic_vehinfo
        ls_m_leader_up_asc = dic_vehinfo['ls_m_leader_up_asc']
        ls_r_leader_up = dic_vehinfo['ls_r_leader_up']
        ls_mr_leader_up = ls_m_leader_up_asc + ls_r_leader_up
        for id in ls_mr_leader_up:
            dp_step = ''.join(filter(str.isdigit, id))  # departure time
            if id == 'mav11232':
                pass
            platoon_type = self.data_recorder.dic_avhid_ptype.get(id)
            if platoon_type is None:
                continue  # jump

            # get platoon type
            # dpt is step, should transfer to time, int(step)/10
            # dpt = int(int(dp_step)/10)
            # platoon_type = m_dpt_type[dpt] if 'm' in id else r_dpt_type[dpt]

            veh_num = len(platoon_type)
            # special situation, type 'A', no tail vehicle, no platoon length
            if veh_num == 1:
                self.dic_platoon_info[id] = [[platoon_type, None], None]
            # get tail id
            tail_dp_step = int(dp_step) + (veh_num-1)*10 # departure step
            tail_id_suffix = 'hv'+ str(tail_dp_step) if platoon_type[-1] == 'H' else 'av' + str(tail_dp_step)
            tail_id = 'm'+tail_id_suffix if 'm' in id else 'r'+tail_id_suffix
            if tail_id in dic_vehinfo['ls_vehid']:
                dic_head_info = self.get_veh_info(id)
                dic_tail_info = self.get_veh_info(tail_id)
                pos_head = dic_head_info['pos']
                pos_tail = dic_tail_info['pos']
                platoon_length = pos_head-pos_tail
                # record information
                ls_info_partA = [platoon_type, tail_id]
                if id not in self.dic_platoon_info:
                    dq_info_partB = deque(maxlen=10)  # fixed length is 10
                    self.dic_platoon_info[id] = [ls_info_partA, dq_info_partB]
                self.dic_platoon_info[id][1].append(platoon_length)
        return self.dic_platoon_info

    def find_ravh_mavh(self):
        """
        Params:
            self.dic_mplatoon_et: from where
            self.dic_rplatoon_et:
            self.dic_mavh_info: {id, [type, head_time, tail_time]}
            self.dic_ravh_info:
        Return:
            self.dic_ravh_mavh
        """
        c_ts = self.traci.simulation.getTime()

        dic_vehinfo = self.data_recorder.dic_vehinfo
        ls_r_leader_up = dic_vehinfo['ls_r_leader_up']

        # rav/mav information Before merging
        dic_mavhb_info = {k: v for k, v in self.dic_mplatoon_et.items() if v[2] > c_ts}
        dic_ravhb_info = {key: self.dic_rplatoon_et[key] for key in ls_r_leader_up if key in self.dic_rplatoon_et}
        dic_id_tailt = {key: value[2] for key, value in dic_mavhb_info.items()} # tailt => tail_time

        for r_avhid in dic_ravhb_info.keys():
            head_t = dic_ravhb_info[r_avhid][1]
            # sorted_mavh_tailt = sorted(dic_id_tailt.items(), key=lambda x: x[1]) # min => max
            sorted_mavh_tailt = sorted(dic_id_tailt.items(), key=lambda x: int(x[0][4:])) # min => max
            for index in range(len(sorted_mavh_tailt)):
                id, tail_t = sorted_mavh_tailt[index]
                if index == 0 and head_t < tail_t:
                    mavh_id = id
                    self.dic_ravh_mavh[r_avhid] = mavh_id
                    # prc.print_message(f'r_avhid:{r_avhid}, mavh_id:{mavh_id}')
                    break
                elif index > 0 and sorted_mavh_tailt[index - 1][1] <= head_t < tail_t:
                    # head_t is between tail_t(index-1) and tail_t(index)
                    mavh_id = id
                    self.dic_ravh_mavh[r_avhid] = mavh_id
                    # prc.print_message(f'r_avhid:{r_avhid}, mavh_id:{mavh_id}')
                    break
                else:
                    mavh_id = None
                    self.dic_ravh_mavh[r_avhid] = mavh_id
                    # prc.print_message(f'r_avhid:{r_avhid}, mavh_id:{mavh_id}')

        # sorted dic_ravh_mavh, key in increase sequence
        self.dic_ravh_mavh = {k: v for k, v in sorted(self.dic_ravh_mavh.items(), key=lambda item: int(item[0][4:]))}
        # once Mfleet or Rfleet completed pass through the intersection, delete k:v pair
        self._update_dic_rm()
        # compare delay_loss of RAVH or MAVH, to decided which one taking action
        self.dic_rm_c = self._compare_delay_loss()
        return self.dic_ravh_mavh, self.dic_rm_c

    def get_action_params(self, t, dis, v0):
        '''

        Parameters
        ----------
        t : TYPE
            time require.
        dis : TYPE
            length require.
        v0 : TYPE
            current speed.

        Returns
        -------
        ls : list
            the acc/dec strategy.
            (t1, a1, t3, a3, v_rem) or (T, a) v_rem => velocity of reach moment
        '''
        fomula = '2*v0*t+2*a*t*t1-a*t1**2-2*self.dis'
        t1 = (self.max_speed - v0) / self.amax  # duration to accelerate to peak velocity
        x1 = v0 * t1 + 0.5 * self.amax * t1 ** 2  # distance for speed increase to max_v

        t2 = t - t1
        x2 = t2 * self.max_speed
        xx = x1 + x2  # farthest dis could run in t

        # give 1 more seconds as preservation
        if xx >= dis:  # TODO: this part should be replaced
            # t = t_r+1
            # WS: weaving section
            prc.print_message(f"S1: avh will arrive WS in {t} (eg: r_platoon will encounter m_platoon),\n "
                              f"max_travel_dis_in_t {xx} >= current_dis_to_WS {dis}")
            if self.optimizer:
                # optm = GetBVCurve(v0, t, dis=dis) # ??? notice dis
                # new add min speed
                optm = GetBVCurve2(v0, t, dis=dis, min_speed=5)
                # v_rem: velocity of reach moment
                r, v_rem = optm.optimize()  # r.x = (t1, a1, t3, a3)
                # r.x[3] < 0.01 updated 110824, to avoid stop at the end of ramp caused by conflict
                if v_rem < 1 or r.x[3] < 0.01: # to avoid sudden stop; 241003update, avoid stop can start
                    prc.print_message("**Too big sacrifice, can't avoid encounter**")
                    return [None, self.amax]
                r1 = np.append(r.x, v_rem)  # r1 = (t1, a1, t3, a3, v_rem)
                return r1
            else:
                self.ls_v0.append(v0)
                self.ls_teR.append(t)
                optimal_a, optimal_vt, optimal_t1 = self.calculate_optimal_acceleration(v0, t, fomula)
                a = optimal_a
                vt = optimal_vt
                T = optimal_t1
                acc_dis = v0 * T + 0.5 * a * T * T
                cons_dis = vt * (t - T) # consistent speed distance
                ls = [T, a]
                return ls
            prc.print_message(f"action: apply_dec in {optimal_a} last {T}s, then keep vt {vt} for {t - T}s")
            prc.print_message(f'acc_dis {acc_dis}, cons_dis {avg_dis}')
        else:
            prc.print_message(f"S2: avh can't arrive WS in {t} (eg: no conflict), \n"
             f"max_travel_dis_in_t {xx} < current_dis_to_WS {dis}")
            a = self.amax
            T = None
            ls = [T, a]
            prc.print_message(f'action: apply_acc {a}')
            return ls

    def get_avh_action(self, gap=2):
        """
        241201update: origial is filter out mavh after merging, but should filter out action avh after merging as
        both ravh and mavh may take action.
        based on self.get_action_params()
        240625update: get_mavh_action => get_avh_action
        Note: there may have special situation, one value with multiple keys
        :param gap:
               dic_rm_drop:
               self.dic_rm_c: c => be chosen to take action, {(ravh:mavh):ravh, ...}
        :return: self.dic_avh_action = {avh_id:[ls_r, c_ts]}
        """

        c_ts = self.traci.simulation.getTime()
        dic_vehinfo = self.data_recorder.dic_vehinfo

        dic_rm_drop = {} # drop duplicate item, if two keys have same value, only keep the last key
        for r_id, m_id in reversed(self.dic_ravh_mavh.items()):
            if m_id not in dic_rm_drop.values():
                dic_rm_drop[r_id] = m_id
        dic_rm_drop = dict(reversed(dic_rm_drop.items())) # dic_rm_drop = {ravh_id: mavh_id}

        # mavh and ravh before merging
        ls_m_leader_up_asc = dic_vehinfo['ls_m_leader_up_asc']
        ls_r_leader_up_asc = dic_vehinfo['ls_r_leader_up_asc']

        ls_action_avh = self.dic_rm_c.values()
        # filter out action avh in control area
        ls_action_avh_filtered = [
            v for v in ls_action_avh
            if ('r' in v and v in ls_r_leader_up_asc) or ('m' in v and v in ls_m_leader_up_asc)
        ]
        dic_rm_c_filtered = {k: v for k, v in self.dic_rm_c.items() if v in ls_action_avh_filtered}

        for ravh, mavh in dic_rm_c_filtered.keys():
            action_av = dic_rm_c_filtered[(ravh, mavh)]
            if 'm' in action_av:
                # if mavh take action, then estimate r platoon
                r_ts_tail = self.dic_rplatoon_et[ravh][2] # ts_tail of ravh
                mavh_info = self.get_veh_info(mavh)
                dis_m = mavh_info['dis'] # remaining distance before the merging point
                v_m = mavh_info['v'] # current speed of mavh
                # 240624update: get tc according to platoon number
                platoon_type = self.data_recorder.dic_avhid_ptype[ravh]
                p_number = len(platoon_type)
                # tc = self.get_time_compensation(p_number, 'r') # 240625update:compensate in get_rav/mavinfo
                # find the best speed curve of mavh
                m_ts_head_new = r_ts_tail + gap
                ls_r = list(self.get_action_params(m_ts_head_new-c_ts, dis_m, v_m)) # acc strategy
                ls_r.append(c_ts) # ls_r = [t1, a1, t3, a3, v_rem]
                self.dic_avh_action[mavh] = ls_r
                self.ls_avh_act.append(mavh) # list of id need to action
                if mavh == 'mavh7900':
                    pass
                # 241210updated: if mavh in ls_avh_act, then should remove mavh from ls_mavh_scAct
                self.ls_mavh_scAct.remove(mavh) if mavh in self.ls_mavh_scAct else None
                self.dic_mavh_scAction.pop(mavh, None)

                # update head/tail reaching time information
                self._update_info(mavh, m_ts_head_new)
            else: # 'r' in action_av
                m_ts_tail = self.dic_mplatoon_et[mavh][2]
                ravh_info = self.get_veh_info(ravh)
                dis_r = ravh_info['dis']
                v_r = ravh_info['v']
                # get tc according to platoon number
                platoon_type = self.data_recorder.dic_avhid_ptype[mavh]
                p_number = len(platoon_type)
                # tc = self.get_time_compensation(p_number, 'm')
                # find the best speed curve of mavh
                r_ts_head_new = m_ts_tail+gap
                ls_r = list(self.get_action_params(r_ts_head_new-c_ts, dis_r, v_r))  # acc strategy
                ls_r.append(c_ts)
                self.dic_avh_action[ravh] = ls_r
                self.ls_avh_act.append(ravh)
                # update head/tail reaching time information
                self._update_info(ravh, r_ts_head_new)
        self.ls_avh_act = sorted(self.ls_avh_act, key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=True)
        # drop duplicate
        self.ls_avh_act = list(dict.fromkeys(self.ls_avh_act))
        return (self.dic_avh_action, self.ls_avh_act)

    def get_mavh_scAction(self, gap=2):
        """
        sc => space creation
        when ravh taking action, cor mavh' follow mavh need to take action to create space for this ramp platoon
        :param gap:
               mavh_n: the next mavh of this mavh
               self.ls_mavh_scAct: mavh need to take Spacing control action
        :return:
        """
        dic_vehinfo = self.data_recorder.dic_vehinfo
        c_ts = self.traci.simulation.getTime()
        ls_m_leader_up_asc = dic_vehinfo['ls_m_leader_up_asc']
        ls_mr_leader_up = dic_vehinfo['ls_mr_leader_up']  # all av id that before merging
        ls_ravh_act = [id for id in self.ls_avh_act if 'r' in id]
        ls_ravhb_act = [id for id in ls_ravh_act if id in ls_mr_leader_up]
        for ravh in ls_ravhb_act:
            if ravh in self.dic_ravh_mavh.keys():
                mavh = self.dic_ravh_mavh[ravh] # corresponding mavh
            else:
                continue
            if mavh is None:
                continue
            mavh_n = None # the next mavh of this mavh
            for id in ls_m_leader_up_asc:
                if self._extract_number(id) > self._extract_number(mavh):
                    mavh_n = id
                    if mavh_n == 'mavh4670':
                        pass
                    self.ls_mavh_scAct.append(mavh_n)
                    break
            if mavh_n is None:
                continue
            # get new tail time of ravh
            r_ts_tail_new = self.dic_rplatoon_et[ravh][2]
            # get mavh_n info
            dis = self.get_veh_info(mavh_n)['dis']
            v = self.get_veh_info(mavh_n)['v']
            # through into optimiser
            ls_r = list(self.get_action_params(r_ts_tail_new-c_ts+gap, dis, v))
            ls_r.append(c_ts)
            self.dic_mavh_scAction[mavh_n] = ls_r
        self.ls_mavh_scAct = sorted(self.ls_mavh_scAct, key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=True)
        self.ls_mavh_scAct = list(dict.fromkeys(self.ls_mavh_scAct))
        return(self.dic_mavh_scAction, self.ls_mavh_scAct)

    # take action
    def apply_avh_action(self, dic_avh_action):
        c_ts = self.traci.simulation.getTime()
        dic_vehinfo = self.data_recorder.dic_vehinfo

        ls_mr_leader_up = dic_vehinfo['ls_mr_leader_up'] # all avid that before merging
        # dic_avh_ab = {k: v for k, v in self.dic_avh_action.items() if k in ls_mr_leader_up} # action_beforeMerging
        dic_avh_ab = {k: v for k, v in dic_avh_action.items() if k in ls_mr_leader_up}  # action_beforeMerging

        # ls_ms_leader_up_asc = dic_vehinfo['ls_ms_leader_up_asc']
        # dic_mavh_ab = {k: v for k, v in self.dic_avh_action.items() if k in ls_ms_leader_up_asc}  # action_beforeMerging
        for avh, ls_action in dic_avh_ab.items():
            if avh == 'ravh4950':
                pass
            if len(ls_action) > 3:
                t1 = ls_action[0]
                a1 = ls_action[1]
                t3 = ls_action[2]
                a3 = ls_action[3]
                if c_ts < t1 + ls_action[-1]: # ls_action[-1] => the action start time
                    dec_st = ls_action[-1] # dec start time
                    if c_ts % 1 == 0:
                        prc.print_message(f"{avh} in dec_phase!\n a1:{a1}, start_time:{dec_st}, current_time:{c_ts}")
                    if a1 < 1:
                        pass
                    self._apply_acceleration(avh, a1, smooth=True)
                else:
                    acc_st = t1 + ls_action[-1] # acc started time
                    if c_ts % 1 == 0:
                        prc.print_message(f"{avh} in acc_phase!\n a3:{a3}, start_time:{acc_st}, current_time:{c_ts}") #\n => line break
                    self._apply_acceleration(avh, a3, smooth=True)
            else:
                acc2 = 2.6
                if c_ts % 1 == 0:
                    prc.print_message(f"{avh} full speed up!\n acc:{acc2}, current_time:{c_ts}")
                self._apply_acceleration(avh, acc2, smooth=True)

    def flashing_lc(self, step, dic_insertedAV, ls_upB):
        if not dic_insertedAV or not ls_upB:
            return
        ls_insertedAV_his = dic_insertedAV.keys()
        ls_insertedAV = [k for k in ls_insertedAV_his if k in ls_upB] # current lane change av
        for id in ls_insertedAV:
            self._flashing_base(step, 'upstream_1', id)

    def flashing2(self, step, ls_id): # ls_id: veh under control
        for veh_id in ls_id:
            # the order of list is important, if veh_id already not on-spot blow sentence will throw error
            if self.traci.vehicle.getLaneID(veh_id) != 'center_0':
                if step % 2 == 0:  # every 2 steps change color once
                    self.traci.vehicle.setColor(veh_id, (255, 255, 0, 255))  # set yellow
                else:
                    self.traci.vehicle.setColor(veh_id, (255, 0, 0, 255))  # set red
            else:
                self.traci.vehicle.setColor(veh_id, (255, 0, 0, 255))

    def get_ts_a(self, v0, D):
        """
        estimate travel time according to dis and current speed
        :param v0: current speed
        :param D: distance
        :return: travel time
        """
        t_acc = (self.max_speed - v0)/self.amax # reach max_speed's spending time
        d_acc = v0*t_acc + 0.5*self.amax*t_acc**2 # reach max_speed's needing dis
        if D == d_acc:
            ts = t_acc
        elif D < d_acc:
            # a bug here already fixed: self.amax*d_acc => self.amax*D
            ts = (-v0 + math.sqrt(v0**2 + 2 * self.amax * D)) / self.amax
        else: # D > d_acc
            t_uni = (D-d_acc)/self.max_speed
            ts = t_uni+t_acc
        return ts

    def _apply_acceleration(self, veh_id, acc, smooth=False):
        if acc is not None:
            this_vel = self.traci.vehicle.getSpeed(veh_id)
            next_vel = max([this_vel + acc * self.sim_step, 0])
            if smooth:
                self.traci.vehicle.slowDown(veh_id, next_vel, 1e-3)  # 1e-3: 0.001
            else:
                self.traci.vehicle.setSpeed(veh_id, next_vel)

    def _flashing_base(self, step, lane_id, veh_id):  # ls_id: veh under control
        '''
        the foundation of flashing function
        :param step:
        :param lane_id: flashing only out of this lane
        :param veh_id:
        :return:
        '''
        # only flash on upstream_0
        if self.traci.vehicle.getLaneID(veh_id) == lane_id:
            if step % 2 == 0:  # every 2 steps change color once
                self.traci.vehicle.setColor(veh_id, (255, 255, 0, 255))  # set yellow
            else:
                self.traci.vehicle.setColor(veh_id, (255, 0, 0, 255))  # set red
        else:
            self.traci.vehicle.setColor(veh_id, (255, 255, 0, 255)) # default yellow for AV_leader

    def _get_features2(self, id):
        '''
        update from 4 features to 5 features (m or r)
        get features of the vehicle id
        :param id:
        :return:
        '''
        # 1. platoon type
        platoon_type = self.data_recorder.dic_avhid_ptype[id]
        # 2. distance between platoon_head and its leader
        p_veh_info = self.traci.vehicle.getLeader(id)  # p_veh_info = [p_id, dis]
        if p_veh_info is not None:
            leader_dis = p_veh_info[1]
        else:
            leader_dis = None
        # 3. velocity of lead AV
        dic_id_info = self.get_veh_info(id)
        speed_head = dic_id_info['v']
        # 4. remain dis of lead AV
        dis_head = dic_id_info['dis']
        # 5. speed of tail id
        if self.dic_platoon_info.get(id, None) is not None:
            tail_id = self.dic_platoon_info[id][0][1]
            if tail_id == None:
                pass
            dic_tail_info = self.get_veh_info(tail_id)
            speed_tail = dic_tail_info['v']
            # 6. dis of tail id
            dis_tail = dic_tail_info['dis']
        else:
            tail_id = None
            speed_tail = None
            dis_tail = None
        # 6. mainline platoon or ramp platoon
        if 'm' in id:
            m = 1 # True
        else:
            m = 0 # False
        return dis_head, dis_tail, leader_dis, platoon_type, speed_head, speed_tail, tail_id, m

    def _update_info(self, id, ts_head_new):
        """
        update dic_mavh_info/dic_ravh_info when take new action
        :param id:
        :param ts_head_new:
        :return:
        """
        platoon_type = self.dic_mplatoon_et[id][0] if 'm' in id else self.dic_rplatoon_et[id][0]
        ts_head = self.dic_mplatoon_et[id][1] if 'm' in id else self.dic_rplatoon_et[id][1]
        ts_tail = self.dic_mplatoon_et[id][2] if 'm' in id else self.dic_rplatoon_et[id][2]
        dev = ts_head_new - ts_head
        ts_tail_new = ts_tail + dev
        if 'm' in id:
            self.dic_mplatoon_et[id] = [platoon_type, ts_head_new, ts_tail_new]
        else:
            self.dic_rplatoon_et[id] = [platoon_type, ts_head_new, ts_tail_new]

    def _update_dic_rm(self): # update dic_ravh_mavh
        # ramp veh list before merging
        dic_vehinfo = self.data_recorder.dic_vehinfo
        ls_vehid = dic_vehinfo['ls_vehid']
        ls_m_veh_up = dic_vehinfo['ls_m_veh_up']  # mavh list before merging
        ls_r_veh_up = dic_vehinfo['ls_r_veh_up']
        # scripts veh list before merging
        keys_to_remove = []
        for ravh, mavh in self.dic_ravh_mavh.items():
            # check ravh fleet state
            if ravh in self.dic_platoon_info and mavh in self.dic_platoon_info:
                r_tail_id = self.dic_platoon_info[ravh][1]
                m_tail_id = self.dic_platoon_info[mavh][1]
                if r_tail_id in ls_vehid and m_tail_id in ls_vehid:
                    if r_tail_id not in ls_r_veh_up or m_tail_id not in ls_m_veh_up:
                        keys_to_remove.append(ravh)
            # check mavh fleet state
        for key in keys_to_remove:
            del self.dic_ravh_mavh[key]

    def _compare_delay_loss(self):
        for ravh, mavh in self.dic_ravh_mavh.items():
            if ravh is not None and mavh is not None:  # make sure ravh/mavh is not None
                if ravh == 'ravh890':
                    pass
                head_tr = self.dic_rplatoon_et[ravh][1]  # head reaching time of ravh
                tail_tr = self.dic_rplatoon_et[ravh][2]  # tail time
                head_tm = self.dic_mplatoon_et[mavh][1]
                tail_tm = self.dic_mplatoon_et[mavh][2]
                m_give = tail_tr - head_tm
                r_give = tail_tm - head_tr
                if m_give >= r_give:
                    # ravh take action
                    # dic_rm_c: ravh and mavh and the action choosed
                    self.dic_rm_c[(ravh, mavh)] = ravh
                    # 241211update: if ravh take action, corresponding mavh should not take any action, drop it from ls_mavh_scAct and dic_mavh_scAction
                    self.ls_mavh_scAct.remove(mavh) if mavh in self.ls_mavh_scAct else None
                    self.dic_mavh_scAction.pop(mavh, None)
                else:
                    # mavh take action
                    self.dic_rm_c[(ravh, mavh)] = mavh
        return self.dic_rm_c

    def _extract_number(self, s):
        """
        input id and get digital number (depature time)
        "mav110"
        :return: 110 int
        """
        return int(''.join(filter(str.isdigit, s)))




