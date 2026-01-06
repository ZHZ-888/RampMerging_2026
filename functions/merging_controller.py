# merging_controller.py
"""
High-level controller:
Combine regular / jam mode logic and three control modules into one place.

This class does NOT change any algorithm or behavior.
It only wraps existing logic in a cleaner interface.
"""


from functions import print_control as prc
from functions import action_manager as act_mgr
from functions import merging_control_regular as mcr
from functions import merging_control_jam as mcj


class MergingController:
    def __init__(self, data_recorder, traci, av_p, platoon_formation=False, ml=False, loss_rate=0, mpc_interval=70,
                 delta_t=12): # ml: multi-lane
        self.data_recorder = data_recorder
        self.merge_regular = mcr.MergingControlRegular(traci, self.data_recorder)
        self.action_mgr = act_mgr.ActionManager(self.data_recorder, self.merge_regular, loss_rate)
        self.merge_jam = mcj.MergingControlJam(traci, self.data_recorder, self.merge_regular, loss_rate, ml) # vehicle control during_jame3
        self.mode_switch = mcj.ShiftMode(traci, self.data_recorder,av_p)
        self.mpc_interval = mpc_interval
        self.delta_t = delta_t
        self.ls_r_dep_times = []  # list of ramp veh depature time; [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 58, 59, 60]
        self.pf = platoon_formation
        self.speed_log = [] # [step, avg_speed, True/False], True/False Jam

    def step(self, st, step, m_dpt_type, r_dpt_type):
        '''

        :param st: total simulation time
        :param step: simulation step
        :param m_dpt_type: scripts lane veh departure schedule; {4: 'AHHHHHHHHH', 31: 'AHHHHHHHHHHH'}
        :param r_dpt_type: ramp lane veh departure schedule; {4: 'AHHHHHHHHH', 58: 'AHHHHHHHHHHH'}
        :param queue_log: record queue_length; [step, queue_length]

        :return:
        '''
        c_ts = step*0.1

        if not self.ls_r_dep_times:
            for key, value in r_dpt_type.items():
                self.ls_r_dep_times.extend(range(key, key + len(value)))

        # === 1. Unpack veh info ===
        dic_vehinfo = (
            self.data_recorder.record_multi_lane_info()
            if self.pf
            else self.data_recorder.record_vehinfo()
        )
        ls_veh_id = dic_vehinfo['ls_vehid']
        ls_r_veh_net_asc = dic_vehinfo['ls_r_veh_net_asc']
        ls_r_veh_net_last_asc = dic_vehinfo['ls_r_veh_net_last_asc']
        ls_r_leader_up = dic_vehinfo['ls_r_leader_up']
        ls_r_leader_up_asc = dic_vehinfo['ls_r_leader_up_asc']  # min => max
        ls_r_veh_up = dic_vehinfo['ls_r_veh_up']
        ls_m_leader_up_asc = dic_vehinfo['ls_m_leader_up_asc']  # min => max
        ls_m_veh_up = dic_vehinfo['ls_m_veh_up']

        # === 2. Platoon info (scripts + ramp) ===
        dic_platoon_info = self.merge_regular.get_platoon_info2(
            m_dpt_type=m_dpt_type,
            r_dpt_type=r_dpt_type
        )

        # get throughput on 'center'
        tp = self.data_recorder.record_throughput(st, ls_veh_id, 'center')  # throughput

        # Update mainline platoon ET
        dic_mplatoon_et = self.merge_regular.update_platoon_et(
            step,
            ls_m_leader_up_asc,
            m=True,
            interval=self.mpc_interval
        )

        # === 3. Determine mode: regular / jam ===
        regular_mode, jam_mode = self.mode_switch.determine_mode4(
            ls_m_veh_up,
            ls_r_veh_up,
            ls_r_leader_up
        )
        # jam_mode = True
        # regular_mode = False
        # === 4. Apply corresponding control logic ===
        if jam_mode:
            # Jam mode control
            queue_log = self.merge_jam.jam_control(
                step,
                dic_platoon_info,
                ls_m_leader_up_asc,
                ls_m_veh_up,
                dic_mplatoon_et,
                dic_vehinfo,
                self.ls_r_dep_times,
                self.mpc_interval,
                self.delta_t
            ) # queue_length

        elif regular_mode:
            # Regular mode control
            queue_log = []
            dic_rplatoon_et = self.merge_regular.update_platoon_et(step, ls_r_leader_up_asc, m=False,
                                                                   interval=self.mpc_interval)

            if c_ts % 1 == 0: prc.print_message('**in regular mode**')

            # find head rav; if can be ingored
            ravh_id = self.merge_regular.find_ravh(ls_r_veh_net_asc, ls_r_veh_net_last_asc)

            # get action information
            (dic_ravh_mavh,
             dic_avh_action, ls_avh_act,
             dic_mavh_scAction, ls_mavh_scAct) = self.action_mgr.get_action_info(
                                                                           step,
                                                                           interval=self.mpc_interval
                                                                          )

            # execute actions
            self.action_mgr.execute_action(step, dic_avh_action, ls_avh_act, ls_veh_id)
            self.action_mgr.execute_action(step, dic_mavh_scAction, ls_mavh_scAct, ls_veh_id)

        # average_velocity of this step and its jam_state
        step_speed = self.data_recorder.get_average_speed(step, ls_veh_id, jam_mode)
        self.speed_log.append(step_speed)  # collect into one list

        # 240825
        # ls_hinfo_columns = ['veh_id', 'leader_id', 'headway', 'time_headway', 'time']
        # df_headway_info = self.data_recorder.transform_ls_df(headway_snapshot, ls_hinfo_columns)  # dataframe of headway info

        # If neither regular nor jam_mode, jam_mode is False by default.
        return tp, self.speed_log, queue_log
