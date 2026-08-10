def _get_m_leader_action(self, step, first_r_leader,
                         rp_pass_dur, m_leader, max_interval,
                         mpc_interval):
    """
    _get_mavh_action => _get_m_leader_action
    Decide whether a MAVH (mainline leader) should take action to match the desired merging time.

    Params:
        step: current simulation step
        first_r_leader: first ramp leader ID
        pv_m_leader: preceding vehicle of m_leader (short as pv)
        rp_pass_dur: ramp platoon passing duration
        m_leader: candidate mainline vehicle ID
        max_interval: max time gap between ramp platoon and MAVH
        dic_mplatoon_et: estimated arrival time dict for platoon
        delta_t: allowable timing error
        mpc_interval: frequency of evaluation
        ts: timestamp
        dur: duration (time period)
    Returns:
        self.dic_mavh_actionP: dict of MAVH (m_leader) and its action parameters
        => self.dic_m_leader_action_params = {m_leader: [, c_ts]}
    """
    c_ts = round(step / 10 + 0.1, 1)
    allowable_error = self.delta_t  # 0, 2, 4, 6, 8, 10
    last_stop_ts = list(self.stop_times.items())[-1][-1] if self.stop_times else None
    if self.first_ramp_stop_ts is not None:
        # in cooldown period, not action
        if c_ts - self.first_ramp_stop_ts < self.cooldown_dur:
            return self.dic_m_leader_action_params

    if not (step % mpc_interval == 0 or (
            last_stop_ts is not None and c_ts == last_stop_ts + 0.1)):  # *10 because sim_step=0.1
        # move forward only if in mpc_interval or just after stop
        return self.dic_m_leader_action_params

    if not m_leader:
        return self.dic_m_leader_action_params

    pv_m_leader_info = self.traci.vehicle.getLeader(m_leader)
    pv_m_leader, dis_m_leader_to_pv = pv_m_leader_info if pv_m_leader_info else (None, None)
    if not pv_m_leader:
        return self.dic_m_leader_action_params

    pv_m_leader_lane = self.data_recorder.dic_lane[pv_m_leader]
    if not self.stop_state or pv_m_leader_lane != 'inflow_highway_0':
        return self.dic_m_leader_action_params

    pv_tail_reach_ts = self._get_prev_platoon_tail_at_ts(c_ts, m_leader)
    pv_tail_reach_dur = max(0, pv_tail_reach_ts - c_ts)
    # dev_rleader_pmtail: time deviation between r_leader and previous m_tail
    dev_rleader_pmtail = max(0, self.r_leader_acc_dur - pv_tail_reach_dur)  # self.r_leader_acc_dur = 9,3 (ml) or 12
    des_interval = dev_rleader_pmtail + rp_pass_dur + self.buffer * 2
    interval_shortage = des_interval - max_interval
    if interval_shortage <= 0: # no need to take action
        self.dic_m_leader_action_params = {m_leader: []}
        return self.dic_m_leader_action_params

    # interval_shortage > 0, m_leader needs to take action
    if self.stop_times[first_r_leader] == self.first_ramp_stop_ts:
        r_leader_waiting_dur = c_ts - self.stop_times[first_r_leader] - self.cooldown_dur
    else:
        r_leader_waiting_dur = c_ts - self.stop_times[first_r_leader]

    # Case 1: If r_leader has been waiting too long, allow looser error margin to avoid long waiting
    if r_leader_waiting_dur > 30 and interval_shortage <= allowable_error + 10:
        # Looser threshold due to long waiting time
        des_m_leader_reach_dur = des_interval + pv_tail_reach_dur
    # Case 2: Otherwise, allow only if within strict allowable error
    elif interval_shortage <= allowable_error:
        # Strict error control
        des_m_leader_reach_dur = des_interval + pv_tail_reach_dur
    else:
        return self.dic_m_leader_action_params

    dic_m_leader_info = self.data_recorder.get_vid_states(m_leader)
    m_dis = dic_m_leader_info['dis']  # m_leader distance to ws
    m_v0 = dic_m_leader_info['v']
    action_params = list(self.merge_regular.get_action_params(des_m_leader_reach_dur, m_dis, m_v0))
    action_params.append(c_ts)  # (t1, a1, t3, a3, v_reach, c_ts)
    self.m_leader_action_dic[m_leader] = action_params
    self.dic_m_leader_action_params = {m_leader: action_params}
    return self.dic_m_leader_action_params