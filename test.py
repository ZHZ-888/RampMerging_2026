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
    ls_m_leader_up_asc = dic_vid_groups['ls_m_leader_up_asc']
    ls_r_leader_up = dic_vid_groups['ls_r_leader_up']
    ls_mr_leader_up = ls_m_leader_up_asc + ls_r_leader_up

    for leader in ls_mr_leader_up:
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
                self.data_recorder.ls_tail_ids.append(tail_id)  # ls_tail_ids (purpose?)
            dic_head_states = self.data_recorder.get_vid_states(leader)
            dic_tail_states = self.data_recorder.get_vid_states(tail_id)
            pos_head = dic_head_states['pos']
            pos_tail = dic_tail_states['pos']
            platoon_length = pos_head - pos_tail
            # record length
            self.dic_platoon_info[leader][1].append(platoon_length)
    self.data_recorder.dic_platoon_info = self.dic_platoon_info
    return self.dic_platoon_info