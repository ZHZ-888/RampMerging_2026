for i, head_id in enumerate(ls_m_leader_up_asc):
    if head_id not in self.dic_mplatoon_et:
        headway_differences[head_id] = 0

    elif i == 0 and head_id != first_veh:
        ts_head_current = self.dic_mplatoon_et[head_id][1]
        prev_leader = list(self.dic_mplatoon_et.keys())[
            list(self.dic_mplatoon_et.keys()).index(head_id) - 1]
        ts_prev_tail = self.dic_mplatoon_et[prev_leader][2]
        headway_differences[head_id] = ts_head_current - ts_prev_tail

    elif i > 0:
        # Get the arrival time of the current head vehicle
        ts_head_current = self.dic_mplatoon_et[head_id][1]
        prev_leader = list(self.dic_mplatoon_et.keys())[list(self.dic_mplatoon_et.keys()).index(head_id) - 1]
        ts_prev_tail = self.dic_mplatoon_et[prev_leader][2]
        ts_front_remaining = ts_prev_tail - c_ts

        if ts_front_remaining >= self.r_leader_acc_dur:
            # Calculate the time difference
            headway_differences[head_id] = ts_head_current - ts_prev_tail
        else:
            headway_differences[head_id] = ts_head_current - ts_prev_tail - (
                    self.r_leader_acc_dur - ts_front_remaining)
#%%

import re
id = 'm_av180'
type = re.sub(r".*_([A-Za-z]+)[0-9]+$", r"\1", id)
print(type)

