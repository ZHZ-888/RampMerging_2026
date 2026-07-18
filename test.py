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


#%%
dic = {'mb_av10092': [0.8310068845748901, 0.99], 'mb_av10324': [0.4607242941856384, 0.5], 'mb_av1232': [0.8249494433403015, 0.97], 'mb_av2582': [0.3528490662574768, 0.3333333333333333], 'mb_av2639': [0.8219226002693176, -0.1], 'mb_av2812': [0.3494994044303894], 'mb_av3004': [0.2935275137424469, 0.5], 'mb_av3229': [0.23408839106559753, 0.5], 'mb_av4027': [0.3964613974094391, 0.5], 'mb_av4108': [0.833189845085144, 1.0], 'mb_av4420': [0.3317866921424866, 0.42857142857142855], 'mb_av5939': [0.8055134415626526, 1.0], 'mb_av7435': [0.4051932394504547, 0.5], 'mb_av747': [0.8289699554443359, 1.0], 'mb_av7870': [0.8294432163238525, 0.99], 'mb_av8117': [0.8165528774261475, 1.0], 'mb_av931': [0.8308647274971008, 0.99]}
rewards = [v[1] for v in dic.values() if len(v) == 2]
count = len(rewards)
reward_sum = sum(rewards)
reward_avg = reward_sum / count if count else 0
