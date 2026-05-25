# ******** PLATOON INITIALISATION ********
        (dic_tags, ls_leader_AV, ls_follower_AV, dic_platoon_size,
         dic_platoon_members, his_dic_platoon_size) = self.platoon_initialise(ls_ihA)

        # ******** HANDLE OVERSIZED PLATOONS ********
        dic_nonOversizedP = self.splitting(st, step, ls_ihA, ls_ihB_av, dic_platoon_size, dic_platoon_members)

        # ******** HANDLE SPARSE PLATOONS ********
        dic_standard_platoon, dic_id_features = self.collecting(st, step, ls_ihA, ls_ihB_av, dic_nonOversizedP,
                                                                dic_platoon_members, dic_tags, ls_vehid)

#%%
dic = {'a' : [1, 2]}
print(dic.keys())

#%%
stop_times = {'veh_1': 10}
if len(stop_times) == 1:
    first_stp_t = next(iter(stop_times.values()))
    print(first_stp_t)

#%%
rp_type = 'AHH'
rp_type2 = "AHHHHA"

final_rp_type = rp_type + rp_type2
print(final_rp_type)

#%%
dic_platoon_merge_time_by_size = {1: 3.75, 2: 6.17, 3: 8.3, 4: 10.6, 5: 12.67, 6: 14.73, 7: 16.91,
                                                   8: 19.0, 9: 21.02, 10: 23.98, 11: 26.3, 12: 28.99}
cum = dic_platoon_merge_time_by_size.get(13)
print(cum)

#%%
state = 0
if state == 'free_mode' or 0:
    print('666')