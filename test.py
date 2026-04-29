# ******** PLATOON INITIALISATION ********
        (dic_tags, ls_leader_AV, ls_follower_AV, dic_platoon_size,
         dic_platoon_members, his_dic_platoon_size) = self.platoon_initialise(ls_ihA)

        # ******** HANDLE OVERSIZED PLATOONS ********
        dic_nonOversizedP = self.splitting(st, step, ls_ihA, ls_ihB_av, dic_platoon_size, dic_platoon_members)

        # ******** HANDLE SPARSE PLATOONS ********
        dic_standard_platoon, dic_id_features = self.collecting(st, step, ls_ihA, ls_ihB_av, dic_nonOversizedP,
                                                                dic_platoon_members, dic_tags, ls_vehid)