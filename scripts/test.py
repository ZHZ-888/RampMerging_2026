def predict_following_state(self, dic_id_type, ls_vehid, model=False):
    '''
    250520 updated version: platoon-wise prediction.
    If any follower is predicted as 'free', all subsequent followers in the same platoon
    will be automatically labeled as free without prediction (but still recorded).

    :param dic_id_type (dic_tags): {id:tag, ..., } asc order, before merging, on mainlane;
                                    0: HV follower, 1: leader_AV, 2: follower_AV
           ls_vehid: all veh on net at this step; the order is not important
           # dic_promotedAV: {AV_id: type, ...}, type = 'split' or 'free'
           model: whether to perform prediction using fs_model
    :return:
            0:free; 1:following
            self.dic_id_preState = {id: state,... } # id start from the first follower of the first AV_leader
            the sequence: decrease or increase
    '''
    if not dic_id_type:
        return self.dic_id_preState, self.dic_id_features

    # if a new av leader promote or emerge, repredict follower states






    # Only proceed when a new follower appears
    new_follower_id, newest_tag = next(reversed(dic_id_type.items()))  # get the new in veh_id and veh_tag
    # If a new platoon leader appears, reset free_triggered
    if newest_tag == 1:
        self.free_triggered = False
    # Only process new followers (skip if already processed or is a leader)
    if new_follower_id in self.dic_id_features or newest_tag == 1 or new_follower_id not in ls_vehid:
        return self.dic_id_preState, self.dic_id_features
    # == get features == always extract features for training
    arr_select_features = self.get_RFfeatures(new_follower_id)
    if arr_select_features is None:
        return self.dic_id_preState, self.dic_id_features  # or `continue` if used in a loop
    # == prediction ==
    if model:
        # if model == True, predict the state of the newest_follower
        if self.free_triggered:
            # free_triggered = True; No prediction needed, directly mark as free
            pre_state = [0]
        else:
            # free_triggered = False; Perform prediction using model
            pre_state = self.fs_model.predict(arr_select_features)
        self.dic_id_preState[new_follower_id] = pre_state[0]

        if pre_state[0] == 0:  # free
            self.free_triggered = True
    return self.dic_id_preState, self.dic_id_features  # self.dic_id_features includes all id & features