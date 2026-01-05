from functions import print_control as prc
from functions import v2x_disturbance as v2x
class ActionManager:
    def __init__(self, instance_dr, merge_regular, loss_rate=0):
        self.merge_regular = merge_regular
        self.data_recorder = instance_dr
        self.loss_rate = loss_rate
        self.delay_buffer = v2x.UpdateDelayBuffer(loss_rate=self.loss_rate)

        self.dic_ravh_mavh = None
        self.dic_avh_action = None
        self.ls_avh_act = None
        self.dic_mavh_scAction = None
        self.ls_mavh_scAct = None
        self.last_update_payload = None

    def update(self, dic_ravh_mavh, dic_avh_action, ls_avh_act, dic_mavh_scAction, ls_mavh_scAct):
        self.dic_ravh_mavh = dic_ravh_mavh
        self.dic_avh_action = dic_avh_action
        self.ls_avh_act = ls_avh_act
        self.dic_mavh_scAction = dic_mavh_scAction
        self.ls_mavh_scAct = ls_mavh_scAct

    def build_action_payload(self, step, interval=70):
        """
        :param step: the step number to check if it's a multiple of the interval
        :param interval: MPC interval, default is 70
        :return: a tuple containing dictionaries and lists representing various action-related data if the step is a multiple of the interval, otherwise None
        """
        update_payload = None
        if step % interval == 0:
            dic_ravh_mavh, dic_rm_c = self.merge_regular.find_ravh_mavh()
            dic_avh_action, ls_avh_act = self.merge_regular.get_avh_action()
            dic_mavh_scAction, ls_mavh_scAct = self.merge_regular.get_mavh_scAction()
            self.print_decision(dic_rm_c, ls_mavh_scAct)
            update_payload = (dic_ravh_mavh, dic_avh_action, ls_avh_act, dic_mavh_scAction, ls_mavh_scAct)
        return update_payload

    def get_action_disturbed(self, step, interval=70):
        """
        consider v2x disturbance
        :param step: time step
        :param interval: MPC interval, default is 70
        :return: a tuple containing the current state of action info, including dictionaries and lists
        """
        # Generate action commands at each interval
        update_payload = self.build_action_payload(step, interval)
        if update_payload:
            # Step 3: Check if it's a new (non-empty) command
            is_redundant = (update_payload == self.last_update_payload or
                            all(not x for x in update_payload))
            if not is_redundant:  # new command
                self.last_update_payload = update_payload  # store for next comparison
                self.delay_buffer.push(step, update_payload)  # <<< drop or delay here
            else:
                prc.print_message("empty or redundant command")
        delayed_payload = self.delay_buffer.maybe_release(step)  # Check if delayed payload is ready to execute
        delayed_payload and self.update(*delayed_payload)  # <<< Delay-handled update()
        # Return the current (possibly stale) action info
        return (self.dic_ravh_mavh, self.dic_avh_action, self.ls_avh_act, self.dic_mavh_scAction, self.ls_mavh_scAct)

    def get_action_clean(self, step, interval=70):
        """
        Assume perfect V2X
        :param step:
        :param interval:
        :return:
        """
        update_payload = self.build_action_payload(step, interval)
        if update_payload:
            self.update(*update_payload)
        return (self.dic_ravh_mavh, self.dic_avh_action, self.ls_avh_act, self.dic_mavh_scAction, self.ls_mavh_scAct)

    def get_action_info(self, step, interval=70):
        """
        :param step: The step for which action info is requested.
        :param interval: The interval between disturbances (default is 70).
        :return: Action information based on whether there is a disturbance or not.
        """
        disturb = self.loss_rate != 0
        if disturb:
            return self.get_action_disturbed(step, interval)
        else:
            return self.get_action_clean(step, interval)

    def print_decision(self, dic_rm_c, ls_mavh_scAct):
        '''
        print the action decision detail
        :param dic_rm_c: dic of encountered ravh_mavh, and choose action avh
        :param ls_mavh_scAct: action list of
        :return:
        '''
        dic_vehinfo = self.data_recorder.record_vehinfo()
        ls_mr_leader_up = dic_vehinfo["ls_mr_leader_up"] # all avh before merging
        ls_m_leader_up = dic_vehinfo["ls_m_leader_up"] # all mavh before merging
        action_dic = {key: value for key, value in dic_rm_c.items() if value in ls_mr_leader_up}
        # prc.print_message(action_dic)
        # filter out duplicate ravh, only keep the last pair
        result = {}
        for key, value in action_dic.items():
            result[key[0]] = (key, value)
        action_dic_filter = {key: value for key, value in result.values()}
        prc.print_message(f'\n{action_dic_filter}')
        for key, value in action_dic_filter.items():
            ravh = key[0]
            mavh = key[1]
            action_avh = value
            prc.print_message(f'STATE: {ravh} encounter {mavh}')
            if 'm' in action_avh:
                # type 1: only mavh need to take action
                prc.print_message(f'DECISION: *action type 1*\n{action_avh} take action, make sure head {mavh} > tail {ravh}')
            else:
                # type2: ravh take action, mavh no action, but mavh's next mavh may need to take action
                prc.print_message(f'DECISION: *action type 2*\n{action_avh} take action, make sure head {ravh} > tail {mavh}')
                # the next mavh of this mavh
                mavh_time = int(mavh[4:])
                mavh_n = min((vehicle for vehicle in ls_m_leader_up if int(vehicle[4:]) > mavh_time),
                             key=lambda x: int(x[4:]), default=None)  # Handle the case where no later vehicle exists
                if mavh_n in ls_mavh_scAct:
                    prc.print_message(f'{mavh_n} (next avh of {mavh}) need to take action, make sure head {mavh_n} > tail {ravh}')
        prc.print_message() # print one blank

    def execute_action(self, step, dic, ls, ls_veh_id):
        try:
            self.merge_regular.apply_avh_action(dic)
            ls_valid = [veh_id for veh_id in ls if veh_id in ls_veh_id]
            self.merge_regular.flashing2(step, ls_valid)
        except:
            pass

