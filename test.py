def get_action_params(self, t, dis, v0):
    '''

    Parameters
    ----------
    t : TYPE
        time require.
    dis : TYPE
        the remaining distance to weaving section of this leader.
    v0 : TYPE
        current speed.

    Returns
    -------
    ls_acc_profile : list
        the acc/dec strategy.
        (t1, a1, t3, a3, v_arrival) or (T, a) v_arrival => velocity of reaching moment
    '''
    

    t_v0_vmax = (self.max_speed - v0) / self.amax  # duration to accelerate to peak velocity
    dis_v0_vmax = v0 * t_v0_vmax + 0.5 * self.amax * t_v0_vmax ** 2  # distance for speed increase to max_v

    t_left = t - t_v0_vmax
    dis_left = t_left * self.max_speed
    dis_sum = dis_v0_vmax + dis_left  # farthest driving distance in period t

    if dis_sum >= dis:  # TODO: this part should be replaced
        prc.print_message(
            f"CASE 1: this leader will arrive MS within {t:.2f} s "
            f"(e.g., r_platoon will encounter with m_platoon),\n"
            f"max_travel_dis_in_t {dis_sum:.2f} m >= current_dis_to_WS {dis:.2f} m"
        )
        if self.optimizer:
            # new add min speed
            optm = GetBVCurve2(v0, t, dis=dis, min_speed=5)
            # v_arrival: velocity of reaching moment
            res, v_arrival = optm.optimize()  # res.x = (t1, a1, t3, a3)
            # res.x[3] < 0.01 updated 110824, to avoid stop at the end of ramp caused by conflict
            if v_arrival < 1 or res.x[3] < 0.01:  # to avoid sudden stop; 241003update, avoid stop can start
                prc.print_message("**Huge difference, NO WAY to avoid the conflict**")
                return [None, self.amax]
            ls_acc_profile = list(np.append(res.x, v_arrival))  # ls_acc_profile = (t1, a1, t3, a3, v_arrival)
            return ls_acc_profile
        else:
            fomula = '2*v0*t+2*a*t*t_v0_vmax-a*t_v0_vmax**2-2*self.dis'
            self.ls_v0.append(v0)
            self.ls_teR.append(t)
            optimal_a, optimal_vt, optimal_t1 = self.calculate_optimal_acceleration(v0, t, fomula)
            a = optimal_a
            vt = optimal_vt
            T = optimal_t1
            ls_acc_profile = [T, a]
            return ls_acc_profile
    else:
        prc.print_message(
            f"CASE 2: this leader can't arrive MS in {t:.2f} s (eg: no conflict), \n"
            f"max_travel_dis_in_t {dis_sum:.2f} m < current_dis_to_WS {dis:.2f} m"
        )
        ls_acc_profile = [None, self.amax]
        prc.print_message(f'action: apply_acc {self.amax}')
        return ls_acc_profile