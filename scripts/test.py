import numpy as np
import random

def generate_entry_arrivals_poisson(st, p, fr, seed=None):
    """
    'get_schedule_motorway'
    random traffic on motorway/ramp
    every vehicle's departure time

    :param st: simulation time (seconds)
    :param p: av_p
    :param fr: flow rate (veh/h)
    :param seed: random seed
    :return: dic of departure_time and fleet_type
        {1.6: 'HV', 2.6: 'HV', 4.2: 'HV', 5.2: 'HV', 7.5: 'HV'...}
    """
    if seed is not None:
        np.random.seed(seed)  # Ensure reproducibility for NumPy
        random.seed(seed)
    lam = fr / 3600  # lam=>λ; Calculate the arrival rate (veh/h => veh/s)
    min_interval = 1 # the min time headway between vehicles
    av_ratio = p
    result = {}
    t = 0
    while t < st:
        inter_arrival = np.random.exponential(1 / lam)  # Exponential inter-arrival times (Poisson process)
        # Apply the minimum interval constraint
        inter_arrival = max(inter_arrival, min_interval)  # Ensure that vehicles arrive at least 'min_interval' seconds apart
        t += inter_arrival
        # Round the arrival time to one decimal place
        t = round(t, 1)
        if t < st:
            vehicle_type = np.random.choice(["AV", "HV"], p=[av_ratio, 1 - av_ratio])
            result[t] = vehicle_type
    return result

def generate_entry_arrivals_shifted_exp(st, p, fr, min_interval=1.0, seed=None):
    """
    Generate vehicle arrival schedule with:
      - hard constraint: headway >= min_interval (seconds)
      - long-run mean flow approximately equals fr (veh/h)
      - stochasticity: shifted exponential headways

    Returns
    -------
    dict:
        {arrival_time (rounded to 1 decimal): vehicle_type}
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    lam = fr / 3600.0  # veh/s
    if lam <= 0:
        return {}

    mean_headway_target = 1.0 / lam  # seconds
    if min_interval >= mean_headway_target:
        raise ValueError(
            f"Infeasible: min_interval ({min_interval}) must be < target mean headway ({mean_headway_target:.4f}). "
            f"Otherwise you cannot achieve fr={fr} veh/h."
        )

    # Calibrate exponential part
    lam2 = 1.0 / (mean_headway_target - min_interval)

    result = {}
    t = 0.0
    while True:
        extra = np.random.exponential(scale=1.0 / lam2)
        headway = min_interval + extra   # strictly >= min_interval
        t += headway
        if t >= st:
            break

        t_key = round(t, 1)  # keep your original output style
        veh_type = "AV" if (np.random.rand() < p) else "HV"
        result[t_key] = veh_type

    # print info
    print("Final Comparison:")

    sh = st / 3600  # simulation time (s) => simulation hour (h)
    expect_veh_num = fr * sh
    expect_av_num = fr * sh * p
    gen_veh_num = len(result)
    gen_av_num = sum(v == 'AV' for v in result.values())

    print('              ---overview---')
    print(f'expect_veh_num: {expect_veh_num}, gen_veh_num: {gen_veh_num}, diff: {expect_veh_num - gen_veh_num}')
    print(f'expect_av_num: {expect_av_num}, gen_av_num: {gen_av_num}, diff: {expect_av_num - gen_av_num}')
    print(f"Target av_p: {expect_av_num / expect_veh_num:.2f}, av_p: {gen_av_num / gen_veh_num:.2f}")
    return result

#%%
st = 1200
p = 0.3
fr = 1500
seed = 1

#%%
res = generate_entry_arrivals_shifted_exp(st, p, fr, seed=seed)
print(res)

#%%
'''
1h = 60*60 = 3600s
st s => st/3600 h
'''
sh = st/3600 # simulation time (s) => simulation hour (h)
expect_veh_num = fr * sh
expect_av_num = fr * sh * p
# res = generate_entry_arrivals_poisson(st, p, fr)
res = generate_entry_arrivals_shifted_exp(st, p, fr)
gen_veh_num = len(res)
num_av = sum(v == 'AV' for v in res.values())
print(f'expect_veh_num: {expect_veh_num}; gen_veh_num: {gen_veh_num}')
print(f'expect_av_num: {expect_av_num}; num_av: {num_av}')
#%%
print(res)

#%%

def veh_gen_hetero(self, step, dp_times_dict, id_prefix, \
               veh_route, dp_v, dp_lane='0'):
    '''
    ml: multi-lane; hetero: heterogeneous
    2450209: new input format, dic =  {1: 'HV', 31: 'AV', ...,}
    generate veh at corresponding step

    Parameters
    ----------
    dp_times_dict : dic
        departure times of each veh.
        {1: 'HV', 31: 'AV', ...,}
    id_prefix : r or m
        DESCRIPTION.
    veh_route : route1, route2
        DESCRIPTION.
    dp_v : 10 or 27.5 m/s (100 km/h)
        departSpeed.
    dp_lane: 0 or 1 or 2
        departure lane
    '''
    dic_prob = {'hv_cons': 0.51, 'hv_avg': 0.32, 'hv_agg': 0.17}  # probability
    c_ts = step/10 # r_step = time
    if c_ts in dp_times_dict:
        if dp_times_dict[c_ts] == 'AV':
            vehicle_type = 'av'
        else:
            vehicle_type = random.choices(
                population=list(dic_prob.keys()),  # List of HV types
                weights=list(dic_prob.values()),  # Corresponding probabilities
                k=1  # Number of samples to draw
            )[0] # hv_cons, hv_avg, hv_agg

        if dp_lane == '1':
            id = f"{id_prefix}b{vehicle_type}{step}" # use b indicate lane_1
        else:
            id = f"{id_prefix}{vehicle_type}{step}"
        self.add_vehicle(id, veh_route, dp_v, vehicle_type, dp_lane)



