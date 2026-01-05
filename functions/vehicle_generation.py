#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:19:17 2024

@author: zzha
"""

'''
Update in 5/25/2024
1. Considering all (p>0.95) type of fleet according to av_p (the penetration of AV)
2. Automatically choose the best interval according to the flow rate and av_p
3. Plot the departure time of all fleets
'''

# generate group number of every type
# num1*4 + num2*3 + num3*2 = total_vehicles
# flow_rate = [180, 360, 540, 720, 900, 1080, 1260, 1440, 1620, 1800, 1980, 2160]

import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

def vdp(p):
    '''
    Find all Fleet types according to av_p
    vdp:vehicle_platoon_distribution

    Parameters
    ----------
    p : penetration of AV.

    Returns
    -------
    r : dic = {'fleet_type1'：p1, 'fleet_type2'：p2, ''}.

    '''
    distribution = {}
    for n in range(12): # 8
        probability = ((1 - p) ** n) * p
        distribution[f'1 lead {n}'] = probability
    # 0.05
    r = {k: v for k, v in distribution.items() if round(v, 3) > 0.05}  # round(v,2) 保留两位小数
    return r


# improved the efficiency, but not accurate like before
# every type will has assignment
def generate_type_num2(percentage, flow_rate, simulation_time, seed=None):
    '''
    Including vdp()
    Based on the available fleet types, generate the quantity for each fleet type.
    Use the nested function vdp to generate the corresponding fleet type with
    the proportion of AV (autonomous vehicles).

    Parameters
    ----------
    percentage : float
        The proportion of AV in the fleet.
    flow_rate : int
        (36*n) veh/h.
    simulation_time : int
        Time in seconds.
    seed : int, optional
        Seed for random number generator. The default is None.

    Raises
    ------
    ValueError
        If the group number is too small to assign at least one to each type.

    Returns
    -------
    closest_result : dict
        {'1 lead 0': 0, '1 lead 1': 10, ...}
    '''
    if seed is not None:
        random.seed(seed)

    total_veh = int((flow_rate * simulation_time) / 3600)
    group_num = int(total_veh * percentage)

    print(f'Total vehicles: {total_veh}')
    print(f'Group number (AV number): {group_num}')

    # Use the vdp(p) function to get dic_ptype
    dic_ptype = vdp(percentage)  # vdp: vehicle_platoon_distribution
    print(f'dic_ptype: {dic_ptype}')  # dic_ptype = {'1 lead 0': p1, '1 lead 1': p2, ...}

    if group_num < len(dic_ptype):
        raise ValueError("Group number too small to assign at least one to each type.")

    # Extract types and their proportions
    types = list(dic_ptype.keys())
    proportions = list(dic_ptype.values())

    closest_result = None
    closest_total = float('inf')

    for num_type1_0 in range(0, group_num + 1):
        for _ in range(100):  # Limit the number of attempts for random assignments # 100
            assigned_nums = [num_type1_0]  # Start with num_type1_0 for '1 lead 0'
            remaining_group_num = group_num - num_type1_0

            # Assign at least one to each remaining type initially
            for i in range(1, len(types)):
                if remaining_group_num > 0:
                    assigned_num = random.randint(1, remaining_group_num)  # Ensure at least 1
                else:
                    assigned_num = 1
                assigned_nums.append(assigned_num)
                remaining_group_num -= assigned_num

            # Adjust the assigned numbers to match the proportions
            while remaining_group_num > 0:
                for i in range(1, len(types)):
                    if remaining_group_num <= 0:
                        break
                    assigned_nums[i] += 1
                    remaining_group_num -= 1

            current_total = sum([assigned_nums[i] * (i + 1) for i in range(len(assigned_nums))])

            if current_total == total_veh:
                result = dict(zip(types, assigned_nums))
                print(f'Optimal solution found: {result}')
                print(f'current_total: {current_total}, total_veh: {total_veh}')
                return result
            elif abs(current_total - total_veh) < abs(closest_total - total_veh):
                closest_result = dict(zip(types, assigned_nums))
                closest_total = current_total

    print(f"No exact solution found. Closest solution: {closest_result}")
    print(f'closest_total: {closest_total}, total_veh: {total_veh}')
    return closest_result


def generate_dt(dic_type_num, simulation_time, interval=0, seed=None):
    '''
    new version
    update 2024.5.24
    according to input dic_type_num to get departure time of every fleet

    Parameters
    ----------
    dic_type_num : dict
        Dictionary containing the number of each fleet type.
    simulation_time : int
        The total simulation time.
    interval : int, optional
        Additional time interval between fleets. The default is 0.
        *** this interval is TAIL and HEAD ***
    seed : int, optional
        Random seed for reproducibility. The default is None.

    Raises
    ------
    ValueError
        If the simulation time is too short to schedule all fleets.

    Returns
    -------
    dp_times_dict : dict
        Dictionary of departure times for each fleet type.
    dp_times : list
        All departure times sorted.
    '''
    if seed is not None:
        random.seed(seed)

    # Initialize an empty list to store the departure times
    dp_times = []
    dp_times_dict = {k: [] for k in dic_type_num}
    max_attempts = 10000  # Set maximum number of attempts

    # Calculate the required time for each type of fleet
    rt_sum = sum(dic_type_num[key] * (int(key.split()[-1]) + 1) for key in dic_type_num)

    # print(f'rt_sum: {rt_sum}')

    # Check if it's feasible to schedule all fleets within the simulation time
    if simulation_time - rt_sum < 0:
        raise ValueError("Simulation time too short to schedule all fleets with minimum interval.")

    # Generate subsequent departure times ensuring a minimum gap
    attempts = 0
    for fleet_type, num in dic_type_num.items():
        num_vehicles = int(fleet_type.split()[-1]) + 1
        while len(dp_times_dict[fleet_type]) < num:
            if attempts >= max_attempts:
                raise ValueError("Unable to find suitable departure times after many attempts")
            dt = random.randint(0, simulation_time)
            if dt not in dp_times and all(abs(dt - x) >= num_vehicles + interval for x in dp_times):
                dp_times.append(dt)
                dp_times_dict[fleet_type].append(dt)
            attempts += 1

    for k in dp_times_dict:
        dp_times_dict[k].sort()
    dp_times.sort()

    print(f'\ndp_times_dict: {dp_times_dict}')
    print(f'dp_times: {dp_times}')
    print(f'fleet_type:{dp_times_dict.keys()}')

    return dp_times_dict, dp_times


def find_max_interval(simulation_time, dic_type_num):
    """
    Calculate the maximum feasible interval for fleets to depart within the \
        given simulation time.

    Parameters:
        simulation_time (int): The total duration of the simulation in seconds.
        dic_type_num (dict): A dictionary where keys are fleet types and values \
            are the number of fleets of each type.

    Returns:
        max_interval = num, departure_times = [172, 184, 196, 208, 220, 232, 244, 256, 268, 280, 292,...]
        A tuple containing the maximum feasible interval (int) and a \
            list of departure times (list of ints).
    """
    # Calculate total number of fleets
    total_fleets = sum(dic_type_num.values())

    # If no fleets, return the simulation time as the max interval
    if total_fleets == 0:
        return simulation_time, []

    # Calculate the maximum possible interval
    max_interval = simulation_time // (total_fleets - 1)
    # according to the veh number of fleet, get correct tail_head interval
    for key in reversed(dic_type_num):
        if dic_type_num[key] != 0:
            max_vehNum = int(key[-1])  # key = '1 lead 1'
            break
    r_max_interval = max_interval - max_vehNum  # turn Head-head interval to Head-tail interval

    last_departure_time = simulation_time - r_max_interval * (total_fleets - 1)
    departure_times = [last_departure_time + i * r_max_interval for i in range(total_fleets)]

    return r_max_interval, departure_times


def find_optimse_schedule(dic_tn, st, max_interval, max_attempts=5, seed=None):
    '''
    include generate_dt()

    according to max_interval to find the depature time of every fleets, if can\
        not find, then decrease the max_interval until find it
    what was find in here is the INTERVAL between TAIL and HEAD of fleets

    Parameters
    ----------
    dic_tn : TYPE
        get this from generate_type_num. # Number of fleet types
    st : TYPE
        simulation time (seconds).
    max_interval : TYPE
        get this from find_max_interval.
    max_attempts : TYPE, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    r : tuple => departure time of fleet (dic) & sum of departure time (list)
        The same as generate_dt's result.

        dp_times_dict : dict
            Dictionary of departure times for each fleet type.
        dp_times : list
            All departure times sorted.
    '''
    while max_interval >= 0:
        for attempt in tqdm(range(max_attempts), desc=f"Trying interval {max_interval}"):
            try:
                r = generate_dt(dic_tn, st, max_interval, seed)
            except:
                r = None
            if r is not None:
                # print(f'r: {r}')
                return r
        max_interval -= 1
    print('No valid schedule found.')
    return None


def convert(dp_times_dict, dp_times):
    '''
    New 24.06.14
    convert VALUES (1 lead 0) to AH ... AND reformat dic
    :param dp_times_dict: {'1 lead 0': [27, 207], '1 lead 1': [1, 53, 101, 142, 162, 188, 227, 249], ...}
    :param dp_times: [1, 27, 53, 77, 101, 121, 142, 162, 188, 207, 227, 249]
    :return: dic_time_type = {1: 'AH', 27: 'A', 77: 'AHHH', ...}
    '''
    # every timestep should coresponding to a type => dic_time_type = {time:type, time:type, ...}
    # transform the structure of data
    dic_time_type = {}
    for t in dp_times:
        if t in dp_times_dict['1 lead 0']:
            dic_time_type[t] = 'A'
        elif t in dp_times_dict['1 lead 1']:
            dic_time_type[t] = 'AH'
        elif t in dp_times_dict['1 lead 2']:
            dic_time_type[t] = 'AHH'
        elif t in dp_times_dict['1 lead 3']:
            dic_time_type[t] = 'AHHH'
        elif t in dp_times_dict['1 lead 4']:
            dic_time_type[t] = 'AHHHH'
        elif t in dp_times_dict['1 lead 5']:
            dic_time_type[t] = 'AHHHHH'
        else:  # 1 lead 6
            dic_time_type[t] = 'AHHHHHH'
    return dic_time_type


def recombine(dic_time_type, n=3):
    '''
    New 24.06.14
    combine nearby fleet, if the veh number of fleet are too small.
    :param dic_time_type: dic_time_type = {1: 'AH', 27: 'A', 77: 'AHHH', ...}
    :param n: expect veh number of fleet
    :return: new_dic = {1: 'AHA', 53: 'AH', 77: 'AHHH'}
    '''
    # Initialize a new dictionary and temporary variables for the current team
    new_dic = {}
    # Get the keys from the dictionary and sort them
    keys = list(dic_time_type.keys())  # [13, 18, 39...]
    temp_team = dic_time_type[keys[0]]  # AH or A
    start_time = keys[0]
    current_vehicles = len(dic_time_type[keys[0]])  # 2 or 1
    for i in range(1, len(keys)):
        # Current key
        current_key = keys[i]
        next_vehicles = len(dic_time_type[current_key])  # next veh number

        # If the total number of vehicles in the current team and the next team is less than or equal to 3, merge the teams
        if current_vehicles + next_vehicles <= n:
            temp_team += dic_time_type[current_key]  # combine first value AH with second value A
            current_vehicles += next_vehicles
        else:
            # Otherwise, add the previous team to the new dictionary and reset temp_team and vehicle count
            new_dic[start_time] = temp_team
            start_time = current_key
            temp_team = dic_time_type[current_key]
            current_vehicles = next_vehicles
    # Add the last combined team to the new dictionary
    new_dic[start_time] = temp_team
    return new_dic

def plot_departure_times2(dic, p, fr, seed, interval_per_vehicle=1):
    """
    Updated 24.06.14
    Plot the departure times of platoons and their types,
    and visualize the distribution of fleet types with sorted x-axis.
    Includes head-to-tail intervals between platoons.

    Parameters:
        dic (dict): A dictionary where keys are departure times and values are platoon types.
        interval_per_vehicle (int): Time interval (in seconds) between vehicles in a platoon.
    """
    # Extract keys and values from the dictionary
    times = list(dic.keys())
    types = list(dic.values())

    # Calculate the number of platoons
    n = len(dic)

    # Generate y-coordinates
    y = list(range(1, n + 1))

    # Calculate head-to-tail intervals
    intervals = []
    for i in range(n):
        if i == 0:
            intervals.append("N/A")  # First platoon has no previous platoon
        else:
            prev_time = times[i - 1]
            prev_length = len(types[i - 1])  # Length of the previous platoon
            current_time = times[i]
            intervals.append(current_time - (prev_time + (prev_length - 1) * interval_per_vehicle))

    # Set the figure size for the whole plot
    fig, ax1 = plt.subplots(figsize=(8, 14))

    # Create the first plot (scatter plot of departure times)
    ax1.scatter(y, times, marker='_', color='blue', alpha=0.7)

    # Add labels for each point
    for i, (time, platoon_type, interval) in enumerate(zip(times, types, intervals)):
        label = f"{platoon_type} ({time}, {interval})"
        ax1.text(y[i], time, label, fontsize=9, ha='left', va='center')

    # Add labels and title
    ax1.set_xlabel('Groups')
    ax1.set_ylabel('Departure time (s)')
    ax1.set_title('Platoon Groups and Departure Times with Head-Tail Intervals')
    ax1.grid(True)

    # Now create the inset for the second plot (bar chart of fleet type distribution)
    ax2 = ax1.inset_axes([0.19, 0.67, 0.28, 0.28])  # [left, bottom, width, height] - relative to ax1

    # Count occurrences of each type
    type_counts = Counter(types)

    # Sort the platoon types in ascending order
    sorted_types = sorted(type_counts.keys())
    indices_sorted_types = range(len(sorted_types))
    sorted_counts = [type_counts[t] for t in sorted_types]

    # Plot the bar chart in the inset with swapped axes
    bars = ax2.barh(indices_sorted_types, sorted_counts, color='blue', alpha=0.7, edgecolor='black')
    ax2.set_yticks(indices_sorted_types)
    ax2.set_yticklabels(sorted_types)

    # Add labels and title to the inset
    ax2.set_xlabel('Count')
    ax2.set_ylabel('Fleet Type')
    ax2.set_title('Distribution of Fleet Types')

    # Add numbers on top of the bars
    for bar in bars:
        xval = bar.get_width()
        ax2.text(xval + 0.2, bar.get_y() + bar.get_height() / 2, str(int(xval)), ha='left', va='center', fontsize=10)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.title(f'Info: avp:{p}, flow_rate:{fr}, seed:{seed}')
    plt.show()

def plot_departure_times(dic):
    '''
    New 24.06.14
    :param dic:
    :return:
    '''
    # Extract keys and values from the dic
    times = list(dic.keys())
    types = list(dic.values())

    # Calculate the numbert of platoons
    n = len(dic)

    # Generate y-coordinates
    y = list(range(1, n + 1))

    # Create the plot
    plt.figure(figsize=(8, 25))
    plt.scatter(times, y, marker='_')

    # Add labels for each point
    for i, (time, type) in enumerate(zip(times, types)):
        plt.text(time, y[i], type, fontsize=9, ha='left', va='center')

    # Add labels and title
    plt.xlabel('Departure time (s)')
    plt.ylabel('Groups')
    plt.title('Platoon groups and departure time')
    plt.grid(True)

    # Show the plot
    plt.show()


def get_schedule(st, p, fr, max_attempts=5, plot=False, seed=None):
    """
    combine above function
    :param st:
    :param p:
    :param fr:
    :param seed:
    :return: dic of departure_time and fleet_type
    """
    dic_tn = generate_type_num2(percentage=p, flow_rate=fr, simulation_time=st, seed=seed)
    max_interval, ls_dt = find_max_interval(st, dic_tn)
    dp_times_dict, dp_times = find_optimse_schedule(dic_tn, st, max_interval, \
                                                    max_attempts=max_attempts, seed=seed)
    dic_time_type = convert(dp_times_dict, dp_times)
    result = recombine(dic_time_type) # dic_departure_time and fleet type
    if plot:
        # plot_departure_times(dic_time_type)
        # plot_departure_times(result)
        plot_departure_times2(dic_time_type, p, fr, seed)
        plot_departure_times2(result, p, fr, seed)
    print(f'dic_dpt_type:{result}')
    return result


def get_schedule_startT(start_t, p, fr, max_attempts, plot=False, seed=None):
    '''
    100324updated: from set start time generate vehicle
    :param self:
    :param start_t: 600
    :param p:
    :param fr:
    :param max_attempts:
    :param plot:
    :param seed:
    :return:
    '''
    st = 1200
    real_st = st - start_t
    dic_tn = generate_type_num2(percentage=p, flow_rate=fr, simulation_time=real_st, seed=seed)
    max_interval, ls_dt = find_max_interval(real_st, dic_tn)
    dp_times_dict, dp_times = find_optimse_schedule(dic_tn, real_st, max_interval, \
                                                    max_attempts=max_attempts, seed=seed)
    dic_time_type = convert(dp_times_dict, dp_times)
    result = recombine(dic_time_type)  # dic_departure_time and fleet type
    updated_result = {key + start_t: value for key, value in result.items()}
    if plot:
        plot_departure_times(updated_result)
    print(f'dic_dpt_type:{updated_result}')
    return updated_result


class VehGen:
    """
    generate vehicle according to schedule dict
    """
    def __init__(self, traci):
        self.traci = traci

    # ls = rdp_sum or mdp_sum
    def add_vehicle(self, vehicle_id, route_id, dp_v, type_id="idm"):
        self.traci.vehicle.add(vehID=vehicle_id, routeID=route_id, typeID=type_id, \
                          departSpeed=dp_v)
        # set speedmode at the same time
        self.traci.vehicle.setSpeedMode(vehicle_id, 0b000001)

    # step = 0,1,2,3,4,5
    # r_step = 0,0.1,0.2,0.3...
    # veh_route = 'route2' or 'route1'
    # id_prefix = 'r' or 'm'
    def veh_gen5(self, step, dp_times_dict, id_prefix, \
                 veh_route, dp_v):
        '''
        240616: delete data recording as these can get from the departure information
        240614: new input format, dic =  {1: 'AHA', 31: 'AH', 48: 'AHA', 79: 'AA', 107: 'AH', 121: 'AHA'}
        240609: update from dic_rpav_type to dic_av_type
        generate veh at cooresponding step

        Parameters
        ----------
        dp_times_dict : dic
            departure times of each fleet type.
            {'1 lead 0' : [603], '1 lead 1' : [68, 831, ...]}
        dic_av_type : dic
            {av_id : vehicle number}.
        dic_dep : dic
            {av_id : departure times of each av}.
        id_prefix : TYPE
            DESCRIPTION.
        veh_route : TYPE
            DESCRIPTION.
        dp_v : TYPE
            departSpeed.

        Returns
        -------
        None.

        '''
        for dt, type in dp_times_dict.items():
            r_step = step/10 # r_step = time
            veh_num = len(type) # the veh number of this platoon
            # type = 'AHA'
            for i, veh in enumerate(type):
                # dt_v = dt+i # departure time of this veh
                dt_v = dt + i  # departure time of this veh
                if dt_v == r_step:
                    if i==0:
                        id_body = 'avh' # head AV => avh
                    else:
                        id_body = 'av' if veh == 'A' else 'hv'
                    vehicle_type = 'av' if veh=='A' else 'idm'
                    id_suffix = id_prefix + id_body
                    id = f"{id_suffix}{step}"
                    if id == 'rhv4030':
                        pass
                    self.add_vehicle(id, veh_route, dp_v, vehicle_type)


if __name__ == '__main__':
    st = 1000
    p = 0.2
    fr = 1080  # 360
    max_attempts = 7
    seed = None
    plot = True
    dic_dpt_type = get_schedule(st, p, fr, max_attempts, plot=plot, seed=seed)

    # start_t = 600
    # new_fr = 1080
    # new_dic = get_schedule_startT(start_t, p, fr, max_attempts, plot=True, seed=1)






