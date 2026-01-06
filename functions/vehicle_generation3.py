#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:19:17 2024

@author: zzha
"""

'''
Update in 6/15/25
1. Generate vehicles that out of the platoon

Update in 12/13/2024
1. Modified the logic of platoon types pick up under different av_p
2. Improved the logic and accuracy of platoon allocation
2. Optimised the departure time plot

Update in 5/25/2024
1. Considering all (p>0.95) type of fleet according to av_p (the penetration of AV)
2. Automatically choose the best interval according to the flow rate and av_p
3. Plot the departure time of all fleets
'''

# generate group number of every type
# num1*4 + num2*3 + num3*2 = total_vehicles
# flow_rate = [180, 360, 540, 720, 900, 1080, 1260, 1440, 1620, 1800, 1980, 2160]

import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from collections import defaultdict

def vdp(p):
    '''
    Updated 241203
    Determining platoon types based on the internal av_p of the platoon/Report PPT page 75
    Find all Fleet types according to av_p (0.1, 0.15, 0.2, 0.3)
    vdp: vehicle_platoon_distribution

    Parameters
    ----------
    p : penetration of AV.

    Returns
    -------
    r : dic = {'fleet_type1'：p1, 'fleet_type2'：p2, ''}.

    '''
    if p == 0.1:
        return {f"1 lead {i}": 0 for i in range(12)}
    elif p == 0.15:
        return {f"1 lead {i}": 0 for i in range(12)}
    elif p == 0.2:
        return {f"1 lead {i}": 0 for i in range(12)}
    elif p == 0.3:
        return {f"1 lead {i}": 0 for i in range(12)}
    else:
        raise ValueError("Unsupported AV penetration rate.")
    return r

def generate_type_num4(av_percentage, flow_rate, simulation_time, platoon_percentage, seed=None, tries=7):
    '''
    Generate platoon types based on Poisson distribution.
    Each platoon is led by an AV, with up to 11 followers.
    Followers prefer HVs, but AVs can be used if HVs run out.

    :param av_percentage: proportion of AVs in all vehicles (0-1)
    :param flow_rate: vehicles per hour
    :param simulation_time: seconds
    :param platoon_percentage: proportion of vehicles in platoons
    :param seed: random seed
    :param tries: number of attempts to optimize AV/HV allocation
    :other param: lambda_val controls average follower count per AV-led platoon
                  More AVs → smaller lambda (shorter platoons); fewer AVs → larger lambda
    :return: dic_tn =  {'1 lead 0': (count, av_follower_num), '1 lead 1': (count, av_follower_num), ...}
    '''
    total_veh_num = int((flow_rate * simulation_time) / 3600)
    # number of free HV
    free_hv_num = int(total_veh_num * (1-platoon_percentage))
    platoon_veh_num = int(total_veh_num * platoon_percentage)
    av_num = int(total_veh_num * av_percentage)
    hv_num = int(platoon_veh_num - av_num)

    alpha = 1.0
    lambda_val = alpha * (1 - av_percentage) / av_percentage #
    max_follower = 11

    best_result = None
    min_total_diff = float('inf')

    for i in range(tries):
        rng = np.random.default_rng(seed + i if seed is not None else None)

        result = {}
        used_av = 0
        used_hv = 0

        while True:
            if used_av >= av_num:
                break

            n_followers = min(rng.poisson(lambda_val), max_follower)
            n_hv = min(n_followers, hv_num - used_hv)
            n_av = n_followers - n_hv

            if used_av + 1 + n_av > av_num:
                break

            platoon_key = f"1 lead {n_followers}"
            if platoon_key not in result:
                result[platoon_key] = [0, 0]
            result[platoon_key][0] += 1
            result[platoon_key][1] += n_av

            used_av += 1 + n_av
            used_hv += n_hv

        # Evaluate difference
        total_leader_av = sum(v[0] for v in result.values())
        total_follower_av = sum(v[1] for v in result.values())
        result_av = total_leader_av + total_follower_av
        result_hv = sum(int(k.split()[-1]) * v[0] - v[1] for k, v in result.items())

        av_diff = abs(av_num - result_av)
        hv_diff = abs(hv_num - result_hv)
        total_diff = av_diff + hv_diff

        if total_diff < min_total_diff:
            min_total_diff = total_diff
            best_result = result

    # Format best result
    res = {k: (v[0], v[1]) for k, v in best_result.items()}
    # sort res
    sorted_distribution = dict(sorted(
        res.items(),
        key=lambda x: int(x[0].split()[-1])
    ))

    # Final printout
    print("Final Comparison:")
    result_leaderAV_num = sum(v[0] for v in sorted_distribution.values())
    result_followerAV_num = sum(v[1] for v in sorted_distribution.values())
    result_av_num = result_leaderAV_num + result_followerAV_num
    result_hv_num = sum(int(k.split()[-1]) * v[0] - v[1] for k, v in sorted_distribution.items())
    result_total_veh_num = result_hv_num + result_av_num
    print('              ---overview---')
    print(f'total_veh_num: {total_veh_num}, free_hv_num: {free_hv_num}')
    print(f'platoon_veh_num: {platoon_veh_num}, platoon_av: {av_num}, platoon_hv: {hv_num}')
    print('              ---detail---')
    print(f"Target AVs: {av_num}, Allocated AVs: {result_av_num}, Difference: {av_num - result_av_num}")
    print(f"Target HVs: {hv_num}, Allocated HVs: {result_hv_num}, Difference: {hv_num - result_hv_num}")
    print(f'result_leaderAV_num: {result_leaderAV_num}, result_followerAV_num: {result_followerAV_num}')
    print(f"Target av_p: {av_num / total_veh_num:.2f}, av_p: {result_av_num / result_total_veh_num:.2f}")
    return sorted_distribution, result_followerAV_num, free_hv_num

def generate_dt(dic_type_num, simulation_time, interval=0, seed=None):
    '''
    new version
    update 2024.5.24
    according to input dic_type_num to get departure time of every fleet

    Parameters
    ----------
    dic_type_num : dict
        dic_tn =  {'1 lead 0': (count, av_follower_num), '1 lead 1': (count, av_follower_num), ...}
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

    # sum the required time for each type of fleet
    rt_sum = sum(dic_type_num[key][0] * (int(key.split()[-1]) + 1) for key in dic_type_num)

    # Check if it's feasible to schedule all fleets within the simulation time
    if simulation_time - rt_sum < 0:
        raise ValueError("Simulation time too short to schedule all fleets with minimum interval.")

    # Generate subsequent departure times ensuring a minimum gap
    attempts = 0
    for fleet_type, (fleet_num, followerAV_num) in dic_type_num.items():
        num_vehicles = int(fleet_type.split()[-1]) + 1
        while len(dp_times_dict[fleet_type]) < fleet_num:
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

def find_max_interval2(simulation_time, dic_type_num):
    """
    Calculate the maximum feasible interval for fleets to depart within the \
        given simulation time.

    Parameters:
        simulation_time (int): The total duration of the simulation in seconds.
        dic_type_num (dict): {'1 lead 0': (count, av_follower_num), '1 lead 1': (count, av_follower_num), ...}

    Returns:
        max_interval = num, departure_times = [172, 184, 196, 208, 220, 232, 244, 256, 268, 280, 292,...]
        A tuple containing the maximum feasible interval (int) and a \
            list of departure times (list of ints).
    """
    # Calculate total number of fleets
    fleet_num = sum(v[0] for v in dic_type_num.values())
    # Total veh number
    result_leaderAV_num = fleet_num
    result_followerAV_num = sum(v[1] for v in dic_type_num.values())
    result_av_num = result_leaderAV_num + result_followerAV_num
    result_hv_num = sum(int(k.split()[-1]) * v[0] - v[1] for k, v in dic_type_num.items())
    total_veh_num = result_hv_num + result_av_num

    # If no fleets, return the simulation time as the max interval
    if fleet_num == 0:
        return simulation_time

    # Calculate the maximum possible interval
    max_interval = (simulation_time - total_veh_num) // (fleet_num - 1)
    return max_interval

def find_optimse_schedule2(dic_tn, st, max_interval, max_attempts=5, seed=None):
    '''
    include generate_dt()

    according to max_interval to find the depature time of every fleets, if can\
        not find, then decrease the max_interval until find it
    what was find in here is the INTERVAL between TAIL and HEAD of fleets

    Parameters
    ----------
    dic_tn : TYPE
        {'1 lead 0': (count, av_follower_num), '1 lead 1': (count, av_follower_num), ...}
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

def convert2(dp_times_dict, dp_times):
    '''
    Convert departure times to string-type labels such as "A", "AH", "AHH", etc.

    Parameters:
        dp_times_dict (dict): Dictionary where keys are platoon types (e.g. '1 lead 3')
                              and values are lists of departure times.
        dp_times (list): All departure times in the simulation.

    Returns:
        dic_dpt_type (dict): A dictionary mapping each departure time to its type string,
                             dpt => DeParture time
                             e.g. {1: 'AH', 27: 'A', ...}
    '''

    dic_dpt_type = {}

    # Iterate over possible keys from '1 lead 0' to '1 lead 11'
    for i in range(12):
        key = f'1 lead {i}'
        if key not in dp_times_dict:
            continue  # Skip if this platoon type doesn't exist

        # Construct label string: e.g., 'A', 'AH', 'AHH', etc.
        fleet_type = 'A' + 'H' * i

        # Assign type to each departure time in this category
        for t in dp_times_dict[key]:
            dic_dpt_type[t] = fleet_type

    # Ensure every time in dp_times has a label, even if unmatched
    for t in dp_times:
        if t not in dic_dpt_type:
            dic_dpt_type[t] = 'unknown'
    sorted_dic_dpt_type = dict(sorted(dic_dpt_type.items()))
    return sorted_dic_dpt_type

def recombine(dic_dpt_type, n=3):
    '''
    New 24.06.14
    combine nearby fleet, if the veh number of fleet are too small.
    :param dic_dpt_type: dic_dpt_type = {1: 'AH', 27: 'A', 77: 'AHHH', ...}
    :param n: expect veh number of fleet
    :return: new_dic = {1: 'AHA', 53: 'AH', 77: 'AHHH'}
    '''
    # Initialize a new dictionary and temporary variables for the current team
    dic_dpt_type2 = {}
    # Get the keys from the dictionary and sort them
    keys = list(dic_dpt_type.keys())  # [13, 18, 39...]
    temp_team = dic_dpt_type[keys[0]]  # AH or A
    start_time = keys[0]
    current_vehicles = len(dic_dpt_type[keys[0]])  # 2 or 1?? vehicle number of this platoon
    for i in range(1, len(keys)):
        # Current key
        current_key = keys[i]
        next_vehicles = len(dic_dpt_type[current_key])  # next veh number

        # If the total number of vehicles in the current team and the next team is less than or equal to 3, merge the teams
        if current_vehicles + next_vehicles <= n:
            temp_team += dic_dpt_type[current_key]  # combine first value AH with second value A
            current_vehicles += next_vehicles
        else:
            # Otherwise, add the previous team to the new dictionary and reset temp_team and vehicle count
            dic_dpt_type2[start_time] = temp_team
            start_time = current_key
            temp_team = dic_dpt_type[current_key]
            current_vehicles = next_vehicles
    # Add the last combined team to the new dictionary
    dic_dpt_type2[start_time] = temp_team
    return dic_dpt_type2

def plot_departure_times2(dic, p, fr, st, seed, display=False, interval_per_vehicle=1):
    """
    updated241203, add interval text note
    Plot the departure times of platoons and their types,
    and visualize the distribution of fleet types with sorted x-axis.
    Includes head-to-tail intervals between platoons.

    Parameters:
        dic (dict): A dictionary where keys are departure times and values are platoon types.
        interval_per_vehicle (int): Time interval (in seconds) between vehicles in a platoon.
    """
    # define plot length according to av_p
    if p == 0.3:
        plot_length = 24
    elif p == 0.2:
        plot_length = 19
    elif p == 0.1:
        plot_length = 14

    # Extract keys and values from the dictionary
    times = list(dic.keys())
    types = list(dic.values())

    # get real vehicle number and real av_p from dic
    av_count = sum(v.count('A') for v in types if v != 'h')  # *updated for 'h'*
    hv_count = sum(v.count('H') for v in types if v != 'h') + types.count('h')  # *updated for 'h'*
    av_number = int(av_count)
    veh_number = av_number + hv_count
    av_p = av_number/veh_number
    # targe veh number and av_p
    target_number = int((fr*st)/3600)
    target_av_number = int(p*target_number)
    target_av_p = p

    # Calculate the number of platoons
    n = len(dic)
    # Generate y-coordinates
    y = list(range(1, n + 1))

    # Calculate head-to-tail intervals
    intervals = []
    for i in range(len(times)):
        if type[i] == 'h':
            intervals.append("h")
        elif i == 0:
            intervals.append("N/A")  # First platoon has no previous platoon
        else:
            prev_time = times[i - 1]
            prev_length = len(types[i - 1])  # Length of the previous platoon
            current_time = times[i]
            intervals.append(current_time - (prev_time + (prev_length - 1) * interval_per_vehicle))

    # Set the figure size for the whole plot
    fig, ax1 = plt.subplots(figsize=(8, plot_length), dpi=300)

    # Create the first plot (scatter plot of departure times)
    # ax1.scatter(y, times, marker='_', color='blue', alpha=0.7)

    # Add labels for each point
    for i, (t, tp, interval) in enumerate(zip(times, types, intervals)):
        if tp == 'h':  # *free_HV green point*
            ax1.scatter(y[i], t, marker='o', color='green', s=5)  # *added*
            ax1.text(y[i]+0.8, t, f"h ({t})", fontsize=3, ha='left', va='center', color='green')  # *added*
        else:
            ax1.scatter(y[i], t, marker='_', color='blue', s=80)
            ax1.text(y[i], t, f"{tp} ({t}, {interval})", fontsize=9, ha='left', va='center')

    # Add labels and title
    ax1.set_xlabel('Groups')
    ax1.set_ylabel('Departure time (s)')
    ax1.set_title('Platoon Groups and Departure Times with Head-Tail Intervals')
    ax1.grid(True)

    # -----------ax2-----------
    # Now create the inset for the second plot (bar chart of fleet type distribution)
    ax2 = ax1.inset_axes([0.19, 0.67, 0.28, 0.28])  # [left, bottom, width, height] - relative to ax1
    # Count occurrences of each type
    type_counts = Counter(types)
    # Sort the platoon types in ascending order
    # sorted_types = sorted(type_counts.keys())
    sorted_types = dict(sorted(type_counts.items(), key=lambda x: len(x[0])))
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

    # -----------ax3-----------
    # Initialize result dictionary
    converted = defaultdict(int)
    for platoon, count in sorted_types.items():
        if platoon == 'h':
            converted['freeHV_num'] += count
        elif platoon == 'A':
            converted['1 lead 0'] += count
        elif platoon in ['AH', 'AA']:
            converted['1 lead 1'] += count
        else:
            # All others assumed to start with 1 AV and len-1 followers
            lead_key = f"1 lead {len(platoon) - 1}"
            converted[lead_key] += count
    # Convert defaultdict to regular dict
    converted = dict(converted)
    y = range(len(converted))
    x = list(converted.values())
    ax3 = ax1.inset_axes([0.65, 0.85, 0.14, 0.14])
    bars = ax3.barh(y, x, color='red', alpha=0.7, edgecolor='black')
    ax3.set_yticks(y)
    ax3.set_yticklabels(converted.keys())
    # Add labels and title to the inset
    ax3.set_xlabel('Count')
    ax3.set_title('Distribution of Fleet Types2')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.title(f'Info: avp:{p}, flow_rate:{fr}, seed:{seed}')

    custom_text = f"Target: {target_av_number}/{target_number} ({target_av_p:.2f})\nResult: {av_number}/{veh_number} ({av_p:.2f})"
    plt.text(max(y) - max(y)/3, min(times) + 2, custom_text, fontsize=12, color='blue',
             bbox=dict(facecolor='white', alpha=0.5))
    if display:
        plt.show()
    else:
        plt.savefig(f"/home/zzha/PycharmProjects/RampMerging3/picture&video/platoons_generation/{p, fr, st, seed}.png",
                    dpi=150, bbox_inches='tight')

def plot_departure_times3(dic, p, fr, st, seed, display=False, interval_per_vehicle=1):
    """
    Plot departure times of platoons and free HVs, with fleet type distributions.

    Parameters:
        dic (dict): departure_time -> platoon type (e.g., 'AHH', 'h')
        p (float): AV percentage
        fr (int): flow rate (veh/h)
        st (int): simulation time (s)
        seed (int): random seed used
        display (bool): whether to show the plot or save it
        interval_per_vehicle (int): time gap between vehicles in same platoon
    """
    # ----- Plot height depending on AV percentage -----
    plot_length = {0.3: 24, 0.2: 19, 0.1: 14}.get(p, 18)

    # ----- Extract and sort data -----
    times = sorted(dic.keys())
    types = [dic[t] for t in times]

    # ----- Count vehicles -----
    av_count = sum(v.count('A') for v in types if v != 'h')
    hv_count = sum(v.count('H') for v in types if v != 'h') + types.count('h')
    av_number = av_count
    veh_number = av_number + hv_count
    av_p = av_number / veh_number
    target_number = int((fr * st) / 3600)
    target_av_number = int(p * target_number)

    y = list(range(1, len(times) + 1))

    # ----- Compute head-to-tail intervals -----
    intervals = []
    for i in range(len(times)):
        if types[i] == 'h':
            intervals.append("h")
        elif i == 0:
            intervals.append("N/A")
        else:
            j = i - 1
            while j >= 0 and types[j] == 'h':
                j -= 1
            if j >= 0:
                prev_time = times[j]
                prev_len = len(types[j])
                intervals.append(times[i] - (prev_time + (prev_len - 1) * interval_per_vehicle))
            else:
                intervals.append("N/A")

    # ----- Main plot -----
    fig, ax1 = plt.subplots(figsize=(8, plot_length), dpi=300)

    for i, (t, tp, interval) in enumerate(zip(times, types, intervals)):
        if tp == 'h':
            ax1.scatter(y[i], t, marker='o', color='green', s=5)
            ax1.text(y[i] + 0.8, t, f"h ({t})", fontsize=3, ha='left', va='center', color='green')
        else:
            ax1.scatter(y[i], t, marker='_', color='blue', s=80)
            ax1.text(y[i], t, f"{tp} ({t}, {interval})", fontsize=9, ha='left', va='center')

    ax1.set_xlabel('Groups')
    ax1.set_ylabel('Departure time (s)')
    ax1.set_title('Platoon Groups and Departure Times with Head-Tail Intervals')
    ax1.grid(True)

    # ----- Inset plot 1: raw fleet type count -----
    ax2 = ax1.inset_axes([0.19, 0.70, 0.28, 0.28]) # [left, bottom, width, height] - relative to ax1
    type_counts = Counter(types)
    sorted_types = dict(sorted(type_counts.items(), key=lambda x: len(x[0])))
    indices_sorted = range(len(sorted_types))
    counts_sorted = list(sorted_types.values())
    bars2 = ax2.barh(indices_sorted, counts_sorted, color='blue', alpha=0.7, edgecolor='black')
    ax2.set_yticks(indices_sorted)
    ax2.set_yticklabels(sorted_types.keys())
    ax2.set_xlabel('Count')
    ax2.set_title('Raw Fleet Types')
    for bar in bars2:
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{int(bar.get_width())}', va='center', fontsize=10)

    # ----- Inset plot 2: mapped to '1 lead N' format -----
    converted = defaultdict(int)
    for platoon, count in sorted_types.items():
        if platoon == 'h':
            converted['freeHV_num'] += count
        elif platoon == 'A':
            converted['1 lead 0'] += count
        elif platoon in ['AH', 'AA']:
            converted['1 lead 1'] += count
        else:
            converted[f"1 lead {len(platoon) - 1}"] += count

    ax3 = ax1.inset_axes([0.65, 0.85, 0.10, 0.10])
    y2 = range(len(converted))
    x2 = list(converted.values())
    bars3 = ax3.barh(y2, x2, color='red', alpha=0.7, edgecolor='black')
    ax3.set_yticks(y2)
    ax3.set_yticklabels(converted.keys())
    ax3.set_xlabel('Count')
    ax3.set_title('Mapped Fleet Types')
    for bar in bars3:
        ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{int(bar.get_width())}', va='center', fontsize=10)

    # ----- Summary text -----
    summary = (f"Target: {target_av_number}/{target_number} ({p:.2f})\n"
               f"Result: {av_number}/{veh_number} ({av_p:.2f})")
    ax1.text(max(y) - max(y)/3, min(times) + 2, summary, fontsize=10, color='blue',
             bbox=dict(facecolor='white', alpha=0.5))

    # ----- Show or save -----
    plt.tight_layout()
    plt.title(f'Info: avp:{p}, flow_rate:{fr}, seed:{seed}')
    if display:
        plt.show()
    else:
        plt.savefig(
            f"/home/zzha/PycharmProjects/RampMerging3/picture&video/platoons_generation/{p, fr, st, seed}.png",
            dpi=150, bbox_inches='tight'
        )
    plt.close(fig)


def assign_followerAV(dic_dpt_type2, followerAV_num, seed=None):
    """
    Randomly assign available follower AVs into platoons.

    Parameters:
        dic_dpt_type2 (dict): {departure_time: 'AHHHH'}, platoon structure.
        followerAV_num (int): Total number of AVs that can act as followers.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        dict: Updated platoon types with some 'H' replaced by 'A' (followers).
    """
    if seed is not None:
        random.seed(seed)

    # Step 1: Collect all candidate follower positions across platoons
    candidate_positions = []  # List of (departure_time, index_in_string)

    for time, platoon in dic_dpt_type2.items():
        for idx, ch in enumerate(platoon[1:], start=1):  # Skip leader at idx 0
            if ch == 'H':
                candidate_positions.append((time, idx))

    # Step 2: Randomly choose positions to replace with AVs
    chosen_positions = set(random.sample(candidate_positions, min(followerAV_num, len(candidate_positions))))

    # Step 3: Build updated platoon dict
    updated_dic = {}
    for time, platoon in dic_dpt_type2.items():
        new_chars = list(platoon)
        for idx in range(1, len(platoon)):
            if (time, idx) in chosen_positions:
                new_chars[idx] = 'A'
        updated_dic[time] = ''.join(new_chars)
    return updated_dic

def assign_free_HV(dic_dpt_type, free_hv_num, min_gap=2, seed=None):
    """
    Randomly insert free HVs ('h') between platoons, respecting min_gap constraints.

    Parameters:
        dic_dpt_type (dict): e.g., {3: 'AHH', 14: 'AHHH'} — platoon departure times and structure.
        free_hv_num (int): Number of free HVs to insert.
        min_gap (int): Minimum time (in seconds) from any platoon vehicle.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Updated departure time dictionary with free HVs inserted.
    """
    if seed is not None:
        random.seed(seed)

    # Step 1: sort platoons by time and record occupied vehicle times
    sorted_items = sorted(dic_dpt_type.items())
    occupied_times = set()
    platoon_intervals = []

    for start_time, p_type in sorted_items:
        length = len(p_type)
        vehicle_times = [start_time + i for i in range(length)]  # assuming 1s interval
        occupied_times.update(vehicle_times)
        platoon_intervals.append((start_time, start_time + length - 1))  # (head, tail)

    # Step 2: collect candidate times between platoons
    free_slots = []

    # Check gaps between platoons
    for i in range(len(platoon_intervals) - 1):
        _, prev_tail = platoon_intervals[i]
        next_head, _ = platoon_intervals[i + 1]

        start = prev_tail + min_gap
        end = next_head - min_gap

        for t in range(start, end + 1):
            if t not in occupied_times:
                free_slots.append(t)

    # Check before the first platoon
    first_start, _ = platoon_intervals[0]
    for t in range(0, first_start - min_gap):
        if t not in occupied_times:
            free_slots.append(t)

    # Check after the last platoon (up to 60s buffer)
    _, last_end = platoon_intervals[-1]
    for t in range(last_end + min_gap, last_end + 60):
        if t not in occupied_times:
            free_slots.append(t)

    # Step 3: randomly select insertion times
    if free_hv_num > len(free_slots):
        print(f"Warning: Only {len(free_slots)} feasible slots available, but {free_hv_num} free HVs requested.")
        free_hv_num = len(free_slots)

    selected_times = random.sample(free_slots, free_hv_num)

    # Step 4: update dictionary
    updated_dic = dic_dpt_type.copy()
    for t in selected_times:
        updated_dic[t] = 'h'

    return dict(sorted(updated_dic.items()))

def count_vehicle_types(dic):
    """
    Count and print AVs, HVs, AV leaders, AV followers, and free HVs from platoon dictionary.

    Parameters:
        dic (dict): {departure_time: 'AHH', 18: 'h', ...}
    """
    total_AV = 0
    total_HV = 0
    AV_leader = 0
    AV_follower = 0
    free_HV = 0

    for v in dic.values():
        if v == 'h':
            free_HV += 1
        else:
            total_AV += v.count('A')
            total_HV += v.count('H')
            AV_leader += 1
            AV_follower += v[1:].count('A')  # Only followers counted

    total_HV += free_HV  # Add free HVs to total HV count

    print(f"Total AV:        {total_AV}")
    print(f"Total HV:        {total_HV}")
    print(f" - AV leaders:   {AV_leader}")
    print(f" - AV followers: {AV_follower}")
    print(f" - Free HVs:     {free_HV}")

def get_schedule2(st, av_p, fr, platoon_p=1, max_attempts=5, plot=False, seed=None, display=False):
    """
    combine above function
    :param st:
           av_p:
           fr:
           platoon_p: default 1
           dic_tn: dict of platoon type and number
    :return: dic of departure_time and fleet_type => {t1:'h', t2:'AAH', ...}; h=>free_HV
    """
    if fr == 0:
        return {}
    dic_tn, followerAV_num, freeHV_num = generate_type_num4(av_percentage=av_p, flow_rate=fr, simulation_time=st,
                                                            platoon_percentage=platoon_p, seed=seed)
    max_interval = find_max_interval2(st, dic_tn)
    dp_times_dict, dp_times = find_optimse_schedule2(dic_tn, st, max_interval, \
                                                     max_attempts=max_attempts, seed=seed)
    dic_dpt_type = convert2(dp_times_dict, dp_times)
    dic_dpt_type2 = recombine(dic_dpt_type) # dic_departure_time and fleet type
    dic_dpt_type3 = assign_followerAV(dic_dpt_type2, followerAV_num, seed) # assign followerAV
    dic_dpt_type4 = assign_free_HV(dic_dpt_type3, freeHV_num, min_gap=2, seed=seed)
    count_vehicle_types(dic_dpt_type4)
    if plot:
        plot_departure_times3(dic_dpt_type4, av_p, fr, st, seed, display) # after reorganise
    # print(f'dic_dpt_type:{dic_dpt_type4}')
    return dic_dpt_type4

def get_schedule_dynamic(st, change_t, av_p, fr, platoon_p=1, max_attempts=5,
                        plot=False, seed=None, display = False):
    '''
    100324updated: from set start time generate vehicle
    :param self:
    :param change_t: time to change traffic demands
    :param av_p:
    :param fr:
    :param platoon_p:
    :param max_attempts:
    :param plot:
    :param seed:
    :return:
    '''
    real_st = st - change_t # the total time for vehicle generation
    dic_tn, followerAV_num, freeHV_num = generate_type_num4(av_percentage=av_p, flow_rate=fr,
                                                            simulation_time=real_st,
                                                            platoon_percentage=platoon_p,
                                                            seed=seed)
    max_interval = find_max_interval2(real_st, dic_tn)
    dp_times_dict, dp_times = find_optimse_schedule2(dic_tn, real_st, max_interval, \
                                                    max_attempts=max_attempts, seed=seed)
    dic_dpt_type = convert2(dp_times_dict, dp_times)
    dic_dpt_type2 = recombine(dic_dpt_type)  # dic_departure_time and fleet type
    dic_dpt_type3 = assign_followerAV(dic_dpt_type2, followerAV_num, seed)  # assign followerAV
    dic_dpt_type4 = assign_free_HV(dic_dpt_type3, freeHV_num, min_gap=2, seed=seed)
    count_vehicle_types(dic_dpt_type4)
    updated_result = {key + change_t: value for key, value in dic_dpt_type4.items()}
    print(f'new_dic_dpt_type:{updated_result}')
    return updated_result

def get_schedule_HVonly(simulation_time, flow_rate, seed=None):
    """
    Generate a dictionary of unique departure times for free HVs based on flow rate.

    Parameters:
    simulation_time (int): Total simulation time in seconds.
    flow_rate (float): Vehicle flow rate in vehicles per hour.
    seed (int): Optional random seed for reproducibility.

    Returns:
    dict: A dictionary in the form {time: 'h', ...}, where time is in seconds.
    """
    if seed is not None:
        np.random.seed(seed)

    total_vehicles = int(flow_rate * (simulation_time / 3600))

    # Generate inter-arrival times using exponential distribution
    intervals = np.random.exponential(scale=(3600 / flow_rate), size=total_vehicles)
    intervals += np.random.uniform(0, 0.1, size=total_vehicles)  # Small perturbation

    departure_times = np.cumsum(intervals)
    departure_times = np.round(departure_times).astype(int)
    departure_times = departure_times[departure_times < simulation_time]

    # Ensure unique times
    unique_departure_times = []
    last_time = -1
    for t in departure_times:
        if t <= last_time:
            t = last_time + 1
        unique_departure_times.append(t)
        last_time = t

    # Build dictionary {time: 'h'}
    dep_dict = {t: 'h' for t in unique_departure_times}
    return dep_dict

def get_schedule_motorway(st, p, fr, seed=None): # generate_entry_arrivals_poisson
    """
    250209 random traffic on motorway
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
    lam = fr / 3600  # lam=>λ; Calculate the arrival rate (vehicles per second)
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

class VehGen:
    """
    generate vehicle according to schedule dict
    """
    def __init__(self, traci):
        self.traci = traci

    def add_vehicle_ml(self, vehicle_id, route_id, dp_v, dp_lane, type_id="idm"):
        '''
        ml: multi-lane
        :param vehicle_id:
        :param route_id:
        :param dp_v: depature speed (velocity)
        :param dp_lane:
        :param type_id:
        :return:
        '''
        self.traci.vehicle.add(vehID=vehicle_id, routeID=route_id, typeID=type_id, \
                               departSpeed=dp_v, departLane=dp_lane)
        self.traci.vehicle.setSpeedMode(vehicle_id, 0b010111)

    def add_vehicle(self, vehicle_id, route_id, dp_v, type_id="idm"):
        self.traci.vehicle.add(vehID=vehicle_id, routeID=route_id, typeID=type_id, \
                               departSpeed=dp_v)
        if route_id == 'route1': # mainline vehicles
            # Set moderate cooperation level (won't actively give way, but avoids collisions)
            self.traci.vehicle.setParameter(vehicle_id, "lcCooperative", "0.2")
            # High assertiveness: vehicle keeps its lane unless strongly motivated to change
            self.traci.vehicle.setParameter(vehicle_id, "lcAssertive", "0.9")
            # self.traci.vehicle.setSpeedMode(vehicle_id, 0b10110011 & ~(1 << 3))  # 即 bit3 = 0

        else: # ramp vehicles
            pass
        # set speedmode
        self.traci.vehicle.setSpeedMode(vehicle_id, 0b010111)  # ingnore the right of way
        # self.traci.vehicle.setSpeedMode(vehicle_id, 0b000111)  # ingnore brake before red, the right of way
        # self.traci.vehicle.setSpeedMode(vehicle_id, 0b011111) # default, consider all inspection

    # step = 0,1,2,3,4,5
    # r_step = 0,0.1,0.2,0.3...
    # veh_route = 'route2' or 'route1'
    # id_prefix = 'r' or 'm'
    def veh_gen5(self, step, dp_times_dict, id_prefix, \
                 veh_route, dp_v, vType='idm'):
        '''
        240616: delete data recording as these can get from the departure information
        240614: new input format, dic =  {1: 'AHA', 31: 'AH', 48: 'AHA', 79: 'AA', 107: 'AH', 121: 'AHA'}
        240609: update from dic_rpav_type to dic_av_type
        generate veh at cooresponding step

        Parameters
        ----------
        dp_times_dict : dic
            departure times of each fleet type.
            {4: 'AHHHH', 29: 'AHHHH', 45: 'AHHHHHH', 61: 'AHHHH', 73: 'AHH}
        id_prefix : TYPE
            DESCRIPTION.
        veh_route : TYPE
            DESCRIPTION.
        dp_v : TYPE
            departSpeed.
        vType: default idm
            vehicle type: IDM; Krauss; Krauss0.2; Krauss0.8 ...

        Returns
        -------
        None.

        '''
        for dt, type in dp_times_dict.items():
            r_step = step/10 # r_step = time
            veh_num = len(type) # the veh number of this platoon
            # type = 'AHA'
            for i, veh in enumerate(type):
                dt_v = dt+i # departure time of this veh
                if dt_v == r_step:
                    if i==0:
                        id_body = 'avh' # head AV => avh
                    else:
                        id_body = 'av' if veh == 'A' else 'hv'
                    vehicle_type = 'av' if veh=='A' else vType
                    id_suffix = id_prefix + id_body
                    id = f"{id_suffix}{step}"
                    if id == 'rhv4030':
                        pass
                    self.add_vehicle(id, veh_route, dp_v, vehicle_type)

    def veh_gen_hv(self, step, dp_times_dict, id_prefix, \
                 veh_route, dp_v):
        '''
        all hv
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
                dt_v = dt+i # departure time of this veh
                if dt_v == r_step:
                    if i==0:
                        id_body = 'avh' # head AV => avh
                    else:
                        id_body = 'av' if veh == 'A' else 'hv'
                    vehicle_type = 'idm' if veh=='A' else 'idm'
                    id_suffix = id_prefix + id_body
                    id = f"{id_suffix}{step}"
                    if id == 'rhv4030':
                        pass
                    self.add_vehicle(id, veh_route, dp_v, vehicle_type)

    def veh_gen_special(self, step, dp_times_dict, veh_route='route1', dp_v=24.5):
        '''
        all hv
        Parameters
        ----------
        dp_times_dict : dic
            departure times of each veh type.
            {4: 'idm_nosigam', 29: 'idm_sigma', 45: 'krauss', 61: 'krauss_sigma', 73: 'idm'}
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
            # type = 'idm' or 'krauss'
            if dt == r_step:
                vehicle_type = type
                id = f"{type}{step}"
                self.add_vehicle(id, veh_route, dp_v, vehicle_type)

    def veh_gen_heter(self, step, dp_times_dict, id_prefix, p_auto):
        '''
        Add HV heterogeneity
        250610: veh_route and dp_v can be decided by id_prefix, then only keep them inside function
        240616: delete data recording as these can get from the departure information
        240614: new input format, dic =  {1: 'AHA', 31: 'AH', 48: 'AHA', 79: 'AA', 107: 'AH', 121: 'AHA'}
        240609: update from dic_rpav_type to dic_av_type
        generate veh at cooresponding step

        Parameters
        ----------
        dp_times_dict : dic
            departure times of each fleet type.
            {4: 'AHHHH', 29: 'AHHHH', 45: 'AHHHHHH', 61: 'AHHHH', 73: 'AHH}
        id_prefix : TYPE
            DESCRIPTION.
        vType: default idm
            vehicle type: IDM; Krauss; Krauss0.2; Krauss0.8 ...
        p_auto:
            pecentage of hv auto-following (0-1)
        Returns
        -------
        None.

        '''
        # get the probability of different HV following type
        if p_auto == 0:
            dic_prob = {'hv_cons': 0.51, 'hv_avg': 0.32, 'hv_agg': 0.17}  # probability
        else:
            p_hvAuto = p_auto
            p_hvCons = (1 - p_auto) * 0.51
            p_hvAvg = (1 - p_auto) * 0.32
            p_hvAgg = (1 - p_auto) * 0.17
            dic_prob = {'hv_cons': p_hvCons, 'hv_avg': p_hvAvg, 'hv_agg': p_hvAgg, 'idm': p_hvAuto}

        # according to departure lane get route and departure speed
        if id_prefix == 'm':
            veh_route = 'route1'
            dp_v = 24.5  # depature velocity
        else:
            veh_route = 'route2'
            dp_v = 10
        for dt, type in dp_times_dict.items():
            r_step = step / 10  # r_step = time
            veh_num = len(type)  # the veh number of this platoon
            # type = 'AHA'
            for i, veh in enumerate(type):
                dt_v = dt + i  # departure time of this veh
                if dt_v == r_step:
                    if i == 0:
                        id_body = 'avh'  # head AV => avh
                    else:
                        id_body = 'av' if veh == 'A' else 'hv'

                    # vehicle_type = 'av' if veh == 'A' else vType
                    if veh == 'A':
                        vehicle_type = 'av'
                    else:
                        # Probabilistically select HV type based on predefined distribution
                        vehicle_type = random.choices(
                            list(dic_prob.keys()),  # Possible HV types
                            weights=list(dic_prob.values()),  # Corresponding probabilities
                            k=1  # Select 1 item
                        )[0]  # Extract the selected type from the list

                    id_suffix = id_prefix + id_body
                    id = f"{id_suffix}{step}"
                    self.add_vehicle(id, veh_route, dp_v, vehicle_type)

    def get_schedule_startT(start_t, p, fr, max_attempts, plot=False, seed=None, display=False):
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
            plot_departure_times(updated_result, display)
        print(f'dic_dpt_type:{updated_result}')
        return updated_result

    def veh_gen_heter2(self, step, dp_times_dict, id_prefix, p_auto):
        '''
        Add free_HV
        Add HV heterogeneity
        250610: veh_route and dp_v can be decided by id_prefix, then only keep them inside function
        240616: delete data recording as these can get from the departure information
        240614: new input format, dic =  {1: 'AHA', 31: 'AH', 48: 'AHA', 79: 'AA', 107: 'AH', 121: 'AHA'}
        240609: update from dic_rpav_type to dic_av_type
        generate veh at cooresponding step

        Parameters
        ----------
        dp_times_dict : dic
            departure times of each fleet type.
            {4: 'AHHHH', 29: 'h', 45: 'AHHHHHH', 61: 'AHHHH', 73: 'AHH}
        id_prefix : TYPE
            DESCRIPTION.
        p_auto:
            pecentage of hv auto-following (0-1)
        Returns
        -------
        None.

        '''
        # get the probability of different HV following type
        if p_auto == 0:
            dic_prob = {'hv_cons': 0.51, 'hv_avg': 0.32, 'hv_agg': 0.17}  # probability
        else:
            p_hvAuto = p_auto
            p_hvCons = (1 - p_auto) * 0.51
            p_hvAvg = (1 - p_auto) * 0.32
            p_hvAgg = (1 - p_auto) * 0.17
            dic_prob = {'hv_cons': p_hvCons, 'hv_avg': p_hvAvg, 'hv_agg': p_hvAgg, 'idm': p_hvAuto}

        # according to departure lane get route and departure speed
        if id_prefix == 'm':
            veh_route = 'route1'
            dp_v = 24.5  # depature velocity
        else:
            veh_route = 'route2'
            dp_v = 10

        for dt, platoon in dp_times_dict.items():
            r_step = step / 10
            veh_num = len(platoon)

            for i, veh in enumerate(platoon):
                dt_v = dt + i  # actual vehicle departure time
                if dt_v == r_step:
                    # ---- Determine ID suffix ----
                    if platoon == 'h':  # special case: free HV follower
                        id_body = 'hv'
                        vehicle_type = random.choices(
                            list(dic_prob.keys()),
                            weights=list(dic_prob.values()),
                            k=1
                        )[0]
                        id_suffix = f"{id_prefix}{id_body}{step}"
                        self.add_vehicle(id_suffix, veh_route, dp_v, vehicle_type)
                        self.traci.vehicle.setColor(id_suffix, (150, 150, 150, 255)) # gray
                        # self.traci.vehicle.setSpeedMode(id_suffix, 0b011111) # set as default
                        continue  # only one vehicle, no need to continue inner loop

                    # ---- Normal platoon ----
                    if i == 0:
                        id_body = 'avh'  # leader AV
                        vehicle_type = 'av'
                    else:
                        if veh == 'A':
                            id_body = 'av'
                            vehicle_type = 'av'
                        else:
                            id_body = 'hv'
                            vehicle_type = random.choices(
                                list(dic_prob.keys()),
                                weights=list(dic_prob.values()),
                                k=1
                            )[0]

                    id_suffix = f"{id_prefix}{id_body}{step}"
                    self.add_vehicle(id_suffix, veh_route, dp_v, vehicle_type)

    def veh_gen_ml(self, step, dp_times_dict, id_prefix, \
                   veh_route, dp_v, dp_lane='0'):
        '''
        ml: multi-lane
        2450209: new input format, dic =  {1: 'HV', 31: 'AV', ...,}
        generate veh at coorresponding step

        Parameters
        ----------
        dp_times_dict : dic
            departure times of each veh.
            {1: 'HV', 31: 'AV', ...,}
        id_prefix : TYPE
            DESCRIPTION.
        veh_route : TYPE
            DESCRIPTION.
        dp_v : TYPE
            departSpeed.
        dp_lane:
            departure lane

        Returns
        -------
        None.

        '''
        c_ts = step/10 # r_step = time
        if c_ts in dp_times_dict:
            type = dp_times_dict[c_ts]
            vehicle_type = 'av' if dp_times_dict[c_ts] == 'AV' else 'hv'
            mode = 'av' if dp_times_dict[c_ts] == 'AV' else 'idm'
            if dp_lane == '1':
                id = f"{id_prefix}b{vehicle_type}{step}" # use b indicate lane_1
            else:
                id = f"{id_prefix}{vehicle_type}{step}"
            if id == 'mhv18':
                pass
            self.add_vehicle_ml(id, veh_route, dp_v, dp_lane, mode)

    def get_avhid_ptype(self, m_dpt_type=None, r_dpt_type=None):
        '''
         get av head id and it's platoon type

        :param m_dpt_type: {4: 'AHHHHHHHHH', 31: 'AHHHHHHHHHHH'}
        :param r_dpt_type:
        :return:  {'mavh40': 'AHHHHHHHHH', 'mavh310': 'AHHHHHHHHHHH', 'mavh580': 'AHHHHHHHH'}
        '''
        dic_avhid_ptype = {}
        if m_dpt_type:
            for key, value in m_dpt_type.items():
                id1 = 'mavh' + str(key*10)
                dic_avhid_ptype[id1] = value
        if r_dpt_type:
            for key, value in r_dpt_type.items():
                id2 = 'ravh' + str(key*10)
                dic_avhid_ptype[id2] = value
        return dic_avhid_ptype


if __name__ == '__main__':
    st = 1200
    av_p = 0.3
    fr = 1080  # 360
    platoon_p = 1
    max_attempts = 7
    seed = 4
    plot = True
    display = True
    dic_dpt_type = get_schedule2(st, av_p, fr, platoon_p, max_attempts, plot=plot, seed=seed, display=True)


    # start_t = 600
    # new_fr = 1080
    # new_dic = get_schedule_startT(start_t, p, fr, max_attempts, plot=True, seed=1)






