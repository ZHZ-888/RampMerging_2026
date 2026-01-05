#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:19:17 2024

@author: zzha
"""

'''
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
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


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
        return {f"1 lead {i}": 0 for i in range(9)}
    elif p == 0.2:
        return {f"1 lead {i}": 0 for i in range(7)} # 7
    elif p == 0.3:
        return {f"1 lead {i}": 0 for i in range(5)}
    else:
        raise ValueError("Unsupported AV penetration rate.")
    return r

def generate_type_num2(percentage, flow_rate, simulation_time, seed=None):
    # Step 1: Calculate total vehicle numbers
    total_veh_num = int((flow_rate * simulation_time) / 3600)
    av_num = int(total_veh_num * percentage)
    hv_num = total_veh_num - av_num

    # Step 2: Get allowed platoon types from vdp
    platoon_types = vdp(percentage)  # Call the vdp function
    allowed_types = list(platoon_types.keys())

    # Step 3: Define priority platoon type
    priority_mapping = {
        0.1: "1 lead 9",
        0.15: "1 lead 6",
        0.2: "1 lead 4",
        0.3: "1 lead 2",
    }
    priority_type = priority_mapping.get(percentage, None)
    pt_hv_number = int(priority_type.split()[-1])  # Priority type HV number

    if priority_type not in allowed_types:
        raise ValueError(f"Priority platoon type {priority_type} not allowed for penetration rate {percentage}")

    # Initialize platoon distribution
    platoon_distribution = {ptype: 0 for ptype in allowed_types}

    # Step 4: Randomize the percentage for priority platoon type (between 65% to 85%)
    if seed is not None:
        random.seed(seed)
    priority_percentage = random.uniform(0.65, 0.85)
    priority_platoon_number = int(av_num * priority_percentage)  # Priority platoon number = priority_av_number
    priority_hv_number = pt_hv_number * priority_platoon_number

    # Update priority platoon distribution
    platoon_distribution[priority_type] = priority_platoon_number

    # Step 5: Obtain remaining AV/HV numbers
    remaining_av_num = av_num - priority_platoon_number
    remaining_hv_num = hv_num - priority_hv_number

    # Optimization parameters
    max_iterations = 1000
    types_to_distribute = [t for t in allowed_types if t != priority_type]
    type_hv_counts = [int(t.split()[-1]) for t in types_to_distribute]

    # Initialize variables to track the best solution
    best_solution = None
    smallest_difference = float('inf')

    # Loop through multiple seeds to find the optimal solution
    for iteration in range(max_iterations):
        # Use a different random seed for each iteration
        np.random.seed(seed + iteration)

        # Calculate weights inversely proportional to the extremity of the type
        weights = np.array([1 / (1 + abs(hv_count - 5.5)) for hv_count in type_hv_counts])
        weights /= weights.sum()  # Normalize weights

        # Allocate AVs using multinomial distribution
        av_distribution = np.random.multinomial(remaining_av_num, weights)

        # Create AV distribution dictionary
        av_distribution_dict = {t: av for t, av in zip(types_to_distribute, av_distribution)}

        # Calculate HV distribution
        hv_distribution = [av * hv_count for av, hv_count in zip(av_distribution, type_hv_counts)]
        total_av = sum(av_distribution_dict.values())
        total_hv = sum(hv_distribution)

        # Calculate differences
        av_difference = abs(remaining_av_num - total_av)
        hv_difference = abs(remaining_hv_num - total_hv)
        total_difference = av_difference + hv_difference  # Combined difference metric

        # Check if this is the best solution so far
        if total_difference < smallest_difference:
            smallest_difference = total_difference
            best_solution = {
                "iteration": iteration,
                "seed": seed + iteration,
                "av_distribution": av_distribution_dict,
                "total_av": total_av,
                "total_hv": total_hv,
                "av_diff": av_difference,
                "hv_diff": hv_difference
            }

            # Stop if we find a perfect solution
            if av_difference == 0 and hv_difference == 0:
                break

    # Merge the priority platoon into the final distribution
    best_distribution = best_solution['av_distribution']
    best_distribution[priority_type] = priority_platoon_number

    # Sort the final distribution by platoon type (n ascending)
    sorted_distribution = dict(sorted(best_distribution.items(), key=lambda x: int(x[0].split()[-1])))
    print(sorted_distribution)
    # Print final comparison results
    print("Final Comparison:")
    result_av_num = sum(sorted_distribution.values())
    print(f"Target AVs: {av_num}, Allocated AVs: {result_av_num}, Difference: {av_num - result_av_num}")
    result_hv_num = sum(int(k.split()[-1]) * v for k, v in sorted_distribution.items())
    print(f"Target HVs: {hv_num}, Allocated HVs: {result_hv_num}, Difference: {hv_num - result_hv_num}")
    result_total_veh_num = result_hv_num + result_av_num
    print(f"Target av_p: {av_num / total_veh_num:.2f}, av_p: {result_av_num / result_total_veh_num:.2f}")

    return sorted_distribution

def generate_type_num2_241205(percentage, flow_rate, simulation_time, seed=None):
    """
    Generate a platoon sequence prioritizing specific platoon types based on AV penetration rate,
    while reserving a flexible percentage of vehicles for other platoon types.

    Parameters:
    percentage (float): AV penetration rate (e.g., 0.1 for 10%).
    flow_rate (int): Traffic flow rate in vehicles per hour.
    simulation_time (int): Total simulation time in seconds.
    seed (int): Random seed for reproducibility.

    Returns:
    dict: Platoon distribution after prioritization and adjustment.
    """
    if seed is not None:
        random.seed(seed)

    # Step 1: Calculate total vehicle numbers
    total_veh_num = int((flow_rate * simulation_time) / 3600)
    av_num = int(total_veh_num * percentage)
    hv_num = total_veh_num - av_num

    # Step 2: Get allowed platoon types from vdp
    platoon_types = vdp(percentage)
    allowed_types = list(platoon_types.keys())

    # Step 3: Define priority platoon type
    priority_mapping = {
        0.1: "1 lead 9",
        0.15: "1 lead 6",
        0.2: "1 lead 4",
        0.3: "1 lead 2",
    }
    priority_type = priority_mapping.get(percentage, None)
    if priority_type not in allowed_types:
        raise ValueError(f"Priority platoon type {priority_type} not allowed for penetration rate {percentage}")

    # Initialize platoon distribution
    platoon_distribution = {ptype: 0 for ptype in allowed_types}
    remaining_hv = hv_num
    sequence = []

    # Step 4: Randomize the percentage for priority platoon type (between 65% to 85%)
    priority_percentage = random.uniform(0.65, 0.85)
    priority_vehicle_limit = int(av_num * priority_percentage)

    # Step 5: Distribute vehicles to priority platoon type up to the randomized limit
    group_count = 0
    while group_count < priority_vehicle_limit and remaining_hv >= int(priority_type.split()[-1]):
        hv_count = int(priority_type.split()[-1])
        platoon_distribution[priority_type] += 1
        group_count += 1
        remaining_hv -= hv_count
        sequence.extend(["A"] + ["H"] * hv_count)

    # Step 6: Assign remaining platoons to reach the exact target group number
    other_types = [ptype for ptype in allowed_types if ptype != priority_type]
    random.shuffle(other_types)  # Shuffle to introduce randomness in selection

    while group_count < av_num:
        random_type = random.choice(other_types)
        hv_count = int(random_type.split()[-1])
        if remaining_hv >= hv_count:
            platoon_distribution[random_type] += 1
            group_count += 1
            remaining_hv -= hv_count
            sequence.extend(["A"] + ["H"] * hv_count)
        else:
            break  # Stop if not enough HVs left for this type

    # Ensure remaining AVs are added to match the target group number
    av_remaining = av_num - group_count
    sequence.extend(["A"] * av_remaining)
    group_count += av_remaining

    # Adjust sequence to match total vehicle count if needed
    final_vehicle_count = len(sequence)
    if final_vehicle_count > total_veh_num:
        # If we have extra vehicles, remove them
        excess = final_vehicle_count - total_veh_num
        sequence = sequence[:-excess]
    elif final_vehicle_count < total_veh_num:
        # If we have fewer vehicles, add necessary HVs to fill the gap
        missing = total_veh_num - final_vehicle_count
        sequence.extend(["H"] * missing)

    # Step 7: Calculate results and summarize
    # final_vehicle_count = len(sequence)
    final_vehicle_count = sum((1 + int(ptype.split()[-1])) * count for ptype, count in platoon_distribution.items())
    group_num = sum(platoon_distribution.values())
    target_group_num = av_num
    actual_percentage = group_num / final_vehicle_count
    target_percentage = percentage
    vehicle_count_diff = abs(total_veh_num - final_vehicle_count)

    summary = (
        f"Final platoon distribution: {platoon_distribution}\n"
        f"Total vehicles: {final_vehicle_count}, Target vehicles: {total_veh_num}, Vehicle count difference: {vehicle_count_diff}\n"
        f"Group number: {group_num}, Target group number: {target_group_num}\n"
        f"Percentage: {actual_percentage:.2f}, Target percentage: {target_percentage:.2f}\n"
        f"Priority platoon type: {priority_type} assigned {platoon_distribution[priority_type]} times."
    )
    print(summary)
    return platoon_distribution

def generate_type_num2_241203(percentage, flow_rate, simulation_time, seed=None):
    """
    Generate a platoon sequence with random AV intervals while maintaining overall consistency.
    Ensures AVs are distributed with reasonable random intervals, meeting platoon and vehicle count requirements.

    Parameters:
    percentage (float): AV penetration rate (e.g., 0.1 for 10%).
    flow_rate (int): Traffic flow rate in vehicles per hour.
    simulation_time (int): Total simulation time in seconds.
    seed (int): Random seed for reproducibility.

    Returns:
    str: Formatted summary of platoon distribution and comparison with target values.
    """
    if seed is not None:
        random.seed(seed)

    # Step 1: Calculate AV and HV numbers
    total_veh_num = int((flow_rate * simulation_time) / 3600)
    av_num = int(total_veh_num * percentage)
    hv_num = total_veh_num - av_num

    # Step 2: Generate random intervals for AVs
    # Base spacing for even distribution
    base_spacing = hv_num // av_num
    remaining_hv = hv_num - base_spacing * (av_num - 1)
    hv_intervals = [base_spacing] * (av_num - 1)

    # Introduce randomness in intervals
    for i in range(len(hv_intervals)):
        random_adjustment = random.randint(-2, 2)  # Small random offset
        hv_intervals[i] = max(1, hv_intervals[i] + random_adjustment)  # Ensure at least 1 HV between AVs

    # Adjust last interval to absorb any remaining HVs
    hv_intervals[-1] += remaining_hv

    # Step 3: Build the sequence
    sequence = []
    for interval in hv_intervals:
        sequence.append("A")
        sequence.extend(["H"] * interval)
    sequence.append("A")  # Add the last AV

    # Step 4: Adjust sequence to fit vdp-defined platoon types
    platoon_types = vdp(percentage)
    allowed_types = sorted(platoon_types.keys(), key=lambda x: int(x.split()[2]))
    max_size = int(allowed_types[-1].split()[2])

    platoon_distribution = {ptype: 0 for ptype in platoon_types}
    current_platoon = []
    for vehicle in sequence:
        current_platoon.append(vehicle)
        if vehicle == "A" or len(current_platoon) > max_size:
            hv_count = len(current_platoon) - 1
            ptype = f"1 lead {hv_count}"
            if ptype in platoon_types:
                platoon_distribution[ptype] += 1
            current_platoon = []

    # Step 5: Calculate final vehicle count
    final_vehicle_count = sum(
        (1 + int(ptype.split()[2])) * count for ptype, count in platoon_distribution.items()
    )
    vehicle_count_diff = total_veh_num - final_vehicle_count

    # Step 6: Summarize the results
    final_group_num = sum(platoon_distribution.values())
    summary = (
        f"Final platoon distribution: {platoon_distribution}\n"
        f"Final total vehicles: {final_vehicle_count}, Target total vehicles: {total_veh_num}\n"
        f"Final group number: {final_group_num}, Target group number: {av_num}\n"
        f"Final percentage: {final_group_num / final_vehicle_count}, Target percentage: {percentage}\n"
        f"Final vehicle count difference: {abs(vehicle_count_diff)}"

    )
    print(summary)
    return platoon_distribution



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


def convert2(dp_times_dict, dp_times):
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

def convert(dp_times_dict, dp_times):
    '''
    Updated 241128
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
        elif t in dp_times_dict['1 lead 6']:
            dic_time_type[t] = 'AHHHHHH'
        elif t in dp_times_dict['1 lead 7']:
            dic_time_type[t] = 'AHHHHHHH'
        elif t in dp_times_dict['1 lead 8']:
            dic_time_type[t] = 'AHHHHHHHH'
        elif t in dp_times_dict['1 lead 9']:
            dic_time_type[t] = 'AHHHHHHHHH'
        elif t in dp_times_dict['1 lead 10']:
            dic_time_type[t] = 'AHHHHHHHHHH'
        else:  # 1 lead 11
            dic_time_type[t] = 'AHHHHHHHHHHH'
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
    current_vehicles = len(dic_time_type[keys[0]])  # 2 or 1?? vehicle number of this platoon
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

def plot_departure_times_old(dic):
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

# updated241203, add interval text note
def plot_departure_times(dic, p, fr, st, seed, display=False, interval_per_vehicle=1):
    """
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

    # get real vehicle number and real av_p from dic
    av_count = 0
    hv_count = 0
    for composition in dic.values():
        av_count += composition.count('A')
        hv_count += composition.count('H')
    av_number = int(av_count)
    veh_number = int(av_count + hv_count)
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

    custom_text = f"Target: {target_av_number}/{target_number} ({target_av_p:.2f})\nResult: {av_number}/{veh_number} ({av_p:.2f})"
    plt.text(max(y) - max(y)/3, min(times) + 2, custom_text, fontsize=12, color='blue',
             bbox=dict(facecolor='white', alpha=0.5))
    if display:
        plt.show()
    else:
        plt.savefig(f"/home/zzha/PycharmProjects/RampMerging3/picture&video/platoons_generation/{p, fr, st, seed}.png",
                    dpi=150, bbox_inches='tight')


def plot_departure_times_old(dic, p, fr, seed):
    """
    Plot the departure times of platoons and their types,
    and visualize the distribution of fleet types with sorted x-axis.

    Parameters:
        dic (dict): A dictionary where keys are departure times and values are platoon types.
    """
    # Extract keys and values from the dictionary
    times = list(dic.keys())
    types = list(dic.values())

    # Calculate the number of platoons
    n = len(dic)

    # Generate y-coordinates
    y = list(range(1, n + 1))

    # Set the figure size for the whole plot
    fig, ax1 = plt.subplots(figsize=(8, 14))

    # Create the first plot (scatter plot of departure times)
    ax1.scatter(y, times, marker='_', color='blue', alpha=0.7)

    # Add labels for each point
    for i, (time, platoon_type) in enumerate(zip(times, types)):
        label = f"{platoon_type} ({time})"
        ax1.text(y[i], time, label, fontsize=9, ha='left', va='center')

    # Add labels and title
    ax1.set_xlabel('Departure time (s)')
    ax1.set_ylabel('Groups')
    ax1.set_title('Platoon Groups and Departure Times')
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
        xval = bar.get_width()  # get the width of columns
        ax2.text(xval + 0.2, bar.get_y() + bar.get_height() / 2, str(int(xval)), ha='left', va='center', fontsize=10)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.title(f'Info: avp:{p}, flow_rate:{fr}, seed:{seed}')
    plt.show()


def get_schedule_startT(start_t, p, fr, max_attempts, plot=False, seed=None, display = False):
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

    def veh_gen6(self, step, dp_times_dict, id_prefix, vType='idm'):
        '''
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
        if id_prefix == 'm':
            veh_route = 'route1'
            dp_v = 24.5
        else:
            veh_route = 'route2'
            dp_v = 10
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

if __name__ == '__main__':
    st = 1200
    p = 0.2
    fr = 1080  # 360
    max_attempts = 7
    seed = 4
    plot = True
    dic_dpt_type = get_schedule(st, p, fr, max_attempts, plot=plot, seed=seed)

    # start_t = 600
    # new_fr = 1080
    # new_dic = get_schedule_startT(start_t, p, fr, max_attempts, plot=True, seed=1)






