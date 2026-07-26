'''
hpc_utils.py

Shared utility functions for HPC simulation tasks.
Includes:
- Standard argument parsing
- Performance indicator calculations
- CSV logging
'''

import csv
import argparse
from pathlib import Path
from functions import calc_ttc_sd_exposure
from collections import defaultdict

def standard_arg_parser():
    """Returns a parser with the standard arguments for your experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--r_fr", type=float, default=800)
    parser.add_argument("--m_fr", type=float, default=1500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument('--av_p', type=float, default=0.1, help='AV Penetration Rate')
    parser.add_argument(
        "--max_team_size",
        type=int,
        default=12,
        help="Maximum platoon size used for tagging and evaluation",
    )
    parser.add_argument(
        "--tsg_mode",
        type=str,
        default="predict",
        choices=["off", "train", "predict", "audit", "fix"],
    )
    return parser

def training_arg_parser():
    """Parser for RL Training"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--av_p", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--st", type=int, default=1200*100)
    parser.add_argument("--train_agent", type=str, choices=['SA', 'CA', 'TSG', None], default=None)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--train_interval", type=int, default=32) # update_interval = batch_size*2
    # Use nargs='+' to allow multiple integers: --batch_epoch 16 5
    parser.add_argument("--hidden_layer", type=int, nargs='+', default=[32, 32])
    # Note: Training usually doesn't need out_csv for traffic KPIs
    return parser

def get_mc_indicator(speed_log, tp, ssm_path, runtime):
    """Calculates performance indicators after simulation."""
    total_vehicle_num = int(tp/3)
    ttc_metrics_3 = calc_ttc_sd_exposure.calc_ttc_conflict_metrics(ssm_path, total_vehicle_num, ttc_threshold=3)
    ttc_metrics_2 = calc_ttc_sd_exposure.calc_ttc_conflict_metrics(ssm_path, total_vehicle_num, ttc_threshold=2)
    ttc_metrics_1 = calc_ttc_sd_exposure.calc_ttc_conflict_metrics(ssm_path, total_vehicle_num, ttc_threshold=1.5) # 1.5
    ttc_ratio_3 = ttc_metrics_3[2]
    ttc_ratio_2 = ttc_metrics_2[2]
    ttc_ratio_1 = ttc_metrics_1[2]
    avg_speeds = [item[1] for item in speed_log]
    clean_avg_speeds = [v for v in avg_speeds if v is not None]
    average_v = sum(clean_avg_speeds) / len(clean_avg_speeds)
    print("\n           ---merging control performance indicators---")
    print(f'tp: {tp:.0f} veh/h, average_v:{average_v:.2f} m/s, '
          f'ttc_ratio_3: {ttc_ratio_3:.8f}, '
          f'ttc_ratio_2: {ttc_ratio_2:.8f}, '
          f'ttc_ratio_1.5: {ttc_ratio_1:.8f}, '
          f'execution_time:{runtime:.2f} s')
    return tp, average_v, ttc_ratio_3, ttc_ratio_2, ttc_ratio_1, runtime

def get_fc_detail(dic_follower_state, his_dic_platoon_size, max_size=11):
    """
    Get formation control statistics.

    Parameters
    ----------
    dic_follower_state : dict
        Format:
        {'mhv48': ['following_mode', 'mav38'], 'mhv65': ['free_mode', 'mav38']}
    his_dic_platoon_size : dict
        Format:
        {'mav38': 11, 'mav278': 8, 'mav754': 15}
    max_size : int, optional
        Maximum allowed platoon size.
    Returns
    -------
    dict
        {'over_pltn': int,
        'non_over_pltn': int,
        'std_pltn': int,
        'sparse_pltn': int,
        'avg_pltn_size': float}
    """
    ls_oversized_leader = []
    ls_sparse_leader = []

    over_pltn = 0
    non_over_pltn = 0
    std_pltn = 0
    sparse_pltn = 0

    # Store follower states for each leader
    leader_follower_states = defaultdict(list)

    for follower_id, (state, leader_id) in dic_follower_state.items():
        leader_follower_states[leader_id].append(state)

    # Calculate average platoon size
    total_size = sum(his_dic_platoon_size.values())

    for leader_id, platoon_size in his_dic_platoon_size.items():

        # Count oversized platoons
        if platoon_size > max_size:
            if leader_id in ['mb_av9304', 'm_av11610']:
                pass
            over_pltn += 1
            ls_oversized_leader.append(leader_id)
            continue

        # Count non-oversized platoons
        non_over_pltn += 1

        follower_states = leader_follower_states.get(leader_id, [])

        # A non-oversized platoon is considered sparse
        # if any follower is not in following_mode.
        if any(state != "following_mode" for state in follower_states):
            if leader_id in ['mb_av5939', 'mb_av7435']:
                pass
            ls_sparse_leader.append(leader_id)
            sparse_pltn += 1
        else:
            std_pltn += 1

    avg_pltn_size = (
        total_size / len(his_dic_platoon_size)
        if his_dic_platoon_size
        else 0
    )

    sum_platoon_count = over_pltn + non_over_pltn

    result = {
        "over_pltn": over_pltn,
        "non_over_pltn": non_over_pltn,
        "std_pltn": std_pltn,
        "sparse_pltn": sparse_pltn,
        "avg_pltn_size": avg_pltn_size,
    }
    print("\nNon standard platoon leaders:")
    print(f"Oversized leaders: {ls_oversized_leader}")
    print(f"Sparse leaders: {ls_sparse_leader}")

    print("\nFormation Control Statistics:")
    print(f'Sum (Over, NonOver): {sum_platoon_count} ({over_pltn}, {non_over_pltn})')
    print(f'NonOver (Sparse, Standard): {non_over_pltn} ({sparse_pltn}, {std_pltn})')
    print(f"Avg Platoon Size  : {result['avg_pltn_size']:.2f}")
    return result

def get_fc_indicator(dic_follower_state, his_dic_platoon_size, max_size=11):
    '''
    get formation control indicators
    Parameters
    ----------
    dic_follower_state: {'mhv48': ['following_mode', 'mav38'], 'mhv65': ['following_mode', 'mav38']}
    his_dic_platoon_size: {'mav38': 11, 'mav278': 11, 'mav754': 11}; history of dic_platoon_size
    max_size: int, optional
        Maximum allowed platoon size used to define a standard platoon.
    '''
    # PLATOON FORMATION results
    # indicator 1: num.platoon_followers/num.followers %

    # print(dic_follower_state)
    # print(his_dic_platoon_size)

    num_follower = len(dic_follower_state)
    num_platoon_follower = len([k for k, v in dic_follower_state.items() if v[0] == 'following_mode'])
    print("\n           ---formation control performance indicators---")
    print(f"index1: {num_platoon_follower} platoon_followers, {num_follower} followers, "
          f"ratio: {num_platoon_follower / num_follower * 100:.2f}%")
    ca_indicator = round(num_platoon_follower / num_follower, 3)
    # indicator 2: num.normal_size_platoon/num.platoon
    num_platoon = len(his_dic_platoon_size)
    num_normal_size_platoon = len([k for k, v in his_dic_platoon_size.items() if v <= max_size])
    print(f"index2: {num_normal_size_platoon} standard_platoon, {num_platoon} platoon, "
          f"ratio: {num_normal_size_platoon / num_platoon * 100:.2f}%, ")
    sa_indicator = round(num_normal_size_platoon / num_platoon, 3)
    return ca_indicator, sa_indicator

def write_one_row_csv(path: str, row: dict):
    """Writes a single row to a CSV file safely."""
    # Safety check: do nothing if no path is provided
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    file_exists = p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def write_dataframe_csv(path: str, df):
    """
    Save a DataFrame to CSV.
    """
    if path is None:
        return

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(p, index=False)

