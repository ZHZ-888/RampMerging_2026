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

def standard_arg_parser():
    """Returns a parser with the standard arguments for your experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--r_fr", type=float, default=720)
    parser.add_argument("--m_fr", type=float, default=1500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument('--av_p', type=float, default=0, help='AV Penetration Rate')
    return parser

def training_arg_parser():
    """Parser for RL Training"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--av_p", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--st", type=int, default=1200*100)
    parser.add_argument("--train_agent", type=str, choices=['SA', 'CA'], default='CA')
    parser.add_argument("--lr", type=float, default=0.0005)
    # Use nargs='+' to allow multiple integers: --batch_epoch 16 5
    parser.add_argument("--batch_epoch", type=int, nargs='+', default=[16, 5])
    parser.add_argument("--hidden_layer", type=int, nargs='+', default=[32, 32])
    # Note: Training usually doesn't need out_csv for traffic KPIs
    return parser

def get_mc_indicator(speed_log, tp, xml_path, runtime):
    """Calculates performance indicators after simulation."""
    ttc_ratio, avg_speed_std = calc_ttc_sd_exposure.calc_ttc_and_speed_std(xml_path)
    avg_speeds = [item[1] for item in speed_log]
    clean_avg_speeds = [v for v in avg_speeds if v is not None]
    average_v = sum(clean_avg_speeds) / len(clean_avg_speeds)
    print("\n           ---merging control performance indicators---")
    print(f'tp: {tp} veh/h, average_v:{average_v} m/s, '
          f'ttc_ratio: {ttc_ratio}, avg_speed_std: {avg_speed_std}, '
          f'execution_time:{runtime:.1f} s')
    return tp, average_v, ttc_ratio, avg_speed_std, runtime

def get_fc_indicator(dic_follower_state, his_dic_platoon_size):
    '''
    get formation control indicators
    Parameters
    ----------
    dic_follower_state
    his_dic_platoon_size
    '''
    # PLATOON FORMATION results
    # indicator 1: num.platoon_followers/num.followers %
    num_follower = len(dic_follower_state)
    num_platoon_follower = len([k for k, v in dic_follower_state.items() if v[0] == 'following_mode'])
    print("\n           ---formation control performance indicators---")
    print(f"index1: {num_platoon_follower} platoon_followers, {num_follower} followers, "
          f"ratio: {num_platoon_follower / num_follower * 100:.1f}%")
    # indicator 2: num.normal_size_platoon/num.platoon
    num_platoon = len(his_dic_platoon_size)
    num_normal_size_platoon = len([k for k, v in his_dic_platoon_size.items() if v <= 11])
    print(f"index2: {num_platoon} num_platoon, {num_normal_size_platoon} num_normal_size_platoon, "
          f"ratio: {num_normal_size_platoon / num_platoon * 100:.1f}%, ")
    return

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

