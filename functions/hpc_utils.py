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

def get_indicator(speed_log, tp, xml_path, runtime):
    """Calculates performance indicators after simulation."""
    ttc_ratio, avg_speed_std = calc_ttc_sd_exposure.calc_ttc_and_speed_std(xml_path)
    avg_speeds = [item[1] for item in speed_log]
    clean_avg_speeds = [v for v in avg_speeds if v is not None]
    average_v = sum(clean_avg_speeds) / len(clean_avg_speeds)
    print("\n           ---performance indicators---")
    print(f'tp: {tp} veh/h, average_v:{average_v} m/s, '
          f'ttc_ratio: {ttc_ratio}, avg_speed_std: {avg_speed_std}, '
          f'execution_time:{runtime:.1f} s')
    return tp, average_v, ttc_ratio, avg_speed_std, runtime

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

