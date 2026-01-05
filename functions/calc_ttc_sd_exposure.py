# tree = ET.parse("data/Traj/trj_720_0.1_3.xml")
# tree = ET.parse("data/Traj/trajectory_fifo_test.xml")
import os
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import math
from tqdm import tqdm   # pip install tqdm

# -----------------------------
# Function to calculate TTC ratio (high-risk exposure) and average speed std
# -----------------------------
def calc_ttc_and_speed_std(xml_file, ttc_threshold=1.5, veh_length=5.0,
                           merge_xmin=392, merge_xmax=615):
    '''
     <vehicle id="m1_hv_avg" x="5.05" y="139.82" angle="90.00" type="hv_avg"
     speed="24.50" pos="5.10" lane="inflow_highway_0" slope="0.00"/>

    :return:
    '''
    tree = ET.parse(xml_file)
    root = tree.getroot()

    total_pairs = 0
    high_risk_pairs = 0
    speed_std_list = []  # store speed std for each timestep

    for timestep in root.findall('timestep'):
        vehs = []
        merge_speeds = []  # store speeds in merge zone for this timestep

        for veh in timestep.findall('vehicle'):
            vid = veh.get('id')
            x = float(veh.get('x'))
            y = float(veh.get('y'))
            speed = float(veh.get('speed'))
            angle = math.radians(float(veh.get('angle')))
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            vehs.append((vid, x, y, vx, vy))

            # If the vehicle is in merge zone (x-range)
            if merge_xmin <= x <= merge_xmax:
                merge_speeds.append(speed)

        # Calculate local speed std for this timestep
        if len(merge_speeds) > 1:
            speed_std_list.append(np.std(merge_speeds))

        # Calculate TTC exposure ratio
        for i in range(len(vehs)):
            for j in range(len(vehs)):
                if i == j:
                    continue
                id1, x1, y1, vx1, vy1 = vehs[i]
                id2, x2, y2, vx2, vy2 = vehs[j]

                dx = x2 - x1
                dy = y2 - y1
                dist = math.hypot(dx, dy) - veh_length
                if dist <= 0:
                    continue

                dvx = vx2 - vx1
                dvy = vy2 - vy1
                rel_speed_along_line = -(dx * dvx + dy * dvy) / dist

                if rel_speed_along_line > 0:
                    ttc = dist / rel_speed_along_line
                    total_pairs += 1
                    if ttc < ttc_threshold:
                        high_risk_pairs += 1

    # Final results
    ttc_ratio = high_risk_pairs / total_pairs if total_pairs > 0 else 0
    avg_speed_std = np.mean(speed_std_list) if speed_std_list else 0
    return ttc_ratio, avg_speed_std


def batch_process_all(csv_path, traj_dir, output_path, prefix="trj_", model="mpgc"):
    df = pd.read_csv(csv_path)
    # Add new columns for TTC exposure ratio and speed std
    df["ttc_ratio"] = 0.0
    df["avg_speed_std"] = 0.0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Batch processing..."):
        if model == 'mpgc':
            xml_name = (
                f"{prefix}{int(row['r_fr'])}_"
                f"{row['av_p']:.2g}_"
                f"{int(row['seed'])}_"
                f"{row['p_autoFollow']:.2g}_"
                f"{row['platoon_p']:.2g}_"
                f"{row['loss_rate']:.2g}.xml"
            )
            xml_path = os.path.join(traj_dir, xml_name)
        else: # FIFO or RM
            xml_name = (
                f"trj_{int(row['r_fr'])}_"
                f"{row['av_p']:.2g}_"
                f"{int(row['seed'])}.xml"
            ) # fifo and rm
            # ---FIFO---
            # xml_path = os.path.join("data/fifo/traj_fifo/traj_fifo_all", xml_name)  # adjust folder path if needed
            # ---RM---
            # xml_path = os.path.join("data/rm/traj_rm", xml_name)  # adjust folder path if needed
        if os.path.exists(xml_path):
            try:
                ttc, std = calc_ttc_and_speed_std(xml_path)
                df.at[idx, "ttc_ratio"] = ttc
                df.at[idx, "avg_speed_std"] = std
            except Exception as e:
                print(f"[ERROR] {xml_name}: {e}")
        else:
            print(f"[MISSING] {xml_name}")
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    mode = 'batch'
    if mode == 'batch':
        # batch
        csv_path = "/data/mpgc/all_results_mpgc_250825_2_exp1.csv"
        traj_dir = "../data/mpgc/1ideal"
        # output_path = "data/mpgc/df_exp1_add_0825_1.csv"
        output_path = "../data/mpgc/df_exp1_add_0825_2.csv"
        batch_process_all(csv_path, traj_dir, output_path)
    elif mode == 'single':
        xml_path = "../data/mpgc/trj_900_0.1_1_1_1_0.xml"
        ttc_ratio, avg_speed_std = calc_ttc_and_speed_std(xml_path)
        print(f'ttc_ratio:{ttc_ratio}, avg_speed_std:{avg_speed_std}')