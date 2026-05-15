'''
run_mpgc_multi_lane_collect_RF_at_data.py
random forest arrival time feature/target data collection
multi-lane simulation
'''
import os
import time
import pandas as pd
from pathlib import Path

from functions import vehicle_generation3 as vg
from functions import print_control as prc  # the shared fuction of print control
from functions import formation_controller as fc
from functions import merging_controller as mc
from functions import data_recording as dr
# Shared tools for arguments, KPIs, and CSV logging.// CLI and HPC
from functions import hpc_utils


def mpgc_main(av_p, r_fr, m_fr, seed, r_autoFollow_p=1, r_platoon_p=1, loss_rate=0,
              gui=False, plot=False, display=False, lc=False, st=2400):
    # SUMO SETTING
    # path = '/home/zzha/PycharmProjects/RoadNetwork/multi_lane_motorway/multi_lane_motorway1_lefthand_noLaneChanging.sumocfg'
    # path = 'road_network/multi_lane_motorway/cfg_pf.sumocfg'
    # path = '../../road_network/multi_lane_motorway/real/cfg_multi_lane_merge.sumocfg'
    # sumo_config_path = path
    # Project root directory
    ROOT = Path(__file__).resolve().parents[2]
    # SUMO SETTING
    sumo_config_path = (
            ROOT
            / "road_network"
            / "multi_lane_motorway"
            / "real"
            / "cfg_multi_lane_merge.sumocfg"
    )
    # Simulation step length
    sim_step = 0.1
    # Determine the SUMO binary based on whether GUI is needed
    sumo_bin = 'sumo-gui' if gui else 'sumo'
    # Construct the SUMO command and options
    file_name = f'trj_{r_fr}_{av_p}_{seed}_{loss_rate}.xml'
    sumo_cmd = [sumo_bin, "-c", str(sumo_config_path),
                "--no-warnings"]  # , '-S' start auto, and quit auto
    sumo_options = ["--step-length", str(sim_step)]

    # If GUI is enabled, set the GUI view schema
    if gui:
        import traci
        traci.start(sumo_cmd + sumo_options)
        available_views = traci.gui.getIDList()
        print("Available Views:", available_views)
    else:
        import libsumo as traci
        traci.start(sumo_cmd + sumo_options)
    try:
        # VEHICLE GENERATOR
        # scripts road veh departure schedule (lane0 and lane1)
        max_attempts = 7
        av_p0, av_p1 = av_p, av_p
        m0_dpt_type = vg.generate_entry_arrivals_shifted_exp(st, av_p0, m_fr, seed)
        m1_dpt_type = vg.generate_entry_arrivals_shifted_exp(st, av_p1, m_fr, 100 - seed)
        r_dpt_type = vg.get_schedule2(st, av_p, r_fr, r_platoon_p,
                                      max_attempts, plot, seed, display)
        # ramp road veh depature schedule
        veh_gen = vg.VehGen(traci)  # function related to veh generation
        data_recorder = dr.DataRecording(traci)
        data_recorder.get_avhid_ptype(r_dpt_type = r_dpt_type)  # here only have r_dpt_type

        formation_controller = fc.FormationController(data_recorder, traci)
        merging_controller = mc.MergingController(data_recorder, traci, av_p,
                                                  platoon_formation=True, ml=True)

        dic_targets, ls_features = \
            loop(traci, st, data_recorder, veh_gen, formation_controller, merging_controller, lc,
                 r_autoFollow_p, m0_dpt_type, m1_dpt_type, r_dpt_type)
    finally:
        traci.close()
    return (dic_targets, ls_features)

def loop(traci, st, data_recorder, veh_gen, formation_controller, merging_controller, lc,
         r_autoFollow_p, m0_dpt_type=None, m1_dpt_type=None, r_dpt_type=None):
    # START SIMULATION
    step = 0
    # scripts loop
    while step < st * 10:
        # checkpoint
        if step > 740 * 10:
            pass
        traci.simulationStep()  # start simulation
        # mainlane vehicle generation
        veh_gen.veh_gen_homo(step, m1_dpt_type, 'm', 'route_m', 27.5, '0')  # 30m/s => 110km/h
        # ramp vehicle generation
        veh_gen.platoon_gen(step, r_dpt_type, 'r', r_autoFollow_p)

        # disable lane-changing
        # data_recorder.disable_all_lane_changes()

        # core control
        (dic_score_reward, dic_follower_state, his_dic_platoon_size, dic_id_features,
         dic_final_platoon_info) = formation_controller.step(st, step, lc)
        tp, speed_log, queue_log = (
            merging_controller.step(st, step, dic_final_platoon_info, r_dpt_type))

        # record targets
        data_recorder.record_tail_arrival(step) # data_recorder.dic_tail_arrived_ws
        step += 1
    dic_targets = data_recorder.dic_tail_arrived_ws
    ls_features = data_recorder.ls_features
    return (dic_targets, ls_features)

def main(args=None, root=None):
    """
    Unified entry point for CLI (command line interface) / HPC.
    This function is called by run.py.
    It parses command-line arguments and calls mpgc_main().
    """
    prc.PRINT_ENABLED = False

    # 1. Parse Args (HPC/CLI Mode)
    parser = hpc_utils.standard_arg_parser()
    parsed_args = parser.parse_args(args=args)

    # 2. Run simulation
    # Call the original algorithm
    dic_targets, ls_features = mpgc_main(
        av_p=parsed_args.av_p,
        r_fr=parsed_args.r_fr,
        m_fr=parsed_args.m_fr,
        seed=parsed_args.seed,
        gui=parsed_args.gui
    )

    # 3. Save results
    if parsed_args.out_csv:
        collect_RF_data(dic_targets, ls_features, parsed_args.out_csv)

def collect_RF_data(dic_targets, ls_features, output_path):
    columns = [
        "leader_id",
        "record_index",
        "prediction_ts", # timestamp when record the features
        "platoon_type",
        "leader_to_pv_dis", # dis_to_pv
        "leader_speed", # speed leader
        "leader_left_dis", # remain_dis_leader
        "m"
    ]
    df = pd.DataFrame(ls_features, columns=columns)
    df["arrival_ts"] = df["leader_id"].map(
        lambda x: dic_targets[x][1] if x in dic_targets else None)
    # Ensure directory exists (optional but safe)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # CHANGED: Use the variable, not the hardcoded string
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    prc.PRINT_ENABLED = False
    start = time.time()
    dic_targets, ls_features = mpgc_main(
        av_p = 0.1,
        r_fr = 900,
        m_fr = 0, # 1500
        seed = 16,
        r_autoFollow_p = 1,
        r_platoon_p = 1,
        loss_rate = 0,
        gui = True,
        plot = False,
        display = False,
        lc = False,
        st = 1200
    )
    end = time.time()

    # orgnise data
    # output_path = "/home/zzha/PycharmProjects/RampMerging_2026/data/features/df_rf_at_260123.csv"
    output_path = "/home/zzha/PycharmProjects/RampMerging_2026/data/features/df_rf_at_260318.csv"
    collect_RF_data(dic_targets, ls_features, output_path)

