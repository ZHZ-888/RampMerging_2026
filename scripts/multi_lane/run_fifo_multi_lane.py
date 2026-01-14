'''
run_fifo_multi_lane.py
multi-lane simulation
'''
import os
import time
from pathlib import Path
import csv

from functions import vehicle_generation3 as vg
from functions import print_control as prc  # the shared fuction of print control
from functions import calc_ttc_sd_exposure
from functions import data_recording as dr

def fifo_main(av_p, r_fr, m_fr, seed,
         gui=False, plot=False, display=False, st=1200):
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
    traj_dir = Path(os.environ.get(
                "TRAJ_DIR",
                ROOT / "data" / "multi_lane" / "algo"
                 )) # default 'data/mpgc'
    file_name = f'trj_{r_fr}_{av_p}_{seed}.xml'
    xml_path = os.path.join(traj_dir, file_name)
    sumo_cmd = [sumo_bin, "-c", str(sumo_config_path),
                "--fcd-output", str(xml_path), # save path
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
        # scripts road veh depature schedule (lane0 and lane1)
        data_recorder = dr.DataRecording(traci)
        max_attempts = 7
        av_p0 = av_p
        av_p1 = av_p
        m0_dpt_type = vg.generate_entry_arrivals_shifted_exp(st, av_p0, m_fr, seed)
        m1_dpt_type = vg.generate_entry_arrivals_shifted_exp(st, av_p1, m_fr, 100 - seed)

        r_dpt_type = vg.generate_entry_arrivals_shifted_exp(st, av_p, r_fr, seed)
        # ramp road veh depature schedule
        vgvg = vg.VehGen(traci)  # function related to veh generation
        # data_recorder.get_avhid_ptype(r_dpt_type = r_dpt_type)  # here only have r_dpt_type

        speed_log, tp = \
            loop(traci, st, vgvg, data_recorder,
                 m0_dpt_type, m1_dpt_type, r_dpt_type)
    finally:
        traci.close()
    return (speed_log, tp, xml_path)


def loop(traci, st, vgvg, data_recorder,
         m0_dpt_type=None, m1_dpt_type=None, r_dpt_type=None):
    # START SIMULATION
    step = 0
    speed_log = []
    # scripts loop
    while step < st * 10:
        # checkpoint
        if step > 1126 * 10:
            pass
        traci.simulationStep()  # start simulation
        c_ts = traci.simulation.getTime()  # current_timestep
        if c_ts % 1 == 0:
            prc.print_message(f'************current_time, step:{c_ts, step}************')

        # vehicle generation
        vgvg.veh_gen_hetero(step, m0_dpt_type, 'm', 'route0', 27.5, '0')  # 25m/s => 90km/h; ori 29.5
        vgvg.veh_gen_hetero(step, m1_dpt_type, 'm', 'route0', 27.5, '1')  # 30m/s => 110km/h
        # ramp vehicle generation
        # vgvg.veh_gen_heter2(step, r_dpt_type, 'r', r_autoFollow_p)
        vgvg.veh_gen_hetero(step, r_dpt_type, 'r', 'route2', 10, '0')

        # for veh_id in traci.vehicle.getIDList():
        #     traci.vehicle.setLaneChangeMode(veh_id, 0)

        # performance indicator
        dic_vehinfo = data_recorder.record_vehinfo()
        ls_veh_id = dic_vehinfo['ls_vehid']
        step_speed = data_recorder.get_average_speed(step, ls_veh_id)
        # 1. speed_record
        speed_log.append(step_speed)
        # 2. throughput
        tp = data_recorder.record_throughput(st, ls_veh_id, 'center')  # throughput

        step += 1
    return (speed_log, tp)

def main(args=None, root=None):
    """
    Unified entry point.
    This function is called by run.py.
    It parses command-line arguments and calls fifo_main().
    """
    import argparse
    parser = argparse.ArgumentParser(description="FIFO multi-lane simulation")

    # Algorithm parameters
    # fixed params
    av_p = 0
    prc.PRINT_ENABLED = False
    plot = False
    display = False
    st = 1200

    parser.add_argument("--r_fr", type=float, default=720, help="Ramp flow rate")
    parser.add_argument("--m_fr", type=float, default=1500, help="Mainline flow rate")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    # Optional flags
    parser.add_argument("--gui", action="store_true", help="Enable SUMO GUI") # default False
    parser.add_argument("--out_csv", type=str, default=None, help="Write KPIs to this CSV file")

    # Parse arguments passed from run.py
    parsed_args = parser.parse_args(args=args)

    start = time.time()
    # Call the original algorithm
    speed_log, tp, xml_path = fifo_main(
        av_p=av_p,
        r_fr=parsed_args.r_fr,
        m_fr=parsed_args.m_fr,
        seed=parsed_args.seed,
        gui=parsed_args.gui
    )
    end = time.time()
    runtime = end - start
    tp, average_v, ttc_ratio, avg_speed_std, runtime = (
        get_indicator(speed_log, tp, xml_path, runtime))
    # write indicator into csv
    if parsed_args.out_csv:
        row = {
            "algo": "fifo_multi_lane",
            "ramp_demand": parsed_args.r_fr,
            "mainline_demand": parsed_args.m_fr,
            "seed": parsed_args.seed,
            "throughput": tp,
            "avg_speed": average_v,
            "ttc_ratio": ttc_ratio,
            "avg_speed_std": avg_speed_std,
            "runtime": runtime
        }
        write_one_row_csv(parsed_args.out_csv, row)

def get_indicator(speed_log, tp, xml_path, runtime):
    # Performance indicators
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
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    file_exists = p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            w.writeheader()
        w.writerow(row)

if __name__ == '__main__':
    r_fr = 720  # 540
    m_fr = 1500  # 1080
    seed = 1  # 4

    st = 1200  # 1200
    av_p = 0

    prc.PRINT_ENABLED = False
    gui = True
    plot = False
    display = True

    start = time.time()
    (speed_log, tp, xml_path) \
        = fifo_main(av_p, r_fr, m_fr, seed, gui, plot, display, st)
    end = time.time()

    runtime = end - start
    get_indicator(speed_log, tp, xml_path, runtime)


