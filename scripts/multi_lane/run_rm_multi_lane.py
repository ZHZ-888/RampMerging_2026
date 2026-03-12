'''
run_rm_multi_lane.py
multi-lane simulation
'''
import os
import time
from datetime import datetime
from pathlib import Path

from functions import vehicle_generation3 as vg
from functions import data_recording as dr
from functions import print_control as prc  # the shared function of print control
# Shared tools for arguments, KPIs, and CSV logging.// CLI and HPC
from functions import hpc_utils

from comparsion_algo import ramp_metering_algo2 as rm

def rm_main(av_p, r_fr, m_fr, seed,
         gui=False, plot=False, display=False, st=1200):
    """
    Main simulation logic
    """
    # Project root directory
    ROOT = Path(__file__).resolve().parents[2]
    # SUMO SETTING
    sumo_config_path = (
            ROOT
            / "road_network"
            / "multi_lane_motorway"
            / "real"
            / "cfg_multi_lane_merge_rm.sumocfg"
    )
    # Simulation step length
    sim_step = 0.1
    # Determine the SUMO binary based on whether GUI is needed
    sumo_bin = 'sumo-gui' if gui else 'sumo'
    # Construct the SUMO command and options
    traj_dir = Path(os.environ.get(
                    "TRAJ_DIR",
                    ROOT / "data" / "multi_lane" / "algo"
                    ))  # default 'data/mpgc'
    file_name = f"trj_rm_{r_fr}_{av_p}_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"
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
        av_p0, av_p1 = av_p, av_p
        m0_dpt_type = vg.generate_entry_arrivals_shifted_exp(st, av_p0, m_fr, seed)
        m1_dpt_type = vg.generate_entry_arrivals_shifted_exp(st, av_p1, m_fr, 100 - seed)
        r_dpt_type = vg.generate_entry_arrivals_shifted_exp(st, av_p, r_fr, seed)

        veh_gen = vg.VehGen(traci)  # function related to veh generation
        rm_algo = rm.Func(traci)  # function related to ramp_metering algo
        data_recorder = dr.DataRecording(traci, sim_step)

        speed_log, tp = \
            loop(traci, st, veh_gen, data_recorder, rm_algo,
                 m0_dpt_type, m1_dpt_type, r_dpt_type)
    finally:
        traci.close()
    return (speed_log, tp, xml_path)


def loop(traci, st, veh_gen, data_recorder, rm_algo,
         m0_dpt_type=None, m1_dpt_type=None, r_dpt_type=None):
    # START SIMULATION
    step = 0
    speed_log = []
    # rm related parameters
    target_density = 25  # veh_num/km
    gain = 20
    p_release_rate = 200  # veh_num/h

    # scripts loop
    while step < st * 10:
        # checkpoint
        if step > 1126 * 10:
            pass
        traci.simulationStep()  # start simulation
        c_ts = traci.simulation.getTime()  # current_timestep

        if c_ts % 1 == 0:
            prc.print_message(f'************current_time, step:{c_ts, step}************')

        # mainlane vehicle generation
        veh_gen.veh_gen_hetero(step, m1_dpt_type, 'm', 'route0', 27.5, '1')  # 30m/s => 110km/h
        veh_gen.veh_gen_hetero(step, m0_dpt_type, 'm', 'route0', 27.5, '0')  # 25m/s => 90km/h; ori 29.5
        # ramp vehicle generation
        veh_gen.veh_gen_hetero(step, r_dpt_type, 'r', 'route2', 10, '0')

        # performance indicator
        dic_vehinfo = data_recorder.record_vehinfo()
        ls_veh_id = dic_vehinfo['ls_vehid']
        step_speed = data_recorder.get_average_speed(step, ls_veh_id)
        # 1. speed_record
        speed_log.append(step_speed)
        # 2. throughput
        tp = data_recorder.record_throughput(st, ls_veh_id, 'center')  # throughput

        if c_ts % 90 == 0:
            des_center = rm_algo.get_current_density("center_0") # center_0; current center flow
            des_ramp = rm_algo.get_current_density("inflow_merge_0")

            c_release_rate = rm_algo.alinea_control(des_center, target_density, gain, p_release_rate)
            prc.print_message(f'c_release_rate: {c_release_rate}')

            # set the green time of the signal
            g_time, r_time = rm_algo.set_ramp_metering_signal('center', c_release_rate)

            # update last release_rate
            p_release_rate = c_release_rate  # p-previous, c-current
        try:
            prc.print_message(f'g_time: {g_time}, r_time:{r_time}')
            prc.print_message(f'center:{des_center},main:{des_ramp}')
        except:
            pass

        step += 1
    return (speed_log, tp)

def main(args=None, root=None):
    """
    Unified entry point for CLI (command line interface) / HPC.
    This function is called by run.py.
    It parses command-line arguments and calls fifo_main().
    """

    # Algorithm parameters
    # fixed params
    prc.PRINT_ENABLED = False

    # 1. Parse Args (HPC/CLI Mode)
    parser = hpc_utils.standard_arg_parser()
    # Parse arguments passed from run.py
    parsed_args = parser.parse_args(args=args)

    # 2. Run simulation
    start = time.time()
    # Call the original algorithm
    speed_log, tp, xml_path = rm_main(
                                        av_p=0,
                                        r_fr=parsed_args.r_fr,
                                        m_fr=parsed_args.m_fr,
                                        seed=parsed_args.seed,
                                        gui=parsed_args.gui
                                    )
    end = time.time()
    runtime = end - start

    tp, average_v, ttc_ratio, avg_speed_std, runtime = (
        hpc_utils.get_mc_indicator(speed_log, tp, xml_path, runtime))

    # 3. Save results
    if parsed_args.out_csv:
        row = {
            "algo": "rm_multi_lane",
            "ramp_demand": parsed_args.r_fr,
            "mainline_demand": parsed_args.m_fr,
            "seed": parsed_args.seed,
            "throughput": tp,
            "avg_speed": average_v,
            "ttc_ratio": ttc_ratio,
            "avg_speed_std": avg_speed_std,
            "runtime": runtime
        }
        hpc_utils.write_one_row_csv(parsed_args.out_csv, row)

if __name__ == '__main__':
    print(">>> Running Local RM Test (Direct Function Call)...")
    prc.PRINT_ENABLED = False
    start = time.time()
    speed_log, tp, xml_path = rm_main(
        av_p = 0,
        r_fr = 1000,
        m_fr = 1500,
        seed = 1,
        gui = True,
        plot = False,
        display = False,
        st = 1200
    )
    end = time.time()
    runtime = end - start
    hpc_utils.get_mc_indicator(speed_log, tp, xml_path, runtime)
