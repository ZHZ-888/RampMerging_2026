"""Independent CVMC runner; no PF or platoon-level merging controller."""
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from functions import vehicle_generation3 as vg
from functions import ramp_platoon_generator as rpg
from functions import data_recording as dr
from functions import hpc_utils, print_control as prc
from functions.cvmc_controller import CVMCConfig, CVMCController


ROOT = Path(__file__).resolve().parents[2]


def cvmc_main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.gui:
        import traci
    else:
        import libsumo as traci
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    task = os.environ.get('SLURM_ARRAY_TASK_ID', 'local')
    tag = f'cvmc_spr{args.ramp_spr}_r{args.r_fr}_av{args.av_p}_seed{args.seed}_{task}_{stamp}'
    traj_dir = Path(os.environ.get('TRAJ_DIR', ROOT / 'data/multi_lane/cvmc'))
    traj_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: traj_dir / f'{prefix}_{tag}.xml' for name, prefix in
             [('xml_path', 'trj'), ('lc_path', 'lc'), ('ssm_path', 'ssm'),
              ('tripinfo_path', 'tripinfo')]}
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    config = CVMCConfig(interval=args.update_steps, buffer=args.time_buffer, v_floor=args.v_floor,
                        min_speed=args.min_speed, distance_tolerance=args.distance_tolerance,
                        speed_tolerance=args.speed_tolerance)
    sumo_config = ROOT / 'road_network/multi_lane_motorway/real/cfg_multi_lane_merge.sumocfg'
    cmd = [args.sumo_binary or ('sumo-gui' if args.gui else 'sumo'), '-c', str(sumo_config),
           '--seed', str(args.seed), '--step-length', '0.1', '--no-warnings',
           '--fcd-output', str(paths['xml_path']), '--tripinfo-output', str(paths['tripinfo_path']),
           '--lanechange-output', str(paths['lc_path']),
           '--device.ssm.probability', '1', '--device.ssm.file', str(paths['ssm_path']),
           '--device.ssm.measures', 'TTC DRAC', '--device.ssm.thresholds', '3 3',
           '--device.ssm.trajectories', 'true', '--device.ssm.write-lane-positions', 'true']
    traci.start(cmd)
    try:
        print('SUMO engine:', traci.getVersion())
        main0 = vg.generate_entry_arrivals_shifted_exp(args.st, args.av_p, args.m_fr, args.seed)
        main1 = vg.generate_entry_arrivals_shifted_exp(args.st, args.av_p, args.m_fr, 100-args.seed)
        ramp_input = rpg.RampPlatoonGenerator(
            st=args.st, av_p=args.av_p, fr=args.r_fr,
            ramp_spr=args.ramp_spr, seed=args.seed,
        ).generate()
        ramp_departures = {round(row["dpt_ts"] * 10): row
                           for row in ramp_input["schedule"]}
        generator = vg.VehGen(traci, args.seed)
        recorder = dr.DataRecording(traci)  # Evaluation only; never passed to CVMC.
        with (log_dir / f'{tag}.csv').open('w', newline='') as log_file:
            controller = CVMCController(traci, config, log_file)
            speed_log, tp = [], 0.0
            step = 0
            while step < args.st*10 or traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                if step < args.st*10:
                    generator.veh_gen_hetero(step, main0, 'm', 'route_m',
                                             27.5, '0')
                    generator.veh_gen_hetero(step, main1, 'm', 'route_m',
                                             27.5, '1')
                    generator.ramp_schedule_gen(step, ramp_departures)
                controller.step(step)
                if step < args.st*10:
                    vehicles = recorder.record_vehinfo()['ls_vehid']
                    speed_log.append(recorder.get_average_speed(step, vehicles))
                    tp = recorder.record_throughput(args.st, vehicles, 'center',
                                                    warmup_time=hpc_utils.DEFAULT_WARMUP_TIME)
                step += 1
            stats = controller.stats.copy()
    finally:
        traci.close()
    return speed_log, tp, paths, stats


def main(args=None, root=None, **local_settings):
    prc.PRINT_ENABLED = False
    parser = hpc_utils.standard_arg_parser(include_legacy_ramp=False)
    parser.add_argument('--ramp_spr', type=float, default=0.7,
                        help='Target fraction of ramp platoons classified as Standard')
    parser.add_argument('--time_buffer', type=float, default=1.5)
    parser.add_argument('--update_steps', type=int, default=60)
    parser.add_argument('--v_floor', type=float, default=0.1)
    parser.add_argument('--min_speed', type=float, default=5.0)
    parser.add_argument('--distance_tolerance', type=float, default=5.0)
    parser.add_argument('--speed_tolerance', type=float, default=2.0)
    parser.add_argument('--sumo_binary', default=None)
    parser.add_argument('--log_dir', default='data/cvmc_logs')
    parser.set_defaults(**local_settings)
    parsed = parser.parse_args(args)
    hpc_utils.validate_safety_csv_path(parser, parsed)
    if parsed.st <= hpc_utils.DEFAULT_WARMUP_TIME:
        parser.error('--st must exceed the 180 s warmup')
    if parsed.update_steps < 1 or parsed.v_floor <= 0 or parsed.time_buffer < 0:
        parser.error('update_steps and v_floor must be positive; time_buffer must be nonnegative')
    start = time.time()
    speed_log, tp, paths, stats = cvmc_main(parsed)
    runtime = time.time() - start
    _, avg_speed, ttc3, ttc2, ttc15, _ = hpc_utils.get_mc_indicator(
        speed_log, tp, paths['ssm_path'], runtime, max_time=parsed.st)
    delay = hpc_utils.get_delay_indicator(paths['tripinfo_path'])
    entries, mrm, entry_ids = hpc_utils.get_mrm_insertion_counts(
        paths['xml_path'], hpc_utils.DEFAULT_WARMUP_TIME, parsed.st)
    mr3, mr2, mr15 = hpc_utils.get_mr_ttc_ratios(paths['ssm_path'], entry_ids, max_time=parsed.st)
    row = dict(algo='cvmc_multi_lane', av_p=parsed.av_p, ramp_demand=parsed.r_fr,
               mainline_demand=parsed.m_fr, seed=parsed.seed, ramp_spr=parsed.ramp_spr,
               time_buffer=parsed.time_buffer, update_steps=parsed.update_steps,
               v_floor=parsed.v_floor, min_speed=parsed.min_speed,
               distance_tolerance=parsed.distance_tolerance, speed_tolerance=parsed.speed_tolerance,
               throughput=tp, avg_speed=avg_speed, ttc_ratio_3=ttc3, ttc_ratio_2=ttc2,
               ttc_ratio_1=ttc15, mr_ttc_ratio_3=mr3, mr_ttc_ratio_2=mr2,
               **{'mr_ttc_ratio_1.5': mr15}, **delay,
               ramp_entry_count=entries, mrm_insertion_count=mrm, runtime=runtime)
    row.update({f'cvmc_{key}': value for key, value in stats.items()})
    hpc_utils.write_one_row_csv(parsed.out_csv, row)
    if parsed.safety_out_csv:
        hpc_utils.write_safety_metrics(parsed, "cvmc_multi_lane", paths)
    print('CVMC summary:', stats, '| runtime:', round(runtime, 2))


if __name__ == '__main__':
    # Local test: edit parameters here, then run this file in PyCharm.
    # HPC/CLI uses run.py --algo cvmc_multi_lane and its command-line parameters.
    main(
        args=[],
        av_p=0.3,
        r_fr=800,
        m_fr=800,
        seed=2,
        ramp_spr=0.7,
        st=1500,
        gui=True,
        time_buffer=1.5,
        update_steps=60,  # 6 s
        min_speed=5.0,
        out_csv=None,  # Set a CSV path to save the local result.
    )
