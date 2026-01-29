# run_mpgc_multi_lane.py
'''
multi-lane simulation
test longer platoon
'''
import os
import time
from pathlib import Path

from functions import vehicle_generation3 as vg
from functions import print_control as prc  # the shared fuction of print control
from functions import formation_controller as fc
from functions import merging_controller as mc
from functions import data_recording as dr
# Shared tools for arguments, KPIs, and CSV logging.// CLI and HPC
from functions import hpc_utils


def mpgc_main(av_p, r_fr, m_fr, seed, r_autoFollow_p, r_platoon_p, loss_rate=0,
         gui=False, plot=False, display=False, lc=False, st=1000):
    # SUMO SETTING
    ROOT = Path(__file__).resolve().parents[2]
    sumo_config_path = (ROOT
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
    '''
    Look for a system setting named TRAJ_DIR. If it exists, use it. If not, use this default folder.
    '''
    traj_dir = os.environ.get("TRAJ_DIR", "../../data/multi_lane/algo")
    file_name = f'trj_{r_fr}_{av_p}_{seed}_{loss_rate}.xml'
    xml_path = os.path.join(traj_dir, file_name)
    sumo_cmd = [sumo_bin, "-c", str(sumo_config_path),
                "--fcd-output", str(xml_path), # save path
                "--no-warnings"]  #
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

        (dic_score_reward, dic_follower_state, his_dic_platoon_size,
         dic_id_features, tp, speed_log, queue_log) = \
            loop(traci, st, data_recorder, veh_gen, formation_controller, merging_controller, lc,
                 r_autoFollow_p, m0_dpt_type, m1_dpt_type, r_dpt_type)
    finally:
        traci.close()
    return (dic_score_reward, dic_follower_state, his_dic_platoon_size, dic_id_features,
            tp, speed_log, queue_log, xml_path)

def loop(traci, st, data_recorder, veh_gen, formation_controller, merging_controller, lc,
         r_autoFollow_p, m0_dpt_type=None, m1_dpt_type=None, r_dpt_type=None):
    # START SIMULATION
    step = 0
    # scripts loop
    while step < st * 10:
        # checkpoint
        if step > 55 * 10:
            pass
        traci.simulationStep()  # start simulation

        c_ts = traci.simulation.getTime()  # current_timestep
        if c_ts % 1 == 0:
            prc.print_message(f'************current_time, step:{c_ts, step}************')

        # main vehicle generation
        veh_gen.veh_gen_homo(step, m1_dpt_type, 'm', 'route0', 27.5, '1')  # 30m/s => 110km/h
        veh_gen.veh_gen_homo(step, m0_dpt_type, 'm', 'route0', 27.5, '0')  # 25m/s => 90km/h; ori 29.5
        # ramp vehicle generation
        veh_gen.veh_gen_heter2(step, r_dpt_type, 'r', r_autoFollow_p)

        (dic_score_reward, dic_follower_state, his_dic_platoon_size,
         dic_id_features, dic_final_platoon_info) = formation_controller.step(st, step, lc)
        tp, speed_log, queue_log = merging_controller.step(st, step,
                                                           dic_final_platoon_info, r_dpt_type)

        data_recorder.record_tail_arrival(step)
        step += 1
    return (dic_score_reward, dic_follower_state, his_dic_platoon_size, dic_id_features,
            tp, speed_log, queue_log)

def _set_dynamic_traffic(step, start_t, r_dpt_type, dynamic=True):
    '''
    set dynamic traffic. For example: default r_demands=720 veh/h; start_t=5 min, new_fr=180 veh/h;
                                      after 5 min simulation, 720 veh/h => 180 veh/h
    '''
    # 100324update: new ramp flow rate after 350
    start_t = 300
    dynamic = False
    if step == start_t * 10 and dynamic:
        p = 0.3
        new_fr = 180
        seed = 1
        new_r_dpt_type = vg.get_schedule_startT(start_t, p, new_fr, max_attempts=1, seed=seed)  # after start_t
        r_dpt_type = {key: value for key, value in r_dpt_type.items() if key <= start_t}  # before start_t
        r_dpt_type.update(new_r_dpt_type)  # merge together
    return r_dpt_type


if __name__ == '__main__':
    prc.PRINT_ENABLED = False
    start = time.time()
    (dic_score_reward, dic_follower_state, his_dic_platoon_size, dic_id_features,
     tp, speed_log, queue_log, xml_path) = mpgc_main(
        av_p = 0.3,
        r_fr = 720,
        m_fr = 1200,
        seed = 1,
        r_autoFollow_p = 1,  # auto follow proportion
        r_platoon_p = 1, # percentage of platoon vehicles
        loss_rate = 0,
        gui = True,
        plot = False,
        display = False,
        lc = False, # if consider HV lane-changing
        st = 1200
    )
    end = time.time()
    runtime = end - start

    hpc_utils.get_fc_indicator(dic_follower_state, his_dic_platoon_size)

    tp, average_v, ttc_ratio, avg_speed_std, runtime = (
        hpc_utils.get_mc_indicator(speed_log, tp, xml_path, runtime))
