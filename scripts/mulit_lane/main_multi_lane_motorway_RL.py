# main_multi_lane_motorway_RL.py
'''
multi-lane simulation
test longer platoon
'''
import os
import time
import pandas as pd
from functions import vehicle_generation3 as vg
from functions import print_control as prc  # the shared fuction of print control
from functions import formation_controller as fc
from functions import merging_controller as mc
from functions import calc_ttc_sd_exposure
from functions import data_recording as dr


def main(av_p, r_fr, m_fr, seed, r_autoFollow_p, r_platoon_p, loss_rate=0,
         gui=False, plot=False, display=False, lc=False, st=1000):
    # SUMO SETTING
    # path = '/home/zzha/PycharmProjects/RoadNetwork/multi_lane_motorway/multi_lane_motorway1_lefthand_noLaneChanging.sumocfg'
    # path = 'road_network/multi_lane_motorway/cfg_pf.sumocfg'
    path = '../../road_network/multi_lane_motorway/real/cfg_multi_lane_merge.sumocfg'
    sumo_config_path = path
    # Simulation step length
    sim_step = 0.1
    # Determine the SUMO binary based on whether GUI is needed
    sumo_bin = 'sumo-gui' if gui else 'sumo'
    # Construct the SUMO command and options
    traj_dir = os.environ.get("TRAJ_DIR", "../../data/multi_lane/algo")  # default 'data/mpgc'
    file_name = f'trj_{r_fr}_{av_p}_{seed}_{loss_rate}.xml'
    xml_path = os.path.join(traj_dir, file_name)
    sumo_cmd = [sumo_bin, "-c", sumo_config_path,
                "--fcd-output", xml_path, # save path
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
        # av_p0 = 0.1 # only for RL agent training, for creating more oversized platoon
        # av_p1 = 0.2
        data_recorder = dr.DataRecording(traci)
        max_attempts = 7
        av_p0 = av_p
        av_p1 = av_p
        m0_dpt_type = vg.get_schedule_motorway(st, av_p0, m_fr, seed)
        m1_dpt_type = vg.get_schedule_motorway(st, av_p1, m_fr, 100 - seed)

        r_dpt_type = vg.get_schedule2(st, av_p, r_fr, r_platoon_p, max_attempts, plot, seed, display)
        # ramp road veh depature schedule
        vgvg = vg.VehGen(traci)  # function related to veh generation
        data_recorder.get_avhid_ptype(r_dpt_type = r_dpt_type)  # here only have r_dpt_type

        formation_controller = fc.FormationController(data_recorder, traci)
        merging_controller = mc.MergingController(data_recorder, traci, av_p, platoon_formation=True, ml=True)

        (dic_score_reward, dic_follower_state, his_dic_platoon_size, dic_id_features,
         tp, speed_log, queue_log) = \
            loop(traci, st, vgvg, formation_controller, merging_controller, lc,
                 r_autoFollow_p, m0_dpt_type, m1_dpt_type, r_dpt_type)
    finally:
        traci.close()
    return (dic_score_reward, dic_follower_state, his_dic_platoon_size, dic_id_features,
            tp, speed_log, queue_log, xml_path)


def loop(traci, st, vgvg, formation_controller, merging_controller, lc, r_autoFollow_p,
         m0_dpt_type=None, m1_dpt_type=None, r_dpt_type=None):
    # START SIMULATION
    step = 0
    # scripts loop
    while step < st * 10:
        # checkpoint
        if step > 1126 * 10:
            pass
        traci.simulationStep()  # start simulation
        c_ts = traci.simulation.getTime()  # current_timestep
        if c_ts % 1 == 0:
            prc.print_message(f'************current_time, step:{c_ts, step}************')

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

        # vehicle generation
        vgvg.veh_gen_ml(step, m0_dpt_type, 'm', 'route0', 27.5, '0')  # 25m/s => 90km/h; ori 29.5
        vgvg.veh_gen_ml(step, m1_dpt_type, 'm', 'route0', 27.5, '1')  # 30m/s => 110km/h
        # ramp vehicle generation
        vgvg.veh_gen_heter2(step, r_dpt_type, 'r', r_autoFollow_p)

        (dic_score_reward, dic_follower_state, his_dic_platoon_size, dic_id_features,
         dic_final_platoon_info) = formation_controller.step(st, step, lc)
        tp, speed_log, queue_log = (
            merging_controller.step(st, step, dic_final_platoon_info, r_dpt_type))
        step += 1
    return (dic_score_reward, dic_follower_state, his_dic_platoon_size, dic_id_features,
            tp, speed_log, queue_log)


if __name__ == '__main__':
    st = 1200  # 1200
    av_p = 0.1

    r_fr = 990  # 540
    r_platoon_p = 1  # percentage of platoon vehicles
    r_autoFollow_p = 1  # auto follow proportion

    m_fr = 1080  # 1080
    seed = 1  # 4
    loss_rate = 0.1
    gui = True
    plot = False
    display = True
    prc.PRINT_ENABLED = False
    lc = False  # if consider HV lane-changing

    start = time.time()
    (dic_score_reward, dic_follower_state, his_dic_platoon_size, dic_id_features,
     tp, speed_log, queue_log, xml_path) \
        = main(av_p, r_fr, m_fr, seed, r_autoFollow_p, r_platoon_p, loss_rate, gui, plot, display, lc, st)
    end = time.time()

    # record features and real states
    feature_target = [features + dic_follower_state.get(key, ['unknown']) for key, features in dic_id_features.items()]
    columns = ['id', 'dis_to_pv', 'v_pv', 'dis_leaderAV', 'pos_this', 'size', 'leaderAV_id_start', 'state',
               'leaderAV_id_end']
    df_fea_tar = pd.DataFrame(feature_target, columns=columns)
    # df_fea_tar.to_csv("data/features/df_fea_tar_multi_merging_251114_2.csv", index=False)

    # PLATOON FORMATION results
    # indicator 1: num.platoon_followers/num.followers %
    num_follower = len(dic_follower_state)
    num_platoon_follower = len([k for k, v in dic_follower_state.items() if v[0] == 'following_mode'])
    print(f"index1: {num_platoon_follower} platoon_followers, {num_follower} followers, "
          f"ratio: {num_platoon_follower / num_follower * 100:.1f}%")
    # indicator 2: num.normal_size_platoon/num.platoon
    num_platoon = len(his_dic_platoon_size)
    num_normal_size_platoon = len([k for k, v in his_dic_platoon_size.items() if v <= 11])
    print(f"index2: {num_platoon} num_platoon, {num_normal_size_platoon} num_normal_size_platoon, "
          f"ratio: {num_normal_size_platoon / num_platoon * 100:.1f}%, "
          f'execution_time:{end - start:.1f} s')

    # MERGING CONTROL results
    ttc_ratio, avg_speed_std = calc_ttc_sd_exposure.calc_ttc_and_speed_std(xml_path)
    avg_speeds = [item[1] for item in speed_log]
    clean_avg_speeds = [v for v in avg_speeds if v is not None]
    average_v = sum(clean_avg_speeds) / len(clean_avg_speeds)
    avg = (sum(q for _, q in queue_log) / len(queue_log)) if queue_log else None
    print(f'tp: {tp} veh/h, average_v:{average_v} m/s, queue_length_avg:{avg}, '
          f'ttc_ratio: {ttc_ratio}, avg_speed_std: {avg_speed_std}, '
          f'execution_time:{end - start:.1f} s')

    # save sic_score_reward as dataframe
    rows = []
    for lc_id, values in dic_score_reward.items():
        score = values[0] if len(values) > 0 else None
        reward = values[1] if len(values) > 1 else None
        rows.append((lc_id, score, reward))
    df_sr = pd.DataFrame(rows, columns=["lc_id", "score", "reward"])