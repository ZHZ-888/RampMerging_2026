# 250810: add HV heter
# no control strategy, following first in first out principle
import pandas as pd
import os
import time
from functions import vehicle_generation2 as vg
from functions import vehicle_generation_hv as vgh
from functions import data_recording as dr


def main(av_p, r_fr, m_fr, seed, gui=False, plot=False, st=1000):
    # SUMO SETTING
    # path = '/home/zzha/PycharmProjects/RoadNetwork/merge_rode12_shapeChanged.sumocfg'
    # path = '/home/zzha/PycharmProjects/RoadNetwork/single_lane_merge/merge_road_250621_shapeChanged.sumocfg'
    # path = 'road_network/single_lane_merge/merge_road_250811_fifo.sumocfg'
    path = '../../road_network/single_lane_merge/cfg_merge_fifo.sumocfg'
    sumo_config_path = path
    # Simulation step length
    sim_step = 0.1
    # Determine the SUMO binary based on whether GUI is needed
    sumo_bin = 'sumo-gui' if gui else 'sumo'
    # Construct the SUMO command and options
    traj_dir = '../../data/fifo/traj_fifo'
    file_name = f'trj_{r_fr}_{av_p}_{seed}.xml'
    output_filename = os.path.join(traj_dir, file_name)
    # Construct the SUMO command and options
    sumo_cmd = [sumo_bin, "-c", sumo_config_path,
                "--fcd-output", output_filename,
                "--no-warnings"] # "data/Traj/trajectory_fifo_test.xml"
    sumo_options = ["--step-length", str(sim_step)]
    # If GUI is enabled, set the GUI view schema
    if gui:
        import traci
        traci.start(sumo_cmd + sumo_options)
        # traci.gui.setSchema('View #0', "real world")
    else:
        import libsumo as traci
        traci.start(sumo_cmd + sumo_options)
    try:
        # VEHICLE GENERATOR
        # generate departure sequence and corresponding fleet types
        # scripts road veh depature schedule
        max_attempts = 8
        m_dpt_type = vgh.get_depature_timels(st, m_fr, seed)
        # ramp road veh depature schedule
        r_dpt_type = vgh.get_depature_timels(st, r_fr, seed)
        drdr = dr.DataRecording(traci, sim_step)
        vgvg = vg.VehGen(traci) # function related to veh generation
        # get dic_avhid_ptype
        dic_id_speed, data_veh, ls_hinfo, tp = loop(traci, st, vgvg, drdr, m_dpt_type, r_dpt_type)
        ls_hinfo_columns = ['veh_id', 'leader_id', 'headway', 'time_headway', 'time']
        df_hinfo = drdr.transform_ls_df(ls_hinfo, ls_hinfo_columns)
    finally:
        traci.close() # Ensure SUMO simulation is properly closed to release memory and system resources
    return dic_id_speed, data_veh, df_hinfo, tp


def loop(traci, st, vgvg, drdr, m_dpt_type=None, r_dpt_type=None):
    # START SIMULATION
    step = 0
    # edge_id, inflow_merge, center
    ramp_entry_edge = 'inflow_merge'
    ramp_exit_edge = 'center'
    # scripts loop
    while step < st*10:
        if step == 200:
            pass
        traci.simulationStep()  # start simulation
        c_ts = traci.simulation.getTime()  # current_timestep
        # add scripts road veh
        vgvg.veh_gen_hv2(step, m_dpt_type, 'm', 'route1', 24.5)
        vgvg.veh_gen_hv2(step, r_dpt_type, 'r', 'route2', 10)

        # get veh info
        dic_vehinfo = drdr.record_vehinfo()
        ls_veh_id = dic_vehinfo['ls_vehid']
        dic_id_speed = drdr.record_vehSpeed(ls_veh_id)
        data_veh = drdr.record_vehData(ls_veh_id, c_ts, dic_id_speed)

        # 2024.8.25, get headway of each vehicle
        dic_hinfo = drdr.get_veh_headwayinfo(ls_veh_id, dic_id_speed)
        ls_hinfo = drdr.organize_veh_hinfo(c_ts, dic_hinfo)
        tp = drdr.record_throughput(st, ls_veh_id, 'center')  # throughput

        step += 1
    return dic_id_speed, data_veh, ls_hinfo, tp


if __name__ == '__main__':
    st = 1200
    av_p = 0
    r_fr = 720
    m_fr = 1080
    seed = 1
    gui = False
    plot = False
    start = time.time()
    dic_id_speed, data_veh_info, df_hinfo, tp = main(av_p, r_fr, m_fr, seed, gui, plot, st)
    end = time.time()
    all_speeds = sum(dic_id_speed.values(), [])
    average_v = sum(all_speeds) / len(all_speeds)
    print(f'average_v:{average_v} m/s, out_throughput: {tp} veh/h')
    print(f'execution_time:{end - start:.1f} s')
    # get df_vehinfo
    df_vehinfo = pd.DataFrame(data_veh_info, columns=["veh_id", "time", "speed", "dis"])
    print(df_vehinfo)
    # df_vehinfo.to_csv("data/veh_info241119_withoutStrag.csv")
    # 240825.df of headway info
    df_hinfo2 = df_hinfo.dropna(subset=['leader_id'])
    # df_hinfo2.to_csv("headway_info_FIFO_630.csv", index=False)
    print(df_hinfo2)
