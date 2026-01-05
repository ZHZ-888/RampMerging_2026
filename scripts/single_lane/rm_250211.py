import os
import time
from functions import vehicle_generation2 as vg
from functions import vehicle_generation_hv as vgh
from functions import data_recording_rm as dr
from functions import print_control as prc # the shared function of print control
from comparsion_algo import ramp_metering_algo2 as rma

def main(av_p, r_fr, m_fr, seed, gui=False, plot=False, st=1000):
    # SUMO SETTING
    # path = '/home/zzha/PycharmProjects/RampMerging4_250208/road_network/merge_rode13_rampmeter_acc.sumocfg'
    # path = 'road_network/single_lane_merge/merge_road13_rm_acc.sumocfg'
    path = '../../road_network/single_lane_merge/cfg_merge_rm.sumocfg'
    sumo_config_path = path
    # Simulation step length
    sim_step = 0.1
    # Determine the SUMO binary based on whether GUI is needed
    sumo_bin = 'sumo-gui' if gui else 'sumo'
    # Construct the SUMO command and options
    traj_dir = '../../data/rm/traj_rm'
    file_name = f'trj_{r_fr}_{av_p}_{seed}.xml'
    output_filename = os.path.join(traj_dir, file_name)
    # Construct the SUMO command and options
    sumo_cmd = [sumo_bin, "-c", sumo_config_path,
                "--fcd-output", output_filename,
                "--no-warnings"]
    sumo_options = ["--step-length", str(sim_step)]
    # If GUI is enabled, set the GUIcomparsion_algo' view schema
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
        # generate only hv
        ls_m_dpt = vgh.get_depature_timels(st, m_fr, seed)
        ls_r_dpt = vgh.get_depature_timels(st, r_fr, seed)
        vgvg = vg.VehGen(traci) # function related to veh generation
        rmaf = rma.Func(traci) # function related to ramp_metering algo
        drdr = dr.DataRecording(traci, sim_step)
        dic_id_speed, data_veh, ls_hinfo, tp = loop(traci, st, vgvg, drdr, rmaf, ls_m_dpt, ls_r_dpt)
        # df_vsj = pd.DataFrame(ls_vsj_c, columns=['step', 'r_avg_speed', 'm_avg_speed', 'avg_speed', 'jam_mode'])
        # df_vsj.to_csv("/home/zzha/PycharmProjects/RampMerging3/data/vsj_rm_0122_4.csv", index=False)
        # 240825
        ls_hinfo_columns = ['veh_id', 'leader_id', 'headway', 'time_headway', 'time']
        df_hinfo = drdr.transform_ls_df(ls_hinfo, ls_hinfo_columns)  # dataframe of hinfo
    finally:
        traci.close()
    return dic_id_speed, data_veh, df_hinfo, tp


def loop(traci, st, vgvg, drdr, rmaf, ls_m_dpt=None, ls_r_dpt=None):
    # START SIMULATION
    step = 0
    ls_vsj_c = []  # collection of average_velocity of every step and its jam_state
    # edge_id, inflow_merge, center
    ramp_entry_edge = 'inflow_merge'
    ramp_exit_edge = 'center'
    target_density = 25 # veh_num/km
    gain = 20
    p_release_rate = 200 # veh_num/h

    accident_state = False
    # scripts loop
    while step < st*10:
        if step == 200:
            pass
        traci.simulationStep()  # start simulation
        prc.print_message("\n**************")
        c_ts = traci.simulation.getTime()  # current_timestep
        prc.print_message(f'current_time, step:{c_ts, step}')
        # only hv
        # vgvg.veh_gen_hv(step, ls_m_dpt, 'm', 'route1', 24.5)
        # vgvg.veh_gen_hv(step, ls_r_dpt, 'r', 'route2', 10)
        vgvg.veh_gen_hv2(step, ls_m_dpt, 'm', 'route1', 24.5)
        vgvg.veh_gen_hv2(step, ls_r_dpt, 'r', 'route2', 10)

        # get veh info
        dic_vehinfo = drdr.record_vehinfo()
        ls_veh_id = dic_vehinfo['ls_vehid']
        ls_vehid = dic_vehinfo['ls_vehid'] # list of all veh id
        dic_id_speed = drdr.record_vehSpeed(ls_vehid)
        data_veh = []

        # get current density
        main_id = 'inflow_highway_0'
        ramp_id = 'inflow_merge_0'

        # 2024.8.25, get headway of each vehicle
        dic_hinfo = drdr.get_veh_headwayinfo(ls_veh_id)
        ls_hinfo = drdr.organize_veh_hinfo(c_ts, dic_hinfo)
        # list of headway info [veh_id, leader_id, headway, time_headway, time]
        tp = drdr.record_throughput(st, ls_veh_id, 'center')  # throughput

        # simulate a sudden accident
        # accident_state = ac.sudden_accident(traci, ls_veh_id, 200, 100, accident_state)
        if c_ts % 90 == 0:
            des_center = rmaf.get_current_density("center_0") # center_0; current center flow
            des_ramp = rmaf.get_current_density("inflow_merge_0")

            c_release_rate = rmaf.alinea_control(des_center, target_density, gain, p_release_rate)
            prc.print_message(f'c_release_rate: {c_release_rate}')

            # set the green time of the signal
            g_time, r_time = rmaf.set_ramp_metering_signal('center', c_release_rate)

            # update last release_rate
            p_release_rate = c_release_rate  # p-previous, c-current
        try:
            prc.print_message(f'g_time: {g_time}, r_time:{r_time}')
            prc.print_message(f'center:{des_center},main:{des_ramp}')
        except:
            pass

        id_on_ramp = traci.edge.getLastStepVehicleIDs(ramp_entry_edge)
        id_all = traci.vehicle.getIDList()

        jam_mode = False
        ls_vsj = drdr.get_average_speed(step, jam_mode, ls_veh_id)  # average_velocity of this step and its jam_state
        ls_vsj_c.append(ls_vsj)  # collect into one list

        step += 1
    return dic_id_speed, data_veh, ls_hinfo, tp


if __name__ == '__main__':
    prc.PRINT_ENABLED = False
    st = 1200
    av_p = 0
    r_fr = 720
    m_fr = 1080
    seed = 1
    gui = False
    plot = False
    start = time.time()
    dic_id_speed, data_veh, df_hinfo, tp = main(av_p, r_fr, m_fr, seed, gui, plot, st)
    end = time.time()
    all_speeds = sum(dic_id_speed.values(), [])
    average_v = sum(all_speeds) / len(all_speeds)
    print(f'average_v:{average_v}')
    print(f'execution_time:{end - start:.1f} s')
    # 240825.df of headway info
    df_hinfo2 = df_hinfo.dropna(subset=['leader_id'])
    # df_hinfo2.to_csv("headway_info_rm_630.csv", index=False)
    print(df_hinfo2)

