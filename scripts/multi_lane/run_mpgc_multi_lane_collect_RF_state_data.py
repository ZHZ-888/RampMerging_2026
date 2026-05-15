'''
run_mpgc_multi_lane_collect_RF_state_data.py
this version is for collecting RF train data
multi-lane simulation
test longer platoon
'''

import pandas as pd
from functions import vehicle_generation3 as vg
from functions import merging_control_regular as vc
# from functions import data_recording_pf as dr
from functions import data_recording as dr

from functions import print_control as prc # the shared fuction of print control
from functions import action_manager as vchl
from functions import merging_control_jam as vchj
# Updated imports - using new modular structure
from functions import platoon_basic as pbasic
from functions import platoon_oversized_handler as poversized
from functions import platoon_sparse_handler as psparse
from functions import platoon_lane_manager as plane

import argparse # for changing parameter setting on HPC

# RL modules
# from rl_model.rl_module import RLScoringAgent
# from rl_model import main_agent_handler as ah

def main(av_p, r_fr, m_fr, seed, loss_rate=0, gui=False, plot=False, display=False, lc_hv=False, lc_av=False, st=1000):
    # SUMO SETTING
    # path = '/home/zzha/PycharmProjects/RoadNetwork/multi_lane_motorway/multi_lane_motorway1_lefthand_noLaneChanging.sumocfg'
    # path = 'road_network/multi_lane_motorway/cfg_pf.sumocfg'
    path = '../../road_network/multi_lane_motorway/real/cfg_multi_lane_merge.sumocfg'
    sumo_config_path \
        = path
    # Simulation step length
    sim_step = 0.1
    # Determine the SUMO binary based on whether GUI is needed
    sumo_bin = 'sumo-gui' if gui else 'sumo'
    # Construct the SUMO command and options
    output_filename = f"/home/zzha/PycharmProjects/RampMerging3/Traj/trj_ml_{r_fr}_{av_p}_{seed}.xml"
    sumo_cmd = [sumo_bin, "-c", sumo_config_path,
                # "--fcd-output", output_filename, # save path
                "--no-warnings"] # , '-S' start auto, and quit auto
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

        av_p0 = av_p
        av_p1 = av_p
        m0_dpt_type = vg.get_schedule_motorway(st, av_p0, m_fr, seed)
        m1_dpt_type = vg.get_schedule_motorway(st, av_p1, m_fr, 100 - seed)
        # ramp road veh depature schedule
        vgvg = vg.VehGen(traci) # function related to veh generation
        drdr = dr.DataRecording(traci, sim_step)
        vcfunc = vc.Func1(traci, drdr)  # vehicle control; put instance of dr into vc
        vch = vchl.ActionParamsExecute(drdr, vcfunc, loss_rate) # vehicle control high level
        vcjfunc = vchj.Func(traci, drdr, vcfunc, loss_rate) # during jam

        # Initialize new modular platoon formation components
        p_basic = pbasic.PlatoonBasic(traci, vcfunc)
        p_oversized = poversized.PlatoonOversizedHandler(traci, vcfunc, p_basic)
        p_sparse = psparse.PlatoonSparseHandler(traci, vcfunc, p_basic)
        p_lane = plane.PlatoonLaneManager(traci, vcfunc)

        # RL module
        dic_follower_state, his_dic_platoon_size, dic_id_features = \
            loop(traci, st, vgvg, vcfunc, drdr, vch, vcjfunc, p_basic, p_oversized, p_sparse, p_lane,
                 lc_hv, lc_av, m0_dpt_type, m1_dpt_type)
    finally:
        traci.close()
    return dic_follower_state, his_dic_platoon_size, dic_id_features

def loop(traci, st, vgvg, vcfunc, drdr, vch, vcjfunc, p_basic, p_oversized, p_sparse, p_lane,
         lc_hv=False, lc_av=False, m0_dpt_type=None, m1_dpt_type=None):
    # START SIMULATION
    step = 0
    # scripts loop
    dic_insertedAV = {} # {AV_id: type, ...}, type = 'split' or 'free', candidates
    while step < st*10:
        # checkpoint
        if step > 160*10:
            pass
        if step%10 == 0:
            pass
        traci.simulationStep()  # start simulation
        c_ts = traci.simulation.getTime()  # current_timestep
        if c_ts % 1 == 0:
            prc.print_message("********************")
            prc.print_message("********************")
            prc.print_message(f'current_time, step:{c_ts, step}')

        # vehicle generation
        vgvg.veh_gen_ml(step, m1_dpt_type, 'm', 'route_m', 27.5, '1')  # 30m/s => 110km/h
        vgvg.veh_gen_ml(step, m0_dpt_type, 'm', 'route_m', 27.5, '0')  # 25m/s => 90km/h; ori 29.5

        # obtain veh information
        dic_vehinfo = drdr.record_vehinfo()
        ls_vehid = dic_vehinfo['ls_vehid']  # tuple, all vehicle in this step
        length_ih = drdr.length_ih # obtain the length of inflow_highway

        # on inflow_highway
        ls_ihA = dic_vehinfo['ls_ihA'] # all veh on inflow_highway, big => small
        # update
        ls_ihA_av = dic_vehinfo['ls_ihA_av']
        ls_ihA_hv = dic_vehinfo['ls_ihA_hv']  # small => big
        ls_ihB = dic_vehinfo['ls_ihB']
        ls_ihB_av = dic_vehinfo['ls_ihB_av']
        ls_ihAB_hv = dic_vehinfo['ls_ihAB_hv']
        ls_wsBC_hv = dic_vehinfo['ls_wsBC_hv']
        ls_centerA_av = dic_vehinfo['ls_centerA_av']
        ls_centerB_av = dic_vehinfo['ls_centerB_av']

        # SC1: find oversized platoon
        dic_id_type, ls_leader_AV, ls_follower_AV, dic_AVroleChange \
            = p_basic.tag_vehicles13(ls_ihA, max_team_size=11) # SPLIT_PROMOTE
        his_dic_platoon_size, dic_platoon_size, dic_platoon_members \
            = p_basic.get_platoon_size3(ls_ihA, ls_leader_AV)
        dic_current_oversizedP, dic_current_upBav, dic_nonOversizedP \
            = p_oversized.find_oversizedP_nearbyAV(ls_ihB_av, dic_platoon_size, dic_platoon_members)

        # SC2: Predict (find) free followers
        dic_id_preState, dic_id_features = p_sparse.predict_flw_state(dic_id_type, ls_vehid, model=True)
        dic_sparseP = p_sparse.find_sparse_platoon(dic_nonOversizedP, dic_id_preState)
        # sparseP => sparse_platoon = {av_leader:first_free_hv}
        promote_av = p_sparse.free_promote(dic_sparseP, dic_platoon_members) # FREE_PROMOTE

        # restric av lc behaviour
        ls_av = ls_ihA_av + ls_ihB_av
        p_lane.restrict_av_lc(lc_av, ls_av)
        # encourage innner lane change which lane_0 hv close to ramp entry
        p_lane.manage_lc_behavior_near_ws(lc_hv, ls_ihAB_hv, ls_wsBC_hv, length_ih, p_to_inner=0.8)
        # record target value
        ls_follower_op, dic_follower_state = p_basic.record_follower_state2(step, length_ih, dic_id_type, ls_ihA)

        # control gaps between platoons
        p_basic.form_platoon3(ls_leader_AV, ls_follower_AV)
        p_basic.restore_speed_limit2(ls_centerA_av)
        step += 1
    return dic_follower_state, his_dic_platoon_size, dic_id_features

if __name__ == '__main__':
    # python run_mpgc_multi_lane_collect_RF_state_data.py --nogui --st 30000
    parser = argparse.ArgumentParser()
    # default True, add "--nogui" False
    parser.add_argument("--nogui", action="store_true", help="disable SUMO GUI")
    parser.add_argument("--st", type=int, default=None,
                        help="simulation duration; default = internal value")
    args = parser.parse_args()

    prc.PRINT_ENABLED = False
    default_st = 3600  # 1000
    st = args.st if args.st is not None else default_st
    av_p = 0.1 # 0.2
    r_fr = 810 # (540, 360)
    m_fr = 1080 # 1080
    seed = 21 # 4
    loss_rate = 0
    gui = not args.nogui
    gui = False
    plot = True
    display = True
    lc_hv = False # if consider HV lane-changing
    lc_av = False
    print(f"=======gui:{gui},st:{st}========")
    dic_follower_state, his_dic_platoon_size, dic_id_features \
        = main(av_p, r_fr, m_fr, seed, loss_rate, gui, plot, display, lc_hv, lc_av, st)

    # record features and real states
    feature_target = [features + dic_follower_state.get(key, ['unknown']) for key, features in dic_id_features.items()]
    columns = ['id', 'dis_to_pv', 'v_pv', 'dis_leaderAV', 'pos_this', 'size', 'leaderAV_id_start', 'state',
               'leaderAV_id_end']
    df_fea_tar = pd.DataFrame(feature_target, columns=columns)
    # df_fea_tar.to_csv("data/features/df_fea_tar_multi_merging_251118.csv", index=False)
    df_fea_tar.to_csv("data/features/df_fea_tar_new251121.csv", index=False)

    # indicator 1: num.platoon_followers/num.followers %
    num_follower = len(dic_follower_state)
    num_platoon_follower = len([k for k, v in dic_follower_state.items() if v[0] == 'following_mode'])
    print(f"index1: {num_platoon_follower} platoon_followers, {num_follower} followers, "
          f"ratio: {num_platoon_follower / num_follower * 100:.1f}%")

    # indicator 2: num.normal_size_platoon/num.platoon
    num_platoon = len(his_dic_platoon_size)
    num_normal_size_platoon = len([k for k, v in his_dic_platoon_size.items() if v <= 11])
    print(f"index2: {num_platoon} num_platoon, {num_normal_size_platoon} num_normal_size_platoon, "
          f"ratio: {num_normal_size_platoon / num_platoon * 100:.1f}%")

    ls_av_p = [0.3] # 0.1, 0.2, 0.3
    ls_seed = list(range(1, 31))
    ls_r_fr = [180, 270, 360, 450, 540, 630, 720, 810, 900, 990]
