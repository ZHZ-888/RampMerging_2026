def sudden_accident(traci, vehicle_ids, stop_time, stop_position, accident_state):
    '''

    :param stop_time:
    :param stop_position: on the merge lane (center_0), suggest stop_pos = 50
    :return:
    '''
    c_ts = traci.simulation.getTime()
    stop_duration = 60 # second
    if c_ts > stop_time and accident_state == False:
        for id in vehicle_ids:
            lane = traci.vehicle.getLaneID(id)
            pos = traci.vehicle.getLanePosition(id)
            speed = traci.vehicle.getSpeed(id)
            if pos < stop_position and lane == 'center_0':
                try:
                    traci.vehicle.setStop(id, 'center', stop_position, laneIndex=0, duration=stop_duration)
                    accident_state = True
                    print(f'stop_id:{id}')
                    break
                except:
                    accident_state = False
                    continue
    return accident_state