class DetectorStateRecogniser:
    """
    Detector-based vehicle count.

    This class uses SUMO induction loop detectors located at the entrance of
    the merging control zone (MCZ).

    The recogniser does not require HV driving-style parameters, real-time HV
    speed, or continuous spacing information. It only uses the vehicle passing
    time at the detector and the ordered platoon member list.

    Logic:
    Count the vehicle number after a leader;

    """

    def __init__(self, traci, detector_ids, temporal_headway_threshold=3.0):
        """
        Args:
            traci: Active SUMO TraCI connection.
            detector_ids: List of SUMO induction loop detector IDs.
            temporal_headway_threshold: Temporal headway threshold in seconds.
        """

        self.traci = traci
        self.detector_ids = detector_ids # the id of detector
        self.temporal_headway_threshold = temporal_headway_threshold

        # Passing time of each vehicle at the detector
        self.dic_pass_time = {}

        # Detected state of each vehicle
        self.dic_id_preState = {}

        # Temporal headway of each vehicle to its preceding platoon member
        self.dic_id_temporal_headway = {}

    def update(self, step, dic_platoon_members):
        """
        Update detector records and platoon member states.

        This function should be called at every simulation step. However,
        vehicle passing time is recorded only when a vehicle passes the detector.

        Args:
            dic_platoon_members: Ordered platoon members.
                Format: {leader_id: [leader_id, follower1, follower2, ...]}
        """

        c_ts = round(step/10 + 0.1, 1)

        # Step 1: Record vehicles that passed the detector in the last step
        for detector_id in self.detector_ids:
            try:
                passed_vehicle_ids = (
                    self.traci.inductionloop.getLastStepVehicleIDs(detector_id)
                )
            except self.traci.TraCIException:
                continue

            for veh_id in passed_vehicle_ids:
                # Record the first detector passing time only
                if veh_id not in self.dic_pass_time:
                    self.dic_pass_time[veh_id] = c_ts

        # Step 2: Update platoon states based on detector passing times
        self._update_platoon_member_states(dic_platoon_members)

    def _update_platoon_member_states(self, dic_platoon_members):
        """
        Update follower states based on platoon order and detector passing time.

        Once a follower is classified as free_mode, all vehicles behind it in
        the same platoon are also classified as free_mode.
        """

        for leader_id, members in dic_platoon_members.items():

            if not members:
                continue

            # The leader itself is not a follower. If it has passed the detector,
            # mark it as leader_mode for completeness.
            if leader_id in self.dic_pass_time:
                self.dic_id_preState[leader_id] = 'leader_mode'
                self.dic_id_temporal_headway[leader_id] = None

            platoon_broken = False

            for i in range(1, len(members)):
                veh_id = members[i]
                prev_id = members[i - 1]

                # If the follower has not passed the detector yet,
                # do not update its state.
                if veh_id not in self.dic_pass_time:
                    continue

                # If an upstream follower has already broken the platoon,
                # all downstream followers are treated as free_mode.
                if platoon_broken:
                    self.dic_id_preState[veh_id] = 'free_mode'
                    self.dic_id_temporal_headway[veh_id] = None
                    continue

                # If the preceding platoon member has not passed the detector,
                # the current follower cannot be confirmed as attached.
                if prev_id not in self.dic_pass_time:
                    self.dic_id_preState[veh_id] = 'free_mode'
                    self.dic_id_temporal_headway[veh_id] = None
                    platoon_broken = True
                    continue

                temporal_headway = (
                    self.dic_pass_time[veh_id] - self.dic_pass_time[prev_id]
                )

                self.dic_id_temporal_headway[veh_id] = temporal_headway

                if 0 <= temporal_headway <= self.temporal_headway_threshold:
                    self.dic_id_preState[veh_id] = 'following_mode'
                else:
                    self.dic_id_preState[veh_id] = 'free_mode'
                    platoon_broken = True

    def get_state(self, veh_id):
        """
        Return the detected state of a vehicle.

        If the vehicle has not passed the detector or has not been classified yet,
        it is treated as free_mode by default.
        """

        return self.dic_id_preState.get(veh_id, 'free_mode')

    def get_temporal_headway(self, veh_id):
        """
        Return the temporal headway of a vehicle to its preceding platoon member.

        Returns None if the temporal headway is not available.
        """

        return self.dic_id_temporal_headway.get(veh_id, None)

    def has_passed_detector(self, veh_id):
        """
        Check whether a vehicle has passed the detector.
        """

        return veh_id in self.dic_pass_time

    def get_pass_time(self, veh_id):
        """
        Return the detector passing time of a vehicle.

        Returns None if the vehicle has not passed the detector.
        """

        return self.dic_pass_time.get(veh_id, None)