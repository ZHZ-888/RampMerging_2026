import math

class Func:
    def __init__(self, traci):
        self.traci = traci
        # --- minimal knobs with safe defaults ---
        self.K = 70.0                 # >0 for ALINEA
        self.use_occupancy = True     # True: use occupancy for "density" input (0~1); False: veh/km
        self.target_occupancy = 0.20  # only used when use_occupancy=True
        self.target_density = 28.0    # veh/km if use_occupancy=False

        # release-rate bounds (veh/h)
        self.r_min = 0.0
        self.r_max = 1800.0
        self._r_prev = 900.0          # start from a mid value

        # signal timing
        self.cycle_time = 20          # s
        self.yellow_time = 3          # s
        self.sat_flow = 1900.0        # veh/h-of-green on ramp
        self.min_green = 1.0          # s
        self.min_red   = 2.0          # s

        # if TLS controls multiple links, which index is the ramp movement (0-based)
        self.ramp_conn_index = 0

    # ---------- ALINEA control ----------
    def alinea_control(self, current_density, target_density, gain, previous_release_rate):
        """
        ALINEA control law: r_k = r_{k-1} + K * (target - current)
        NOTE: use K>0. This function keeps your original signature.
        """
        # choose error source
        if self.use_occupancy:   # treat current_density as occupancy 0~1
            error = self.target_occupancy - float(current_density)
        else:                     # treat current_density as veh/km
            error = self.target_density - float(current_density)

        r = self._r_prev + self.K * error
        r = max(self.r_min, min(self.r_max, r))
        self._r_prev = r
        return r

    # ---------- legacy on/off (kept) ----------
    def set_ramp_signal(self, ramp_signal_id, rate):
        self.traci.trafficlight.setRedYellowGreenState(
            ramp_signal_id, "G" if rate <= 0.5 else "r"
        )

    # ---------- measurement helper (kept name) ----------
    def get_current_density(self, lane_id):
        """
        If use_occupancy=True, returns lastStepOccupancy (0~1).
        Else returns density (veh/km).
        """
        if self.use_occupancy:
            try:
                return self.traci.lane.getLastStepOccupancy(lane_id)  # 0~1
            except Exception:
                # very crude fallback from veh/km -> occupancy
                veh = self.traci.lane.getLastStepVehicleNumber(lane_id)
                Lkm = max(1e-6, self.traci.lane.getLength(lane_id) / 1000.0)
                dens = veh / Lkm
                return min(1.0, dens / 140.0)
        else:
            veh = self.traci.lane.getLastStepVehicleNumber(lane_id)
            Lkm = max(1e-6, self.traci.lane.getLength(lane_id) / 1000.0)
            return veh / Lkm

    # ---------- rate -> signal timing (fixed) ----------
    def set_ramp_metering_signal(self, ramp_meter_id, release_rate):
        """
        Single-lane ramp: enforce "1 vehicle per green".
        Uses ALINEA's release_rate (veh/h) to determine the average headway,
        and adjusts the red time to meet that headway.
        The green time is fixed to allow exactly one vehicle to pass,
        based on saturation flow and a start-up lost time.
        """
        import math

        C_min = 5  # minimum total cycle time (s) to avoid unstable very short cycles
        Y = int(self.yellow_time)  # yellow time (s)

        # 1) Target: one vehicle per green phase
        vehicles_per_green = 1
        # Effective green time for one vehicle:
        #   ideal discharge time = 3600 / sat_flow (s/veh)
        #   + start-up lost time (~0.5–0.8 s, empirical)
        start_loss = 0.7
        g_one = vehicles_per_green * (3600.0 / max(1e-6, self.sat_flow)) + start_loss
        # Ensure minimum green
        g = max(self.min_green, g_one)

        # 2) From release_rate (veh/h) -> target average headway (s/veh)
        r = max(1e-6, float(release_rate))
        headway = 3600.0 / r  # seconds per vehicle

        # 3) Use red time to achieve the target headway:
        #    headway = green + yellow + red
        R_raw = headway - g - Y
        R = max(self.min_red, R_raw)

        # 4) Round to integers and ensure total time = green + yellow + red
        G = int(round(g))
        R = int(round(R))
        total = G + Y + R
        if total < C_min:
            # If total cycle is too short, increase red to meet minimum
            R = C_min - (G + Y)
        # Enforce non-negative and minimum times
        G = max(int(math.ceil(self.min_green)), G)
        R = max(int(math.ceil(self.min_red)), R)

        # 5) Build TLS state strings:
        #    - All-red except the ramp connection index is set to G/y/r
        num_links = len(self.traci.trafficlight.getControlledLinks(ramp_meter_id)) or 1
        idx = min(max(0, int(self.ramp_conn_index)), num_links - 1)

        def all_red():
            return "r" * num_links

        def with_char(ch):
            s = ["r"] * num_links
            s[idx] = ch
            return "".join(s)

        # 6) Create the TLS program and apply it  —— libsumo-safe
        Phase = self.traci.trafficlight.Phase
        Logic = self.traci.trafficlight.Logic  # in libsumo this is TraCILogic

        phases = [
            Phase(G, with_char('G')),
            Phase(Y, with_char('y')),
            Phase(R, all_red()),
        ]

        # IMPORTANT: use positional args, not keywords
        # signature: Logic(programID, type, currentPhaseIndex, phases)
        program = Logic('alinea_1veh', 0, 0, phases)

        self.traci.trafficlight.setCompleteRedYellowGreenDefinition(ramp_meter_id, program)
        return (G, R)

