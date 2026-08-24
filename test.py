def generate_entry_arrivals_shifted_exp(st, p, fr, seed=None):
    """
    Generate vehicle arrival schedule with:
      - hard constraint: headway >= min_interval (seconds)
      - long-run mean flow approximately equals fr (veh/h)
      - stochasticity: shifted exponential headways

    Returns
    -------
    dict: dic of departure_time and fleet_type
        {arrival_time (rounded to 1 decimal): vehicle_type}
        {1.6: 'HV', 2.6: 'HV', 4.2: 'HV', 5.2: 'HV', 7.5: 'HV'...}
    """
    min_interval = 1.0

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    lam = fr / 3600.0  # veh/s
    if lam <= 0:
        return {}

    mean_headway_target = 1.0 / lam  # seconds
    if min_interval >= mean_headway_target:
        raise ValueError(
            f"Infeasible: min_interval ({min_interval}) must be < target mean headway ({mean_headway_target:.4f}). "
            f"Otherwise you cannot achieve fr={fr} veh/h."
        )

    # Calibrate exponential part
    lam2 = 1.0 / (mean_headway_target - min_interval)

    result = {}
    t = 0.0
    while True:
        extra = np.random.exponential(scale=1.0 / lam2)
        headway = min_interval + extra   # strictly >= min_interval
        t += headway
        if t >= st:
            break

        t_key = round(t, 1)  # keep your original output style
        veh_type = "AV" if (np.random.rand() < p) else "HV"
        result[t_key] = veh_type

    # print info
    sh = st / 3600  # simulation time (s) => simulation hour (h)
    expect_veh_num = int(fr * sh)
    expect_av_num = int(fr * sh * p)
    gen_veh_num = len(result)
    gen_av_num = sum(v == 'AV' for v in result.values())
    gen_demands = int(gen_veh_num/sh)

    print('           ---generation overview---')
    print(f'expect_demands: {fr} veh/h, gen_demands: {gen_demands} veh/h')
    print(f'expect_veh_num: {expect_veh_num}, gen_veh_num: {gen_veh_num}, diff: {expect_veh_num - gen_veh_num}')
    print(f'expect_av_num: {expect_av_num}, gen_av_num: {gen_av_num}, diff: {expect_av_num - gen_av_num}')
    print(f"Target av_p: {expect_av_num / expect_veh_num:.2f}, av_p: {gen_av_num / gen_veh_num:.2f}")

    return result