"""Vehicle-level merging control. Scheduling uses AV self-reports only."""
import csv
from dataclasses import dataclass

import numpy as np
from functions.optimisation_algo import GetBVCurve2


@dataclass
class CVMCConfig:
    interval: int = 60  # 0.1 s steps
    buffer: float = 1.5
    v_floor: float = 0.1
    min_speed: float = 5.0
    mcz_length: float = 800.0
    distance_tolerance: float = 5.0
    speed_tolerance: float = 2.0
    solver_tolerance: float = 1e-3


def fifo_order(states, locked=(), v_floor=0.1):
    """Merge physical stream queues, preserving existing commitments."""
    queues = {s: sorted((a for a in states if a['stream'] == s),
                        key=lambda a: (a['distance'], a['id'])) for s in ('M', 'R')}
    ids = {a['id'] for a in states}
    pending = [vid for vid in locked if vid in ids]
    order = []
    while any(queues.values()):
        heads = [q[0] for q in queues.values() if q]
        eligible = [a for a in heads if a['id'] not in pending or a['id'] == pending[0]]
        if not eligible:
            return None  # Physical stream order contradicts the old commitment.
        chosen = min(eligible, key=lambda a: (a['distance'] / max(a['speed'], v_floor),
                                             a['stream'] != 'M', a['id']))
        order.append(chosen)
        queues[chosen['stream']].pop(0)
        if pending and chosen['id'] == pending[0]:
            pending.pop(0)
    return order


def target_time(state, previous, config):
    eta = state['time'] + state['distance'] / max(state['speed'], config.v_floor)
    if previous is None:
        return eta
    gap = config.buffer if state['stream'] != previous['stream'] else 0.0
    return max(eta, previous['target'] + gap)


def solve_trajectory(state, target, config):
    horizon = target - state['time']
    if horizon <= 0:
        return None
    opt = GetBVCurve2(state['speed'], horizon, dis=state['distance'],
                     min_speed=config.min_speed, max_speed=state['max_speed'])
    result, _ = opt.optimize()
    if not result.success or not np.all(np.isfinite(result.x)):
        return None
    t1, a1, t3, a3 = result.x
    tol = config.solver_tolerance
    speeds = (state['speed'], state['speed'] + a1*t1, state['speed'] + a1*t1 + a3*t3)
    if (abs(opt.constraint1(result.x)) > tol or abs(opt.constraint2(result.x)) > tol
            or min(speeds) < config.min_speed - tol or max(speeds) > state['max_speed'] + tol
            or not (0 <= t1 <= horizon and 0 <= t3 <= horizon)
            or not (-4.5 <= a1 <= 0 and 0 <= a3 <= 2.6)):
        return None
    return {'stream': state['stream'], 'target': target, 'start': state['time'],
            'distance': state['distance'], 'speed': state['speed'], 'curve': result.x.tolist()}


def trajectory_state(plan, now):
    """Remaining distance and speed on the two-phase reference curve."""
    t1, a1, t3, a3 = plan['curve']
    elapsed = max(now - plan['start'], 0)
    u = min(elapsed, t1)
    w = min(max(elapsed - t1, 0), t3)
    v1 = plan['speed'] + a1*t1
    travelled = plan['speed']*u + 0.5*a1*u*u + v1*w + 0.5*a3*w*w
    return plan['distance'] - travelled, plan['speed'] + a1*u + a3*w


class CVMCController:
    def __init__(self, traci, config=None, log_file=None):
        self.traci = traci
        self.config = config or CVMCConfig()
        self.plans = {}
        self.locked = []
        self.previous_ids = set()
        self.recent_entries = []
        self.stats = dict(updates=0, solver_failures=0, fallbacks=0, replans=0, commands=0)
        self.log_file = log_file
        self.writer = None
        if log_file is not None:
            self.writer = csv.writer(log_file)
            self.writer.writerow(['time', 'event', 'id', 'main_avs', 'ramp_avs', 'eta', 'target', 'value'])
        self.lane_lengths = {lane: traci.lane.getLength(lane)
                             for lane in ('inflow_highway_0', 'inflow_merge_0')}

    def log(self, now, event, vid='', nm='', nr='', eta='', target='', value=''):
        if self.writer:
            self.writer.writerow([now, event, vid, nm, nr, eta, target, value])

    def read_states(self):
        # IDs identify AVs in this project. Never query HV positions, speeds or membership.
        states, av_lanes = [], {}
        now = self.traci.simulation.getTime()
        for vid in self.traci.vehicle.getIDList():
            if 'av' not in vid:
                continue
            lane = self.traci.vehicle.getLaneID(vid)
            av_lanes[vid] = lane
            if lane not in self.lane_lengths:
                continue
            distance = self.lane_lengths[lane] - self.traci.vehicle.getLanePosition(vid)
            stream = 'M' if lane == 'inflow_highway_0' else 'R'
            if stream == 'M' and distance > self.config.mcz_length:
                continue
            states.append(dict(id=vid, stream=stream, lane=lane, distance=distance,
                               speed=self.traci.vehicle.getSpeed(vid), time=now,
                               max_speed=min(self.traci.vehicle.getMaxSpeed(vid),
                                             self.traci.lane.getMaxSpeed(lane))))
        return states, av_lanes

    def release(self, vid, av_lanes, now):
        if vid in self.plans and vid in av_lanes:
            self.traci.vehicle.setSpeed(vid, -1)
            self.log(now, 'release', vid)
        self.plans.pop(vid, None)

    def valid_plan(self, state, plan):
        if plan['target'] <= state['time']:
            return False
        distance, speed = trajectory_state(plan, state['time'])
        return (abs(distance - state['distance']) <= self.config.distance_tolerance
                and abs(speed - state['speed']) <= self.config.speed_tolerance
                and speed <= state['max_speed'] + self.config.solver_tolerance)

    def step(self, step):
        states, av_lanes = self.read_states()
        now = self.traci.simulation.getTime()
        active = {a['id']: a for a in states}
        # Remember AV entry times briefly after releasing control at the MS boundary.
        for vid in self.previous_ids - active.keys():
            if vid in av_lanes and (av_lanes[vid].startswith('ws_') or av_lanes[vid].startswith(':c1')):
                self.recent_entries.append(dict(stream='M' if vid.startswith('m') else 'R', target=now))
            self.release(vid, av_lanes, now)
        self.recent_entries = [p for p in self.recent_entries if now - p['target'] < self.config.buffer]
        invalid = [vid for vid, plan in self.plans.items()
                   if vid in active and not self.valid_plan(active[vid], plan)]
        if invalid:
            first = min(self.locked.index(vid) for vid in invalid)
            for vid in self.locked[first:]:
                self.release(vid, av_lanes, now)
            self.locked = self.locked[:first]
            self.stats['replans'] += 1
            self.log(now, 'invalidate', value=','.join(invalid))
        self.locked = [vid for vid in self.locked if vid in active]
        if step % self.config.interval == 0 or active.keys() - self.previous_ids or invalid:
            self.update(states, av_lanes, now)
        self.previous_ids = set(active)
        for vid, plan in self.plans.items():
            t1, a1, t3, a3 = plan['curve']
            acc = a1 if now - plan['start'] < t1 else a3
            # Preserve the existing SUMO 1.19 one-step interpolation compensation.
            self.traci.vehicle.setAcceleration(vid, acc*2, 0.1)
            self.stats['commands'] += 1
            self.log(now, 'command', vid, target=plan['target'], value=acc*2)

    def update(self, states, av_lanes, now):
        self.stats['updates'] += 1
        nm = sum(a['stream'] == 'M' for a in states)
        self.log(now, 'update', nm=nm, nr=len(states)-nm)
        order = fifo_order(states, self.locked, self.config.v_floor)
        if order is None:
            for vid in list(self.plans):
                self.release(vid, av_lanes, now)
            self.locked = []
            self.stats['replans'] += 1
            self.log(now, 'unlock_order')
            order = fifo_order(states, v_floor=self.config.v_floor)
        accepted = []
        previous = max(self.recent_entries, key=lambda p: p['target'], default=None)
        cross_stream = len({a['stream'] for a in states} | {p['stream'] for p in self.recent_entries}) > 1
        for index, state in enumerate(order):
            vid = state['id']
            eta = now + state['distance'] / max(state['speed'], self.config.v_floor)
            target = target_time(state, previous, self.config) if cross_stream else eta
            self.log(now, 'target', vid, eta=eta, target=target)
            if target > eta + 0.01:
                plan = solve_trajectory(state, target, self.config)
                if plan is None:
                    self.stats['solver_failures'] += 1
                    self.stats['fallbacks'] += 1
                    self.log(now, 'solve_failed', vid, target=target)
                    old = self.plans.get(vid)
                    if old and self.valid_plan(state, old) and old['target'] >= target:
                        accepted.append(vid)
                    else:
                        self.release(vid, av_lanes, now)
                    # No new suffix may rely on an infeasible target.
                    for later in order[index+1:]:
                        self.release(later['id'], av_lanes, now)
                    break
                self.plans[vid] = plan
                self.log(now, 'solve_success', vid, target=target)
            else:
                self.release(vid, av_lanes, now)
            accepted.append(vid)
            previous = dict(stream=state['stream'], target=target)
        self.locked = accepted
