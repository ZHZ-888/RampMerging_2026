"""Generate reproducible ramp inputs; manifests are offline ground truth only."""
import csv
import json
import math
from collections import Counter
from numbers import Integral
from pathlib import Path

import numpy as np


class GenerationInfeasibleError(ValueError):
    pass


class DepartureSearchExhausted(GenerationInfeasibleError):
    pass


def round_half_up(value):
    return math.floor(value + 0.5)


class RampPlatoonGenerator:
    mean_sizes = {0.1: 7.75, 0.2: 5.33, 0.3: 4.76}
    n_max = 12

    def __init__(self, st, av_p, fr, ramp_spr, seed=None, sparse_gap_boundary=(5, 10),
                 mean_size=None, max_attempts=1000, schedule_attempts=7,
                 free_follower_probability=0.2, oversized_shared_count=None):
        """Initialise the generator and its output containers.

        generate() populates these attributes. Departure times and gaps are
        in seconds.

        Set oversized_shared_count to the reference run's Oversized count
        to favour longer sizes only beyond the shared prefix.

        schedule: [
            {'vehicle_id': 'ravh10', 'dpt_ts': 1.0, 'vehicle_type': 'av/hv_cons/hv_mean/hv_agg',
             'platoon_id': 'rp0', 'is_leader': True, 'position_in_platoon': 0}, ...]
        manifest: [
            {'platoon_id': 'rp0', 'leader_id': 'ravh10', 'dpt_ts': 1.0,
             'size': 2, 'platoon_category': 'Sparse', 'platoon_type': 'AH',
             'follower_gap': [6]},]
        summary: {'N': 2, 'N_AV': 1, 'N_HV': 1, 'K': 1, 'K_std': 0,
                  'K_sparse': 1, 'K_oversized': 0, 'actual_input_spr': 0.0, ...}
        """
        if st <= 0 or fr < 0 or not 0 <= av_p <= 1 or not 0 <= ramp_spr <= 1:
            raise ValueError('Require st > 0, fr >= 0, and av_p/ramp_spr in [0, 1].')
        if (not isinstance(sparse_gap_boundary, (tuple, list)) or len(sparse_gap_boundary) != 2
                or not all(isinstance(value, Integral) for value in sparse_gap_boundary)
                or not 2 <= sparse_gap_boundary[0] <= sparse_gap_boundary[1]):
            raise ValueError('sparse_gap_boundary must be two ordered integer seconds above 1.')
        if not 0 <= free_follower_probability <= 1:
            raise ValueError('free_follower_probability must lie in [0, 1].')
        if (oversized_shared_count is not None and
                (not isinstance(oversized_shared_count, Integral) or oversized_shared_count < 0)):
            raise ValueError('oversized_shared_count must be a non-negative integer or None.')
        if max_attempts < 1 or schedule_attempts < 1:
            raise ValueError('Attempt limits must be positive.')
        self.st, self.av_p, self.fr = st, av_p, fr
        self.ramp_spr, self.seed = ramp_spr, seed
        self.sparse_gap_boundary = tuple(map(int, sparse_gap_boundary))
        self.free_follower_probability = free_follower_probability
        self.oversized_shared_count = None if oversized_shared_count is None else int(oversized_shared_count)
        self.mean_size = self.mean_sizes.get(av_p) if mean_size is None else mean_size
        self.max_attempts, self.schedule_attempts = max_attempts, schedule_attempts
        self.n = int(fr*st / 3600)
        self.n_av = int(self.n*av_p)
        self.k = self.k_std = 0
        if self.n:
            if self.n_av == 0:
                raise GenerationInfeasibleError(f'No AV leader available: N={self.n}, N_AV=0.')
            if self.mean_size is None or not math.isfinite(self.mean_size) or self.mean_size <= 0:
                raise ValueError('Uncalibrated av_p: provide a positive mean_size explicitly.')
            self.k = min(self.n_av, max(1, round_half_up(self.n / self.mean_size)))
            self.k_std = round_half_up(ramp_spr*self.k)
            if self.n < self.k_std + 2*(self.k-self.k_std) or (self.k_std == self.k and self.n > 12*self.k):
                raise GenerationInfeasibleError(f'Infeasible composition: {self._counts()}')
        self.schedule, self.manifest, self.summary = [], [], {}

    def _counts(self):
        return dict(N=self.n, N_AV=self.n_av, N_HV=self.n-self.n_av, K=self.k, K_std=self.k_std)

    def _oversized_size_sequence(self):
        """Draw a common size sequence independently of the target SPR."""
        lower = self.n_max + 1
        upper = self.n - self.k + 1
        if upper < lower:
            return np.array([], dtype=int)
        rng = np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(1,)))
        sizes = np.arange(lower, upper + 1)
        rate = max(self.mean_size - 1, np.finfo(float).tiny)
        log_weights = np.concatenate(([0.0], np.cumsum(np.log(rate) - np.log(sizes[1:] - 1))))
        weights = np.exp(log_weights - log_weights.max())
        sequence = rng.choice(sizes, size=self.k, p=weights / weights.sum())
        # If all smaller platoons were full, the first Oversized platoon
        # would still have to accommodate this many vehicles.
        sequence[0] = max(sequence[0], self.n - self.n_max * (self.k - 1))
        return sequence

    def _generate_composition(self, rng):
        nonstandard = self.k - self.k_std
        threshold = self.n_max
        # Size is one leader plus a Poisson number of followers.
        def poisson_tail(mean_size):
            rate = max(0.0, mean_size - 1)
            return max(0.0, 1 - math.fsum(
                math.exp(-rate) * rate**i / math.factorial(i) for i in range(threshold)
            ))

        probability = 0.5 if self.av_p == 0.1 else min(
            1.0, 0.5 * poisson_tail(self.mean_size) / poisson_tail(self.mean_sizes[0.1])
        )
        shared_sizes = self._oversized_size_sequence()
        prefix_totals = np.concatenate(([0], np.cumsum(shared_sizes[:nonstandard])))
        counts = np.arange(len(prefix_totals))
        minimum_total = prefix_totals + self.k_std + 2 * (nonstandard - counts)
        maximum_total = prefix_totals + threshold * (self.k - counts)
        feasible = np.flatnonzero((minimum_total <= self.n) & (self.n <= maximum_total))
        if (not feasible.size or (probability == 0 and feasible[0] > 0)
                or (probability == 1 and feasible[-1] < nonstandard)):
            raise GenerationInfeasibleError(f'No feasible type draw: {self._counts()}')
        minimum, maximum = feasible[0], feasible[-1]
        for attempt in range(self.max_attempts):
            oversized = rng.binomial(nonstandard, probability)
            if minimum <= oversized <= maximum:
                break
        else:
            raise GenerationInfeasibleError(
                f'Type sampling exhausted: {self._counts()}, feasible Oversized count={minimum}..{maximum}.')
        categories = rng.permutation(
            ['Standard'] * self.k_std + ['Sparse'] * (nonstandard - oversized) + ['Oversized'] * oversized
        ).tolist()
        fixed_sizes = shared_sizes[:oversized].tolist()
        other_categories = [kind for kind in categories if kind != 'Oversized']
        if self.oversized_shared_count is not None and self.oversized_shared_count < oversized:
            spare = self.n - sum(fixed_sizes) - self.k_std - 2 * (nonstandard - oversized)
            extra_indices = rng.permutation(np.arange(self.oversized_shared_count, oversized))
            for index in extra_indices:
                low = max(17, fixed_sizes[index] + 1)
                high = min(20, fixed_sizes[index] + spare)
                if low <= high and rng.random() < 0.75:
                    size = int(rng.integers(low, high + 1))
                    spare -= size - fixed_sizes[index]
                    fixed_sizes[index] = size
        other_sizes = iter(self._assign_sizes(other_categories, self.n - sum(fixed_sizes), rng)
                           if other_categories else [])
        oversized_sizes = iter(fixed_sizes)
        sizes = [next(oversized_sizes) if kind == 'Oversized' else next(other_sizes)
                 for kind in categories]
        members = [['A'] + ['H']*(int(size)-1) for size in sizes]
        positions = [(i, j) for i, seq in enumerate(members) for j in range(1, len(seq))]
        for index in rng.choice(len(positions), self.n_av-self.k, replace=False):
            i, j = positions[index]
            members[i][j] = 'A'
        platoons = []
        for i, (kind, seq) in enumerate(zip(categories, members)):
            gaps = [1]*(len(seq)-1)
            if kind == 'Sparse':
                required = rng.integers(len(gaps))
                free = rng.random(len(gaps)) < self.free_follower_probability
                free[required] = True
                lower, upper = self.sparse_gap_boundary
                for index in np.flatnonzero(free):
                    gaps[index] = int(rng.integers(lower, upper + 1))
            offsets = [0] + np.cumsum([gap*10 for gap in gaps]).tolist()
            platoons.append(dict(platoon_id=f'rp{i}', size=len(seq), platoon_category=kind,
                                 platoon_type=''.join(seq), follower_gap=gaps, offsets=offsets))
        return platoons, attempt

    def _assign_sizes(self, categories, total, rng):
        counts = Counter(categories)
        bounds = {'Standard': (1, self.n_max), 'Sparse': (2, self.n_max)}
        remaining_min = sum(bounds[kind][0] for kind in categories)
        spare = total - remaining_min
        supports = {kind: np.arange(low, min(high, low + spare) + 1)
                    for kind, (low, high) in bounds.items() if counts[kind]}
        remaining_max = sum(counts[kind] * int(values[-1]) for kind, values in supports.items())
        log_factorials = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, self.n_max)))))

        # Tune the common Poisson rate to centre the truncated sizes on the
        # required total before sequentially conditioning on feasible bounds.
        lower_rate, upper_rate = -30.0, math.log(total) + 10
        for _ in range(24):
            log_rate = (lower_rate + upper_rate) / 2
            expected = 0.0
            for kind, values in supports.items():
                log_weights = (values - 1) * log_rate - log_factorials[values - 1]
                weights = np.exp(log_weights - log_weights.max())
                expected += counts[kind] * np.average(values, weights=weights)
            if expected < total:
                lower_rate = log_rate
            else:
                upper_rate = log_rate

        remaining = total
        sizes = []
        for kind in categories:
            support = supports[kind]
            remaining_min -= int(support[0])
            remaining_max -= int(support[-1])
            low = max(int(support[0]), remaining - remaining_max)
            high = min(int(support[-1]), remaining - remaining_min)
            values = np.arange(low, high + 1)
            log_weights = (values - 1) * log_rate - log_factorials[values - 1]
            weights = np.exp(log_weights - log_weights.max())
            size = int(rng.choice(values, p=weights / weights.sum()))
            sizes.append(size)
            remaining -= size
        return sizes

    def _assign_departures(self, platoons, rng):
        # Descending gap search with random head placement, as in the original approach.
        # Free intervals implement rejection-free selection among feasible integer-second heads.
        end = math.ceil(self.st*10)-1
        spans = [p['offsets'][-1] for p in platoons]
        if sum(spans) + 10*(self.k-1) > end:
            raise GenerationInfeasibleError(f'Departure window too short: {self._counts()}, spans={sum(spans)/10}s, st={self.st}.')
        upper = (end-sum(spans)) // (10*(self.k-1)) if self.k > 1 else 0
        for gap in range(upper, 0 if self.k > 1 else -1, -1):
            for _ in range(self.schedule_attempts):
                free, starts = [(0, end)], {}
                for i in rng.permutation(self.k):
                    slots = [(lo, hi, math.ceil(lo/10), math.floor((hi-spans[i])/10)) for lo, hi in free]
                    slots = [s for s in slots if s[2] <= s[3]]
                    counts = [s[3]-s[2]+1 for s in slots]
                    if not counts:
                        break
                    pick = int(rng.integers(sum(counts)))
                    for (lo, hi, first, last), count in zip(slots, counts):
                        if pick < count:
                            start = (first+pick)*10
                            break
                        pick -= count
                    starts[int(i)] = start
                    free.remove((lo, hi))
                    free.extend([(lo, start-gap*10), (start+spans[i]+gap*10, hi)])
                    free = [(a, b) for a, b in free if a <= b]
                if len(starts) == self.k:
                    return starts, gap
        raise DepartureSearchExhausted(f'Random departure search exhausted: {self._counts()}, st={self.st}.')

    def generate(self):
        # Reset the local RNG so repeated calls reproduce exactly, without changing global state.
        rng = np.random.default_rng(self.seed)
        self.schedule, self.manifest = [], []
        resamples, gap = 0, None
        if self.n:
            platoons, resamples = self._generate_composition(rng)
            # Longer Sparse gaps can make random packing harder. Retry the
            # existing scheduler with the same composition and gap samples.
            for retry in range(20):
                try:
                    starts, gap = self._assign_departures(platoons, rng)
                    break
                except DepartureSearchExhausted:
                    if retry == 19:
                        raise
            for i, platoon in enumerate(platoons):
                leader_id = f'ravh{starts[i]}'
                for j, (member, offset) in enumerate(zip(platoon['platoon_type'], platoon['offsets'])):
                    tick = starts[i] + offset
                    body = 'avh' if j == 0 else ('av' if member == 'A' else 'hv')
                    vehicle_type = 'av' if member == 'A' else str(rng.choice(
                        ['hv_cons', 'hv_mean', 'hv_agg'], p=[0.51, 0.32, 0.17]))
                    self.schedule.append(dict(vehicle_id=f'r{body}{tick}', dpt_ts=tick/10,
                                              vehicle_type=vehicle_type, platoon_id=platoon['platoon_id'],
                                              is_leader=j == 0, position_in_platoon=j))
                self.manifest.append({key: value for key, value in platoon.items() if key != 'offsets'}
                                     | dict(leader_id=leader_id, dpt_ts=starts[i]/10))
        self.schedule.sort(key=lambda row: row['dpt_ts'])
        self.manifest.sort(key=lambda row: row['dpt_ts'])
        kinds = Counter(p['platoon_category'] for p in self.manifest)
        self.summary = dict(target_tp=self.fr, actual_tp=len(self.schedule)*3600/self.st,
                            target_avp=self.av_p, actual_avp=self.n_av/self.n if self.n else None,
                            target_spr=self.ramp_spr, actual_input_spr=self.k_std/self.k if self.k else None,
                            **self._counts(), K_sparse=kinds['Sparse'], K_oversized=kinds['Oversized'],
                            K_AV_followers=self.n_av-self.k, mean_size_reference=self.mean_size,
                            mean_size_realised=self.n/self.k if self.k else None,
                            size_histogram=dict(sorted(Counter(p['size'] for p in self.manifest).items())),
                            sparse_gap_boundary=list(self.sparse_gap_boundary),
                            free_follower_probability=self.free_follower_probability,
                            inter_platoon_gap=gap,
                            seed=self.seed, num_resamples=resamples, st=self.st, n_max=self.n_max,
                            max_attempts=self.max_attempts, schedule_attempts=self.schedule_attempts)
        if self.oversized_shared_count is not None:
            self.summary['oversized_shared_count'] = self.oversized_shared_count
        self.validate()
        return dict(schedule=self.schedule, manifest=self.manifest, summary=self.summary)

    def validate(self):
        counts = Counter(row['vehicle_type'] == 'av' for row in self.schedule)
        assert len(self.schedule) == self.n and counts[True] == self.n_av, 'Vehicle count mismatch.'
        assert sum(p['platoon_category'] == 'Standard' for p in self.manifest) == self.k_std, 'SPR mismatch.'
        times = [r['dpt_ts'] for r in self.schedule]
        assert all(0 <= t < self.st for t in times) and all(a < b for a, b in zip(times, times[1:])), 'Invalid departure order.'
        previous_tail = None
        for p in self.manifest:
            rows = [r for r in self.schedule if r['platoon_id'] == p['platoon_id']]
            assert len(rows) == p['size'] and rows[0]['vehicle_id'] == p['leader_id'], 'Platoon boundary mismatch.'
            assert rows[0]['is_leader'] and rows[0]['vehicle_type'] == 'av', 'Missing AV leader.'
            gaps = p['follower_gap']
            lower, upper = self.sparse_gap_boundary
            assert len(gaps) == p['size'] - 1, 'Gap count mismatch.'
            assert all(isinstance(gap, int) and (gap == 1 or lower <= gap <= upper)
                       for gap in gaps), 'Invalid follower gap.'
            has_free_follower = any(gap > 1 for gap in gaps)
            expected = 'Oversized' if p['size'] > self.n_max else ('Sparse' if has_free_follower else 'Standard')
            assert p['platoon_category'] == expected, 'Platoon classification mismatch.'
            assert has_free_follower == (expected == 'Sparse'), 'Invalid free follower count.'
            for a, b, gap in zip(rows, rows[1:], gaps):
                assert abs(b['dpt_ts']-a['dpt_ts']-gap) < 1e-8, 'Gap mismatch.'
            if previous_tail is not None:
                assert rows[0]['dpt_ts']-previous_tail >= self.summary['inter_platoon_gap']-1e-8, 'Inter-platoon overlap.'
            previous_tail = rows[-1]['dpt_ts']

    def export(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / 'vehicle_schedule.csv').open('w', newline='') as file:
            fields = ['vehicle_id', 'dpt_ts', 'vehicle_type', 'platoon_id', 'is_leader', 'position_in_platoon']
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.schedule)
        (output_dir / 'platoon_manifest.json').write_text(json.dumps(self.manifest, indent=2))
        (output_dir / 'summary.json').write_text(json.dumps(self.summary, indent=2))


def print_ramp_platoon_manifest(manifest, category):
    platoons = [row for row in manifest if row["platoon_category"] == category]
    print(f"\n{category} ramp platoons ({len(platoons)})")
    if not platoons:
        print("  None generated.")
        return

    type_width = max(len("Platoon type"), max(len(row["platoon_type"]) for row in platoons))
    header = (
        f"{'Platoon ID':<12} {'Leader ID':<12} {'Departure (s)':>13} "
        f"{'Size':>5}  {'Platoon type':<{type_width}}  Follower gaps (s)")
    print(header)
    print("-" * len(header))
    for row in platoons:
        gaps = ", ".join(f"{gap:g}" for gap in row["follower_gap"])
        print(
            f"{row['platoon_id']:<12} {row['leader_id']:<12} {row['dpt_ts']:>13.1f} "
            f"{row['size']:>5}  {row['platoon_type']:<{type_width}}  [{gaps}]")


if __name__ == '__main__':
    # Local generation test: edit these inputs, then run this file.
    generator = RampPlatoonGenerator(st=1500, av_p=0.1, fr=1200, ramp_spr=0.7,
                                     seed=3)
    generator.generate()
    generator.export('data/ramp_inputs/local_test')
    print(json.dumps(generator.summary, indent=2))
