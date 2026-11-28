# tree = ET.parse("data/Traj/trj_720_0.1_3.xml")
# tree = ET.parse("data/Traj/trajectory_fifo_test.xml")
import os
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import math
import re
from tqdm import tqdm   # pip install tqdm
from pathlib import Path


def _parse_xy(pos):
    if pos is None or pos == "NA":
        return None
    try:
        x_str, y_str = pos.split(",")
        return float(x_str), float(y_str)
    except (ValueError, AttributeError):
        return None


def _in_ms_lane(lane_id):
    return lane_id is not None and lane_id.startswith("ws_")


def calc_ttc_conflict_metrics(ssm_file, total_vehicles=None, ttc_threshold=3.0,
                              measure="TTC", min_time=None, max_time=None):
    """Calculate TTC or DRAC conflict metrics within the merging section.
        Return: avg_value, conflict_count, conflict_rate, risky_mr_ramp_ids
    """

    ssm_file = Path(ssm_file)
    if not ssm_file.exists():
        return 0.0, 0, 0.0, set()

    root = ET.parse(ssm_file).getroot()
    measure = measure.upper()

    span_name = "TTCSpan" if measure == "TTC" else "DRACSpan"
    min_name = "minTTC" if measure == "TTC" else "maxDRAC"

    pair_values = {}

    for conflict in root.findall(".//conflict"):
        ego = conflict.get("ego")
        foe = conflict.get("foe")
        pair = tuple(sorted((ego, foe)))

        time_span = conflict.find("timeSpan")
        measure_span = conflict.find(span_name)
        ego_lanes = conflict.find("egoLane")
        foe_lanes = conflict.find("foeLane")

        if (time_span is None or measure_span is None
                or ego_lanes is None or foe_lanes is None):
            continue

        times = time_span.get("values", "").split()
        values = measure_span.get("values", "").split()
        ego_lane = ego_lanes.get("values", "").split()
        foe_lane = foe_lanes.get("values", "").split()

        for t, value, e_lane, f_lane in zip(times, values, ego_lane, foe_lane):
            if value == "NA":
                continue

            try:
                t = float(t)
                value = float(value)
            except ValueError:
                continue

            if min_time is not None and t < min_time:
                continue
            if max_time is not None and t > max_time:
                continue

            if not (_in_ms_lane(e_lane) or _in_ms_lane(f_lane)):
                continue

            if measure == "TTC":
                if pair not in pair_values or value < pair_values[pair]:
                    pair_values[pair] = value
            else:
                if pair not in pair_values or value > pair_values[pair]:
                    pair_values[pair] = value

    mr_ramp_values = {}
    for pair, value in pair_values.items():
        first, second = pair
        is_mr_pair = (
            (first.startswith("m") and second.startswith("r"))
            or (first.startswith("r") and second.startswith("m"))
        )
        if not is_mr_pair:
            continue

        ramp_id = first if first.startswith("r") else second
        if (ramp_id not in mr_ramp_values
                or (measure == "TTC" and value < mr_ramp_values[ramp_id])
                or (measure != "TTC" and value > mr_ramp_values[ramp_id])):
            mr_ramp_values[ramp_id] = value

    if measure == "TTC":
        risky_values = [v for v in pair_values.values() if v < ttc_threshold]
        risky_mr_ramps = {rid for rid, v in mr_ramp_values.items() if v < ttc_threshold}
    else:
        risky_values = [v for v in pair_values.values() if v > ttc_threshold]
        risky_mr_ramps = {rid for rid, v in mr_ramp_values.items() if v > ttc_threshold}

    conflict_count = len(risky_values)
    avg_value = float(np.mean(risky_values)) if risky_values else 0.0

    conflict_rate = (
        conflict_count / total_vehicles
        if total_vehicles is not None and total_vehicles > 0
        else 0.0
    )

    return avg_value, conflict_count, conflict_rate, risky_mr_ramps


_MAINLINE_LANES = {"inflow_highway_0", "inflow_highway_1"}
_REGIONS = ("pfz", "mcz")
_PFZ_END = 1200.0


def _classify_region(lane_id, lane_pos):
    if lane_id == "inflow_merge_0":
        return "mcz"
    if lane_id in _MAINLINE_LANES and lane_pos is not None:
        if 0 <= lane_pos < _PFZ_END:
            return "pfz"
        if lane_pos >= _PFZ_END:
            return "mcz"
    return None


def _classify_pair_region(ego_lane, ego_pos, foe_lane, foe_pos):
    ego_region = _classify_region(ego_lane, ego_pos)
    foe_region = _classify_region(foe_lane, foe_pos)
    if ego_region == foe_region:
        return ego_region
    return None


def _in_time_window(time, min_time, max_time):
    return (min_time is None or min_time <= time) and (max_time is None or time <= max_time)


def _iter_safety_xml(path, root_tag):
    """Yield each direct child, releasing it before reading the next one."""
    try:
        with Path(path).open("rb") as source:
            root = None
            depth = 0
            for event, element in ET.iterparse(source, events=("start", "end", "comment")):
                if event == "comment":
                    if root is None and root_tag == "SSMLog":
                        yield element
                elif event == "start":
                    if root is None:
                        root = element
                        if root.tag != root_tag:
                            raise ValueError(f"{path}: expected <{root_tag}>, found <{root.tag}>")
                    depth += 1
                else:
                    depth -= 1
                    if depth == 1:
                        yield element
                        root.remove(element)
                        element.clear()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing safety input <{root_tag}>: {path}") from exc
    except ET.ParseError as exc:
        raise ValueError(f"Invalid safety XML {path}: {exc}") from exc


def _parse_pfz_lane_changes(lc_path, min_time, max_time):
    lane_changes = set()
    lipr_lane_changes = set()
    for change in _iter_safety_xml(lc_path, "lanechanges"):
        if change.tag != "change":
            continue
        time = float(change.attrib["time"])
        if not _in_time_window(time, min_time, max_time):
            continue
        from_lane = change.attrib["from"]
        to_lane = change.attrib["to"]
        pos = float(change.attrib["pos"])
        if from_lane == to_lane or from_lane not in _MAINLINE_LANES or to_lane not in _MAINLINE_LANES:
            continue
        if _classify_region(to_lane, pos) != "pfz":
            continue
        event_id = (change.attrib["id"], time, from_lane, to_lane)
        vehicle_type = change.attrib["type"]
        reason = change.get("reason", "")
        event = (event_id, vehicle_type, reason)
        lane_changes.add(event)
        if vehicle_type == "av" and from_lane == "inflow_highway_1" and "traci" in reason:
            lipr_lane_changes.add(event)
    return lane_changes, lipr_lane_changes


def _ssm_sample_columns(conflict, ssm_path, pair):
    columns = []
    for tag in ("timeSpan", "TTCSpan", "DRACSpan", "egoLane", "egoLanePosition",
                "foeLane", "foeLanePosition"):
        span = conflict.find(tag)
        if span is None or "values" not in span.attrib:
            raise ValueError(f"{ssm_path}: conflict {pair} is missing {tag} values")
        columns.append(span.attrib["values"].split())
    if len({len(column) for column in columns}) != 1:
        raise ValueError(f"{ssm_path}: conflict {pair} has misaligned SSM samples")
    return zip(*columns)


def _ssm_number(value):
    if value == "NA":
        return None
    return float(value)


def _mark_risky_changes(events_by_target, risky_events, vehicle, target_lane, time):
    for event_id in events_by_target.get((vehicle, target_lane), ()):
        change_time = event_id[1]
        if change_time - 1.0 <= time <= change_time + 2.0:
            risky_events.add(event_id)


def _collect_region_pair_metrics_and_lc_risks(
    ssm_path, lane_changes, lipr_lane_changes, min_time, max_time,
):
    minimum_ttc = {region: {} for region in _REGIONS}
    maximum_drac = {region: {} for region in _REGIONS}
    risky_events = set()
    events_by_target = {}
    for event_id, _, _ in lane_changes:
        vehicle, _, _, target_lane = event_id
        events_by_target.setdefault((vehicle, target_lane), set()).add(event_id)

    has_conflict = False
    has_both_measures = False
    for conflict in _iter_safety_xml(ssm_path, "SSMLog"):
        if conflict.tag is ET.Comment:
            # SUMO embeds its run configuration in the leading XML comment.
            match = re.search(r'<device\.ssm\.measures\s+value="([^"]+)"', conflict.text or "")
            if match is not None:
                has_both_measures = {"TTC", "DRAC"}.issubset(match[1].split())
            continue
        if conflict.tag != "conflict":
            continue
        has_conflict = True
        ego = conflict.attrib["ego"]
        foe = conflict.attrib["foe"]
        pair = tuple(sorted((ego, foe)))
        samples = _ssm_sample_columns(conflict, ssm_path, pair)
        for sample_time, ttc, drac, ego_lane, ego_pos, foe_lane, foe_pos in samples:
            time = float(sample_time)
            if not _in_time_window(time, min_time, max_time):
                continue
            region = _classify_pair_region(
                ego_lane, _ssm_number(ego_pos), foe_lane, _ssm_number(foe_pos),
            )
            if region is None:
                continue
            ttc_value = _ssm_number(ttc)
            drac_value = _ssm_number(drac)
            if ttc_value is not None:
                ttc_by_pair = minimum_ttc[region]
                ttc_by_pair[pair] = min(ttc_value, ttc_by_pair.get(pair, ttc_value))
                if region == "pfz" and ttc_value < 3.0:
                    _mark_risky_changes(events_by_target, risky_events, ego, foe_lane, time)
                    _mark_risky_changes(events_by_target, risky_events, foe, ego_lane, time)
            if drac_value is not None:
                drac_by_pair = maximum_drac[region]
                drac_by_pair[pair] = max(drac_value, drac_by_pair.get(pair, drac_value))

    if not has_conflict and not has_both_measures:
        raise ValueError(f"{ssm_path}: no TTCSpan or DRACSpan and no TTC DRAC configuration")
    lipr_ids = {event_id for event_id, _, _ in lipr_lane_changes}
    return minimum_ttc, maximum_drac, risky_events, risky_events & lipr_ids


def _count_region_vehicles(fcd_path, min_time, max_time):
    vehicle_ids = {region: set() for region in _REGIONS}
    for timestep in _iter_safety_xml(fcd_path, "fcd-export"):
        if timestep.tag != "timestep":
            continue
        time = float(timestep.attrib["time"])
        if not _in_time_window(time, min_time, max_time):
            continue
        for vehicle in timestep.iterfind("vehicle"):
            region = _classify_region(vehicle.attrib["lane"], float(vehicle.attrib["pos"]))
            if region is not None:
                vehicle_ids[region].add(vehicle.attrib["id"])
    return {region: len(ids) for region, ids in vehicle_ids.items()}


def calculate_safety_counts(ssm_path, fcd_path, lc_path, min_time=180, max_time=None):
    """Count PFZ/MCZ exposures from completed simulation output."""
    lane_changes, lipr_changes = _parse_pfz_lane_changes(lc_path, min_time, max_time)
    ttc, drac, risky_changes, risky_lipr_changes = _collect_region_pair_metrics_and_lc_risks(
        ssm_path, lane_changes, lipr_changes, min_time, max_time,
    )
    vehicle_counts = _count_region_vehicles(fcd_path, min_time, max_time)
    return {
        "pfz_vehicle_count": vehicle_counts["pfz"],
        "pfz_ttc_risky_pair_count": sum(value < 3.0 for value in ttc["pfz"].values()),
        "pfz_drac_risky_pair_count": sum(value > 3.0 for value in drac["pfz"].values()),
        "pfz_lane_change_count": len({event_id for event_id, _, _ in lane_changes}),
        "pfz_risky_lane_change_count": len(risky_changes),
        "pfz_lipr_lane_change_count": len({event_id for event_id, _, _ in lipr_changes}),
        "pfz_lipr_risky_lane_change_count": len(risky_lipr_changes),
        "mcz_vehicle_count": vehicle_counts["mcz"],
        "mcz_ttc_risky_pair_count": sum(value < 3.0 for value in ttc["mcz"].values()),
        "mcz_drac_risky_pair_count": sum(value > 3.0 for value in drac["mcz"].values()),
    }


if __name__ == "__main__":
    ssm_file = (
        "/home/zzha/PycharmProjects/RampMerging_2026/"
        "data/multi_lane/algo/ssm_rm_1400_0_0_20260807_194334.xml"
    )
    avg_min_ttc, conflict_count, conflict_rate, mr_ramp_ids = calc_ttc_conflict_metrics(
        ssm_file, total_vehicles=1235, ttc_threshold=3)
    print(f"Avg min TTC: {avg_min_ttc:.2f}, Conflict Count: {conflict_count}, "
          f"Conflict Rate: {conflict_rate:.4f}, M-R Ramp Count: {len(mr_ramp_ids)}")
