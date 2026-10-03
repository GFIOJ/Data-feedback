"""Export fixed geometry, trajectories and collection arrivals for paper 2."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .paper2_dataset import interpolate_timed_path, write_arrival_files


def _first_read_on_segment(tag, segment, start_time, speed, radius, half_angle):
    start, end, axis = segment[:3], segment[3:6], segment[6:9]
    line = end - start
    length = np.linalg.norm(line)
    axis = axis / np.linalg.norm(axis)
    vector = tag - start
    forward = float(vector @ axis)
    if forward < 0:
        return None
    if length < 1e-12:
        distance = np.linalg.norm(vector)
        return start_time if distance <= radius and (distance < 1e-12 or forward / distance >= np.cos(half_angle)) else None
    direction = line / length
    along = float(vector @ direction)
    cross = vector - forward * axis - along * direction
    lateral2 = float(cross @ cross)
    allowance = min(radius ** 2 - forward ** 2 - lateral2,
                    forward ** 2 * np.tan(half_angle) ** 2 - lateral2)
    if allowance < -1e-9:
        return None
    reach = np.sqrt(max(0., allowance))
    distance_along = max(0., along - reach)
    if distance_along > length + 1e-9 or along + reach < -1e-9:
        return None
    return start_time + min(distance_along, length) / speed


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _slot_states(paths, takeoff, returns, slot_seconds, slot_count):
    starts = np.arange(slot_count) * slot_seconds
    ends, mids = starts + slot_seconds, starts + slot_seconds / 2
    positions = np.full((slot_count, len(paths), 3), np.nan)
    online = np.zeros((slot_count, len(paths)), bool)
    full = np.zeros_like(online)
    boundary = np.zeros_like(online)
    for uav, path in enumerate(paths):
        for slot, time_s in enumerate(mids):
            positions[slot, uav], online[slot, uav] = interpolate_timed_path(path, time_s)
        full[:, uav] = (takeoff[uav] <= starts + 1e-12) & (returns[uav] >= ends - 1e-12)
        overlaps = (returns[uav] > starts) & (takeoff[uav] < ends)
        boundary[:, uav] = overlaps & ~full[:, uav]
    return starts, ends, mids, positions, online, full, boundary


def validate_paper2_inputs(output_dir, report_path=None):
    """Validate cross-file invariants without rerunning either planner stage.

    ``report_path`` lets verification runs keep their report outside the fixed
    input directory.  Existing callers retain the original ``validation.json``
    behavior.
    """
    from .paper2_dataset import Paper2Dataset, _read_events

    output = Path(output_dir)
    data = Paper2Dataset(output)
    scenario, states = data.scenario, data.node_states
    events = _read_events(output / 'collection_events.csv')
    with np.load(output / 'slot_arrivals.npz') as arrivals:
        total_bytes = arrivals['new_data_bytes_total'].copy()
        cumulative = arrivals['cumulative_data_bytes_total'].copy()
    checks = {}
    checks['trajectory_endpoints_match_online_times'] = all(
        (len(path) and np.isclose(path[0, 3], states['takeoff_time_s'][u]) and
         np.isclose(path[-1, 3], states['return_time_s'][u])) or
        (not len(path) and np.isnan(states['takeoff_time_s'][u]) and np.isnan(states['return_time_s'][u]))
        for u, path in enumerate(data.paths))
    checks['global_timestamps_not_shifted'] = checks['trajectory_endpoints_match_online_times']
    checks['planner_path_validation_passed'] = bool(scenario['path_validation_passed'])
    checks['one_event_per_target'] = (len(events) == scenario['inspection_target_count'] and
                                      len({e['target_id'] for e in events}) == len(events))
    checks['events_inside_scan_windows'] = all(
        e['scan_start_s'] - 1e-8 <= e['collection_complete_time_s'] <= e['scan_end_s'] + 1e-8
        for e in events)
    checks['events_during_uav_online_period'] = all(
        states['takeoff_time_s'][e['source_uav_id']] <= e['collection_complete_time_s'] <
        states['return_time_s'][e['source_uav_id']] for e in events)
    checks['sendable_only_from_next_slot'] = all(
        e['available_from_slot_zero_based'] == e['enqueue_slot_zero_based'] + 1 and
        e['enqueue_slot_paper_one_based'] == e['enqueue_slot_zero_based'] + 1 and
        e['available_from_slot_paper_one_based'] == e['available_from_slot_zero_based'] + 1
        for e in events)
    checks['bits_equal_eight_times_bytes'] = all(e['data_bits'] == 8 * e['data_bytes'] for e in events)
    expected = np.zeros_like(total_bytes)
    for event in events:
        expected[event['enqueue_slot_zero_based']] += event['data_bytes']
    checks['event_slot_and_cumulative_amounts_match'] = (np.array_equal(expected, total_bytes) and
                                                         np.array_equal(np.cumsum(expected), cumulative))
    inactive = ~states['online_at_slot_midpoint']
    checks['offline_uav_positions_are_nan'] = bool(np.isnan(states['positions_at_slot_midpoint_m'][inactive]).all())
    full_expected = ((states['takeoff_time_s'][None, :] <= states['slot_start_seconds'][:, None] + 1e-12) &
                     (states['return_time_s'][None, :] >= states['slot_end_seconds'][:, None] - 1e-12))
    checks['full_slot_eligibility_is_correct'] = np.array_equal(full_expected, states['full_slot_online'])
    env = data.environment
    checks['exported_geometry_counts_match'] = (
        len(env['obstacle_ids']) == scenario['aggregate_obstacle_count'] and
        len(env['target_ids']) == scenario['inspection_target_count'] and
        np.all(env['obstacle_min_m'] <= env['obstacle_max_m']))
    from .common import segment_hits
    checks['exported_geometry_paths_are_obstacle_free'] = all(
        not segment_hits(a[:3], b[:3], env['obstacle_min_m'], env['obstacle_max_m']).any()
        for path in data.paths for a, b in zip(path[:-1], path[1:]))
    from .uav_mission_planner import first_path_conflict
    safety_radius = float(scenario.get('safety_radius_m', 0.))
    checks['exported_trajectories_are_mutually_conflict_free'] = all(
        first_path_conflict(data.paths[u], data.paths[v], safety_radius) is None
        for u in range(len(data.paths)) for v in range(u + 1, len(data.paths)))
    target_positions = {int(i): p for i, p in zip(env['target_ids'], env['target_positions_m'])}
    scan_segments = env['scan_segments_m']
    radius = float(scenario['rfid_max_read_range_m'])
    cosine = np.cos(np.deg2rad(float(scenario['rfid_max_scan_angle_deg']) / 2))
    def valid_scan_event(event):
        reader = np.array([event['uav_x_m'], event['uav_y_m'], event['uav_z_m']])
        vector = target_positions[event['target_id']] - reader
        distance = np.linalg.norm(vector)
        axis = scan_segments[event['scan_task_id'], 6:9]
        criterion = distance <= radius + 1e-8 and (distance < 1e-12 or vector @ axis / distance >= cosine - 1e-8)
        interpolated, online = interpolate_timed_path(data.paths[event['source_uav_id']],
                                                       event['collection_complete_time_s'])
        return criterion and online and np.allclose(reader, interpolated, atol=1e-7)
    checks['events_match_actual_scan_position_and_criterion'] = all(valid_scan_event(e) for e in events)
    checks = {key: bool(value) for key, value in checks.items()}
    report = {'passed': all(checks.values()), 'checks': checks, 'event_count': len(events),
              'total_data_bytes': int(sum(e['data_bytes'] for e in events)),
              'total_data_bits': int(sum(e['data_bits'] for e in events))}
    destination = Path(report_path) if report_path is not None else output / 'validation.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if not report['passed']:
        raise RuntimeError('Paper 2 input validation failed: ' + ', '.join(k for k, v in checks.items() if not v))
    return report


def export_paper2_inputs(stage1, stage2, output_dir, slot_seconds=1.,
                         data_per_collection_bytes=5_000_000,
                         base_station_position_m=None, scenario_config=None):
    if not stage1.flyByRFID:
        raise ValueError('Paper 2 export requires continuous scan mode')
    if slot_seconds <= 0 or data_per_collection_bytes <= 0:
        raise ValueError('slot_seconds and data_per_collection_bytes must be positive')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tags = np.asarray(stage1.tagPos, float)
    paths = [np.asarray(path, float) for path in stage2.solution.paths]
    takeoff = np.array([path[0, 3] if len(path) else np.nan for path in paths], float)
    returns = np.array([path[-1, 3] if len(path) else np.nan for path in paths], float)
    simulation_end = float(np.nanmax(returns))
    base_station = np.asarray(base_station_position_m if base_station_position_m is not None
                              else [stage2.depotPos[0], stage2.depotPos[1], 30.], float)
    if base_station.shape != (3,):
        raise ValueError('base_station_position_m must contain x, y, z')

    task_windows = {}
    for uav, timeline in enumerate(stage2.solution.timelinePerUAV):
        for node, start, end in timeline.serviceWindows:
            task_windows[int(node) - 1] = (uav, float(start), float(end))
    first = [None] * len(tags)
    radius = float(stage1.rfid.maxReadRange)
    half_angle = np.deg2rad(float(stage1.rfid.maxScanAngleDeg) / 2)
    for task, segment in enumerate(np.asarray(stage1.scanSegments, float)):
        if task not in task_windows:
            raise RuntimeError(f'Scan task {task} has no service window')
        uav, start, end = task_windows[task]
        for target_index, tag in enumerate(tags):
            read_time = _first_read_on_segment(tag, segment, start, stage2.uavSpeed, radius, half_angle)
            if read_time is not None and (first[target_index] is None or read_time < first[target_index][0]):
                length = np.linalg.norm(segment[3:6] - segment[:3])
                fraction = 0. if length == 0 else np.clip((read_time - start) * stage2.uavSpeed / length, 0., 1.)
                uav_position = segment[:3] + fraction * (segment[3:6] - segment[:3])
                first[target_index] = (read_time, uav, task, start, end, uav_position)
    missing = [index for index, event in enumerate(first) if event is None]
    if missing:
        raise RuntimeError(f'{len(missing)} inspection targets have no timestamped read event')

    target_ids = np.asarray(stage1.Tags.id, int)
    events = []
    for data_id, (target_id, tag, record) in enumerate(zip(target_ids, tags, first)):
        time_s, uav, task, scan_start, scan_end, uav_position = record
        slot = int(np.floor(time_s / slot_seconds))
        events.append(dict(data_id=data_id, target_id=int(target_id), source_uav_id=int(uav),
            scan_task_id=int(task), first_read_time_s=float(time_s), collection_complete_time_s=float(time_s),
            enqueue_slot_zero_based=slot, enqueue_slot_paper_one_based=slot + 1,
            available_from_slot_zero_based=slot + 1, available_from_slot_paper_one_based=slot + 2,
            data_bytes=int(data_per_collection_bytes), data_bits=int(data_per_collection_bytes) * 8,
            target_x_m=float(tag[0]), target_y_m=float(tag[1]), target_z_m=float(tag[2]),
            uav_x_m=float(uav_position[0]), uav_y_m=float(uav_position[1]), uav_z_m=float(uav_position[2]),
            scan_start_s=float(scan_start), scan_end_s=float(scan_end)))
    slot_count = max(1, int(np.ceil(simulation_end / slot_seconds)),
                     max(e['enqueue_slot_zero_based'] for e in events) + 1)

    dimensions = stage1.scene.get('physical_container_dimensions')
    box = stage1.Containers[0]
    physical_count = int(stage1.scene.get('physical_container_count', len(stage1.Containers)))
    aggregate_count = len(stage1.Containers)
    scenario = dict(scenario_config or {})
    scenario.update(
        coordinate_unit='m', time_unit='s', data_rate_unit='bit/s', global_time_origin_s=0.,
        num_uavs=int(stage2.numUAVs), simulation_end_time_s=simulation_end,
        slot_seconds=float(slot_seconds), slot_count=slot_count,
        physical_container_count=physical_count, aggregate_obstacle_count=aggregate_count,
        inspection_target_count=len(tags), coverage_denominator='specified inspection targets',
        inspection_target_coverage_rate=float(stage1.coverageRate),
        rfid_max_read_range_m=radius,
        rfid_max_scan_angle_deg=float(stage1.rfid.maxScanAngleDeg),
        data_per_collection_bytes=int(data_per_collection_bytes),
        data_per_collection_bits=int(data_per_collection_bytes) * 8,
        base_station_position_m=base_station.tolist(), base_station_always_online=True,
        depot_position_m=np.asarray(stage2.depotPos, float).tolist(),
        communication_online_interval='[takeoff_time_s, return_time_s)',
        scheduling_eligibility='only UAVs online for the entire slot',
        slot_rule='zero-based slot k is [k*dt,(k+1)*dt); an event at k*dt belongs to k; enqueue at slot end; send from k+1',
        identifier_rule='CSV and NPZ identifiers are zero-based; paper slot number = zero-based slot + 1',
        collection_rule='first successful radius-and-angle read creates one complete data unit',
        delivery_rule='delivery is not modeled; return does not clear or deliver queued data',
        trajectory_interpolation='piecewise linear between timestamped waypoints; waits remain stationary',
        makespan_seconds=float(stage2.makespan), soc_seconds=float(stage2.solution.soc),
        load_imbalance_percent=float(stage2.loadImbalance), path_validation_passed=bool(stage2.validation.passed),
        num_scan_tasks=int(stage1.Khover), num_hover_points=int(stage1.num_hover_points),
        container_dimensions_m=dict(length=float(dimensions.length if dimensions else box.length),
                                    width=float(dimensions.width if dimensions else box.width),
                                    height=float(dimensions.height if dimensions else box.height)))
    (output / 'scenario.json').write_text(
        json.dumps(scenario, ensure_ascii=False, indent=2, default=_json_default), encoding='utf-8')

    np.savez_compressed(output / 'trajectories.npz', **{f'uav_{u}': path for u, path in enumerate(paths)})
    obstacle_min = np.array([[c.x, c.y, c.z] for c in stage1.Containers], float)
    obstacle_max = obstacle_min + np.array([[c.width, c.length, c.height] for c in stage1.Containers], float)
    np.savez_compressed(output / 'environment.npz',
        scene_bounds_m=np.array([stage1.scene.x_range, stage1.scene.y_range, stage1.scene.z_range]),
        obstacle_ids=np.arange(aggregate_count), obstacle_min_m=obstacle_min, obstacle_max_m=obstacle_max,
        target_ids=target_ids, target_positions_m=tags,
        target_owner_stack_ids=np.asarray(stage1.Tags.owner_idx, int),
        target_side_normals=np.asarray(stage1.Tags.normal, float),
        scan_task_ids=np.arange(len(stage1.scanSegments)),
        scan_segments_m=np.asarray(stage1.scanSegments, float),
        depot_position_m=np.asarray(stage2.depotPos, float), base_station_position_m=base_station)

    starts, ends, mids, positions, online, full, boundary = _slot_states(
        paths, takeoff, returns, slot_seconds, slot_count)
    np.savez_compressed(output / 'node_states.npz', slot_start_seconds=starts,
        slot_end_seconds=ends, slot_midpoint_seconds=mids,
        positions_at_slot_midpoint_m=positions, online_at_slot_midpoint=online,
        full_slot_online=full, scheduling_eligible=full, boundary_slot=boundary,
        takeoff_time_s=takeoff, return_time_s=returns, communication_exit_time_s=returns)
    write_arrival_files(output, events, stage2.numUAVs, slot_count,
                        data_per_collection_bytes, takeoff, returns)
    validate_paper2_inputs(output)
    return scenario
