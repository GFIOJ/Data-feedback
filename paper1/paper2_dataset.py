"""Read and update the fixed simulation inputs used by paper 2."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


EVENT_FIELDS = [
    'data_id', 'target_id', 'source_uav_id', 'scan_task_id',
    'first_read_time_s', 'collection_complete_time_s',
    'enqueue_slot_zero_based', 'enqueue_slot_paper_one_based',
    'available_from_slot_zero_based', 'available_from_slot_paper_one_based',
    'data_bytes', 'data_bits', 'target_x_m', 'target_y_m', 'target_z_m',
    'uav_x_m', 'uav_y_m', 'uav_z_m', 'scan_start_s', 'scan_end_s',
]


def interpolate_timed_path(path, time_s):
    """Return (position, online); online interval is [takeoff, return)."""
    path = np.asarray(path, float)
    time_s = float(time_s)
    if not len(path) or time_s < path[0, 3] or time_s >= path[-1, 3]:
        return np.full(3, np.nan), False
    index = max(0, min(np.searchsorted(path[:, 3], time_s, side='right') - 1, len(path) - 2))
    start, end = path[index], path[index + 1]
    fraction = 0. if end[3] <= start[3] else (time_s - start[3]) / (end[3] - start[3])
    return start[:3] + np.clip(fraction, 0., 1.) * (end[:3] - start[:3]), True


class Paper2Dataset:
    """Small reader shared by later channel, routing and scheduling experiments."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.scenario = json.loads((self.folder / 'scenario.json').read_text(encoding='utf-8'))
        with np.load(self.folder / 'trajectories.npz') as data:
            self.paths = [data[f'uav_{u}'].copy() for u in range(self.scenario['num_uavs'])]
        with np.load(self.folder / 'environment.npz') as data:
            self.environment = {key: data[key].copy() for key in data.files}
        with np.load(self.folder / 'node_states.npz') as data:
            self.node_states = {key: data[key].copy() for key in data.files}

    def query_uav(self, uav_id, time_s):
        position, online = interpolate_timed_path(self.paths[int(uav_id)], time_s)
        return {'position_m': position, 'communication_online': online}

    def query_base_station(self, time_s):
        end = self.scenario['simulation_end_time_s']
        online = 0. <= float(time_s) <= end
        position = self.environment['base_station_position_m'].copy() if online else np.full(3, np.nan)
        return {'position_m': position, 'communication_online': online}

    def sample_slot_midpoints(self):
        """Return exported midpoint positions and scheduling eligibility masks."""
        return {key: value.copy() for key, value in self.node_states.items()}


def _read_events(path):
    with path.open(newline='', encoding='utf-8-sig') as file:
        rows = list(csv.DictReader(file))
    integer = {'data_id', 'target_id', 'source_uav_id', 'scan_task_id',
               'enqueue_slot_zero_based', 'enqueue_slot_paper_one_based',
               'available_from_slot_zero_based', 'available_from_slot_paper_one_based',
               'data_bytes', 'data_bits'}
    for row in rows:
        for key in integer:
            row[key] = int(row[key])
        for key in set(EVENT_FIELDS) - integer:
            row[key] = float(row[key])
    return rows


def write_arrival_files(folder, events, num_uavs, slot_count, amount_bytes,
                        takeoff_times, return_times):
    """Write event amounts and all derived arrival tables from fixed event times."""
    folder = Path(folder)
    amounts = np.asarray(amount_bytes)
    if amounts.ndim == 0:
        amounts = np.full(len(events), int(amounts), dtype=np.int64)
    if amounts.shape != (len(events),) or not np.issubdtype(amounts.dtype, np.integer):
        raise ValueError('amount_bytes must be one integer or one integer per data_id')
    amounts = amounts.astype(np.int64, copy=False)
    if np.any(amounts <= 0):
        raise ValueError('every data unit size must be positive')
    data_ids = np.array([int(event['data_id']) for event in events], int)
    if not np.array_equal(np.sort(data_ids), np.arange(len(events))):
        raise ValueError('data_id must be contiguous zero-based for per-event amounts')
    new_items = np.zeros((num_uavs, slot_count), np.int64)
    new_bytes = np.zeros_like(new_items)
    first_time = np.full(len(events), np.nan)
    first_uav = np.full(len(events), -1, int)
    first_slot = np.full(len(events), -1, int)
    target_ids = np.array([int(event['target_id']) for event in events], int)
    if len(np.unique(target_ids)) != len(target_ids):
        raise ValueError('target_id must be unique in collection events')
    target_row = {target_id: row for row, target_id in enumerate(target_ids)}
    for event in events:
        event_amount = int(amounts[int(event['data_id'])])
        event['data_bytes'], event['data_bits'] = event_amount, event_amount * 8
        uav, slot = int(event['source_uav_id']), int(event['enqueue_slot_zero_based'])
        np.add.at(new_items, (uav, slot), 1)
        np.add.at(new_bytes, (uav, slot), event_amount)
        row = target_row[int(event['target_id'])]
        first_time[row] = float(event['first_read_time_s'])
        first_uav[row], first_slot[row] = uav, slot
    new_bits = new_bytes * 8
    sendable_items = np.zeros_like(new_items)
    sendable_bytes = np.zeros_like(new_bytes)
    if slot_count > 1:
        sendable_items[:, 1:] = new_items[:, :-1]
        sendable_bytes[:, 1:] = new_bytes[:, :-1]
    sendable_bits = sendable_bytes * 8

    with (folder / 'collection_events.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(events)
    slot_seconds = json.loads((folder / 'scenario.json').read_text(encoding='utf-8'))['slot_seconds']
    starts = np.arange(slot_count) * slot_seconds
    np.savez_compressed(folder / 'slot_arrivals.npz',
        slot_start_seconds=starts, slot_end_seconds=starts + slot_seconds,
        new_items_per_uav=new_items, new_data_bytes_per_uav=new_bytes,
        new_data_bits_per_uav=new_bits, new_items_total=new_items.sum(0),
        new_data_bytes_total=new_bytes.sum(0), new_data_bits_total=new_bits.sum(0),
        sendable_new_items_per_uav=sendable_items,
        sendable_new_data_bytes_per_uav=sendable_bytes,
        sendable_new_data_bits_per_uav=sendable_bits,
        cumulative_data_bytes_per_uav=np.cumsum(new_bytes, axis=1),
        cumulative_data_bits_per_uav=np.cumsum(new_bits, axis=1),
        cumulative_data_bytes_total=np.cumsum(new_bytes.sum(0)),
        cumulative_data_bits_total=np.cumsum(new_bits.sum(0)),
        first_read_time_s=first_time, first_read_uav=first_uav,
        first_read_slot_zero_based=first_slot, first_read_target_ids=target_ids)

    counts, totals = new_items.sum(1), new_bytes.sum(1)
    with (folder / 'uav_metadata.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['uav_id', 'takeoff_time_s', 'return_time_s', 'communication_start_s',
                         'communication_exit_s', 'collection_count', 'generated_data_bytes',
                         'generated_data_bits'])
        for uav in range(num_uavs):
            writer.writerow([uav, f'{takeoff_times[uav]:.9f}', f'{return_times[uav]:.9f}',
                             f'{takeoff_times[uav]:.9f}', f'{return_times[uav]:.9f}',
                             int(counts[uav]), int(totals[uav]), int(totals[uav] * 8)])
    with (folder / 'system_arrivals.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['slot_zero_based', 'slot_paper_one_based', 'slot_start_s', 'slot_end_s',
                         'new_items', 'new_data_bytes', 'new_data_bits',
                         'cumulative_data_bytes', 'cumulative_data_bits'])
        cumulative = np.cumsum(new_bytes.sum(0))
        for slot in range(slot_count):
            writer.writerow([slot, slot + 1, starts[slot], starts[slot] + slot_seconds,
                             int(new_items[:, slot].sum()), int(new_bytes[:, slot].sum()),
                             int(new_bits[:, slot].sum()), int(cumulative[slot]), int(cumulative[slot] * 8)])


def recompute_data_amount(folder, *, data_per_collection_bytes=None, data_per_collection_bits=None):
    """Change data size without regenerating trajectories or collection times."""
    if (data_per_collection_bytes is None) == (data_per_collection_bits is None):
        raise ValueError('provide exactly one of data_per_collection_bytes/data_per_collection_bits')
    if data_per_collection_bits is not None:
        bits = int(data_per_collection_bits)
        if bits != data_per_collection_bits or bits % 8:
            raise ValueError('data_per_collection_bits must be a positive multiple of 8')
        data_per_collection_bytes = bits // 8
    folder = Path(folder)
    scenario_path = folder / 'scenario.json'
    scenario = json.loads(scenario_path.read_text(encoding='utf-8'))
    events = _read_events(folder / 'collection_events.csv')
    with np.load(folder / 'node_states.npz') as states:
        takeoff, returns = states['takeoff_time_s'], states['return_time_s']
        slot_count = len(states['slot_start_seconds'])
    write_arrival_files(folder, events, scenario['num_uavs'], slot_count,
                        data_per_collection_bytes, takeoff, returns)
    scenario['data_per_collection_bytes'] = int(data_per_collection_bytes)
    scenario['data_per_collection_bits'] = int(data_per_collection_bytes) * 8
    scenario['data_per_collection_mb'] = int(data_per_collection_bytes) / 1_000_000
    scenario_path.write_text(json.dumps(scenario, ensure_ascii=False, indent=2), encoding='utf-8')
    from .paper2_export import validate_paper2_inputs
    validate_paper2_inputs(folder)
    return scenario


def recompute_data_amounts(folder, data_bytes_by_id, *, metadata=None):
    """Set one saved integer-byte size per stable data_id and rebuild all derived totals."""
    folder = Path(folder)
    scenario_path = folder / 'scenario.json'
    scenario = json.loads(scenario_path.read_text(encoding='utf-8'))
    events = _read_events(folder / 'collection_events.csv')
    amounts = np.asarray(data_bytes_by_id)
    if amounts.shape != (len(events),) or not np.issubdtype(amounts.dtype, np.integer):
        raise ValueError('data_bytes_by_id must contain one integer per collection event')
    amounts = amounts.astype(np.int64, copy=False)
    if np.any(amounts <= 0):
        raise ValueError('every data unit size must be positive')
    with np.load(folder / 'node_states.npz') as states:
        takeoff, returns = states['takeoff_time_s'], states['return_time_s']
        slot_count = len(states['slot_start_seconds'])
    write_arrival_files(folder, events, scenario['num_uavs'], slot_count,
                        amounts, takeoff, returns)
    for key in ('data_per_collection_bytes', 'data_per_collection_bits', 'data_per_collection_mb'):
        scenario.pop(key, None)
    scenario.update(data_size_mode='per_data_unit', data_unit_count=len(amounts),
        data_size_min_bytes=int(amounts.min()), data_size_max_bytes=int(amounts.max()),
        data_size_mean_bytes=float(amounts.mean()), data_size_std_bytes=float(amounts.std()),
        total_data_bytes=int(amounts.sum()), total_data_bits=int(amounts.sum()) * 8)
    if metadata:
        scenario.update(metadata)
    scenario_path.write_text(json.dumps(scenario, ensure_ascii=False, indent=2), encoding='utf-8')
    from .paper2_export import validate_paper2_inputs
    validate_paper2_inputs(folder)
    return scenario
