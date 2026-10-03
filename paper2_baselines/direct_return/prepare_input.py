"""Independent 5-10 Mbit per-target experiment on the validated 10 m/s input."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from paper1.paper2_dataset import Paper2Dataset, _read_events, recompute_data_amounts
from paper2_model.experiment_config import communication_configs, event_identity, trajectory_speed_check
from paper2_model.return_simulation import run_yard_baseline, write_simulation_result


ROOT = Path(__file__).resolve().parents[2]
SOURCE_INPUT_DIR = ROOT / 'paper1' / 'outputs' / 'paper2_input_10mps'
INPUT_DIR = ROOT / 'paper2_baselines' / 'direct_return' / 'input' / 'input_ref_5to10mbit_v10'
OUTPUT_DIR = ROOT / 'paper2_baselines' / 'direct_return' / 'results' / 'direct_ref_5to10mbit_v10_seed2026_data2027'
MIN_BYTES, MAX_BYTES = 625_000, 1_250_000
DATA_SIZE_SEED = 2027


def sampled_sizes(count, seed=DATA_SIZE_SEED):
    return np.random.default_rng(seed).integers(
        MIN_BYTES, MAX_BYTES + 1, size=count, dtype=np.int64)


def _prepare_input():
    source_events = _read_events(SOURCE_INPUT_DIR / 'collection_events.csv')
    if not INPUT_DIR.exists():
        shutil.copytree(SOURCE_INPUT_DIR, INPUT_DIR)
        sizes = sampled_sizes(len(source_events))
        recompute_data_amounts(INPUT_DIR, sizes, metadata={
            'data_size_distribution': 'discrete_uniform_integer_bytes_inclusive',
            'data_size_min_bytes_requested': MIN_BYTES,
            'data_size_max_bytes_requested': MAX_BYTES,
            'data_size_seed': DATA_SIZE_SEED,
            'expected_total_bits_for_8000_units': 60_000_000_000,
        })
        with (INPUT_DIR / 'data_unit_sizes.csv').open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(['data_id', 'data_bytes', 'data_bits'])
            writer.writerows((data_id, int(value), int(value) * 8)
                             for data_id, value in enumerate(sizes))
    events = _read_events(INPUT_DIR / 'collection_events.csv')
    sizes = np.array([event['data_bytes'] for event in events], dtype=np.int64)
    return source_events, events, sizes


def run_variable_size_experiment():
    channel_config, simulation_config = communication_configs()
    print(f'patrol_speed=10 m/s, data_size=[{MIN_BYTES},{MAX_BYTES}] byte inclusive, '
          f'scene_seed=1, channel_seed={channel_config.random_seed}, '
          f'data_size_seed={DATA_SIZE_SEED}, input={INPUT_DIR}')
    source_events, events, sizes = _prepare_input()
    dataset = Paper2Dataset(INPUT_DIR)
    source = Paper2Dataset(SOURCE_INPUT_DIR)
    speed_ok, maximum_speed = trajectory_speed_check(dataset, 10.)
    regenerated = sampled_sizes(len(events))
    with np.load(INPUT_DIR / 'slot_arrivals.npz') as arrivals:
        arrival_bytes = int(arrivals['new_data_bytes_total'].sum())
        arrival_bits = int(arrivals['new_data_bits_total'].sum())
    with (INPUT_DIR / 'uav_metadata.csv').open(encoding='utf-8-sig') as file:
        uav_generated_bytes = sum(int(row['generated_data_bytes']) for row in csv.DictReader(file))
    immutable_files = ('trajectories.npz', 'node_states.npz', 'environment.npz')
    checks = {
        'all_sizes_are_integer_bytes_in_requested_range': bool(
            np.issubdtype(sizes.dtype, np.integer) and
            np.all((sizes >= MIN_BYTES) & (sizes <= MAX_BYTES))),
        'bits_equal_eight_times_each_saved_byte_size': all(
            event['data_bits'] == event['data_bytes'] * 8 for event in events),
        'same_seed_reproduces_saved_sizes': np.array_equal(sizes, regenerated),
        'stable_data_id_order': [event['data_id'] for event in events] == list(range(len(events))),
        'non_amount_event_fields_match_original_10_mps_input': len(events) == len(source_events)
            and all(event_identity(before) == event_identity(after)
                    for before, after in zip(source_events, events)),
        'trajectory_node_state_and_environment_files_unchanged': all(
            (SOURCE_INPUT_DIR / name).read_bytes() == (INPUT_DIR / name).read_bytes()
            for name in immutable_files),
        'scenario_and_trajectory_confirm_10_mps': dataset.scenario['uav_speed_mps'] == 10.
            and speed_ok and np.isclose(maximum_speed, 10., atol=1e-7),
        'actual_time_horizon_matches_original': dataset.scenario['slot_count'] == source.scenario['slot_count']
            and dataset.scenario['simulation_end_time_s'] == source.scenario['simulation_end_time_s'],
        'event_arrival_and_uav_totals_match': int(sizes.sum()) == arrival_bytes == uav_generated_bytes
            and int(sizes.sum()) * 8 == arrival_bits,
        'paper2_input_validation_passed': bool(json.loads(
            (INPUT_DIR / 'validation.json').read_text(encoding='utf-8'))['passed']),
        'buffer_limit_is_1gb_per_uav': simulation_config.buffer_capacity_bits == 8_000_000_000,
    }
    distribution = {
        'distribution': 'discrete_uniform_integer_bytes_inclusive',
        'data_size_seed': DATA_SIZE_SEED, 'requested_min_bytes': MIN_BYTES,
        'requested_max_bytes': MAX_BYTES, 'actual_min_bytes': int(sizes.min()),
        'actual_max_bytes': int(sizes.max()), 'mean_bytes': float(sizes.mean()),
        'std_bytes_population': float(sizes.std()), 'unit_count': len(sizes),
        'total_bytes': int(sizes.sum()), 'total_bits': int(sizes.sum()) * 8,
        'expected_total_bits_for_8000_units': 60_000_000_000,
    }
    validation = {
        'passed': bool(all(checks.values())), 'checks': {key: bool(value) for key, value in checks.items()},
        'maximum_moving_segment_speed_mps': maximum_speed,
        'slot_count': int(dataset.scenario['slot_count']),
        'simulation_end_time_s': float(dataset.scenario['simulation_end_time_s']),
        'last_collection_time_s': float(max(event['collection_complete_time_s'] for event in events)),
        'distribution': distribution,
    }
    if not validation['passed']:
        raise RuntimeError('variable-size input validation failed: ' +
                           ', '.join(key for key, value in checks.items() if not value))

    result = run_yard_baseline(INPUT_DIR, OUTPUT_DIR, channel_config, simulation_config)
    original = np.array([unit.original_bits for unit in result['data_units']])
    validation['runtime_checks'] = {
        'queue_and_link_capacity_checks_completed_without_error': True,
        'loaded_sizes_equal_saved_event_sizes': np.array_equal(original, sizes * 8),
        'data_conservation_passed': bool(result['conservation_passed']),
        'delivery_flags_use_each_units_own_size': bool(np.array_equal(
            result['complete'], result['base_received_bits'] >= original -
            (simulation_config.absolute_tolerance_bits + simulation_config.relative_tolerance * original))),
        'loaded_total_matches_saved_total': result['total_generated_bits'] == distribution['total_bits'],
    }
    validation['passed'] = validation['passed'] and all(validation['runtime_checks'].values())
    result.update(
        experiment_name='5-10 Mbit variable-size / original 10 m/s direct-return baseline',
        input_folder=str(INPUT_DIR), output_folder=str(OUTPUT_DIR),
        source_input_folder=str(SOURCE_INPUT_DIR), input_validation=validation,
        data_size_seed=DATA_SIZE_SEED, data_size_distribution=distribution,
        actual_uav_count=int(dataset.scenario['num_uavs']), actual_target_count=len(events),
        last_collection_time_s=validation['last_collection_time_s'],
        last_return_time_s=float(np.max(dataset.node_states['return_time_s'])),
        last_delivery_time_s=(float(np.nanmax(result['delivery_time_s']))
                              if result['complete'].any() else None),
        parameter_sources={
            'literature_range': ('The 2022 paper uses a 1-10 Mbit packet range; this experiment uses the '
                'requested 5-10 Mbit inclusive subrange. The 2025 paper supplies the 5 MHz link bandwidth.'),
            'project_settings': ('10 m/s patrol speed, four subchannels, 20 MHz aggregate bandwidth, '
                '1 GB buffer, channel model and direct-return scheduler remain project settings.'),
        })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'input_validation.json').write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    write_simulation_result(result, OUTPUT_DIR)
    with (OUTPUT_DIR / 'incomplete_data_units.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['data_id', 'source_uav_id', 'original_bytes', 'original_bits',
                         'collection_complete_time_s', 'base_received_bits', 'remaining_bits'])
        for k in np.flatnonzero(~result['complete']):
            unit = result['data_units'][k]
            writer.writerow([unit.data_id, unit.source_uav_id, unit.original_bits / 8,
                unit.original_bits, unit.collection_complete_time_s,
                result['base_received_bits'][k], unit.original_bits - result['base_received_bits'][k]])
    return result


if __name__ == '__main__':
    report = run_variable_size_experiment()
    diagnostic = report['capacity_diagnostic']
    print(json.dumps({
        'result_directory': str(OUTPUT_DIR), 'data_size_distribution': report['data_size_distribution'],
        'complete_units': report['complete_data_unit_count'],
        'complete_unit_rate': report['complete_data_unit_rate'],
        'delivered_bit_ratio': report['delivered_bit_ratio'],
        'undelivered_bits': report['undelivered_data_bits'],
        'buffer_violation': report['buffer_constraint_violated'],
        'capacity_upper_bound_bits': diagnostic['bs_capacity_upper_bound_bits'],
        'demand_to_upper_bound': diagnostic['load_to_upper_bound_ratio'],
        'upper_bound_conclusion': diagnostic['conclusion'],
    }, ensure_ascii=False, indent=2))
