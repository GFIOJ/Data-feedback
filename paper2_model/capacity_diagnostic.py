"""Necessary-condition diagnostic for the base station receive-capacity upper bound."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import warnings

import numpy as np

from paper1.paper2_dataset import Paper2Dataset
from .channel import ChannelConfig, ChannelModel


UPPER_BOUND_ASSUMPTIONS = [
    'Only full-slot scheduling-eligible UAVs are considered.',
    'Each UAV uses at most one subchannel and each base-station subchannel receives at most one link.',
    'The best min(F, online UAV count) interference-free UAV-to-base rates are selected in every slot.',
    'Selected UAVs are assumed to always have data; arrival times, buffer locations and relay overhead are ignored.',
    'This is a receive-capacity upper bound, not achieved throughput or a delivery schedule.',
]


def base_station_capacity_upper_bound(dataset, config=None, *, model=None,
                                      total_data_bits=None, data_unit_count=None):
    """Compute U_BS in bit from cached, reproducible per-slot channel snapshots."""
    model = model or ChannelModel(dataset, config)
    config = model.config
    slot_count = int(dataset.scenario['slot_count'])
    slot_seconds = float(dataset.scenario['slot_seconds'])
    if total_data_bits is None or data_unit_count is None:
        with np.load(Path(dataset.folder) / 'slot_arrivals.npz') as arrivals:
            if total_data_bits is None:
                total_data_bits = int(arrivals['new_data_bits_total'].sum())
            if data_unit_count is None:
                data_unit_count = int(arrivals['new_items_total'].sum())
    total_data_bits, data_unit_count = int(total_data_bits), int(data_unit_count)
    if total_data_bits < 0 or data_unit_count <= 0:
        raise ValueError('total_data_bits must be nonnegative and data_unit_count must be positive')

    b = config.subchannel_bandwidth_hz
    noise = config.noise_psd_w_per_hz * b
    per_slot_bits = np.zeros(slot_count)
    selected_counts = np.zeros(slot_count, int)
    for slot in range(slot_count):
        snapshot = model.snapshot(slot)
        online = np.flatnonzero(snapshot.scheduling_eligible[:model.num_uavs])
        if not len(online):
            continue
        gains = snapshot.power_gain[online, model.base_station_id]
        rates = b * np.log2(1 + config.transmit_power_w * gains / noise)
        count = min(config.num_subchannels, len(rates))
        selected_counts[slot] = count
        per_slot_bits[slot] = np.partition(rates, -count)[-count:].sum() * slot_seconds
    upper = float(per_slot_bits.sum())
    ratio = float(total_data_bits / upper) if upper > 0 else float('inf')
    infeasible = bool(total_data_bits > upper)
    return dict(
        channel_config=asdict(config), random_seed=int(config.random_seed),
        slot_count=slot_count, slot_seconds=slot_seconds, data_unit_count=data_unit_count,
        total_data_bits=total_data_bits, bs_capacity_upper_bound_bits=upper,
        load_to_upper_bound_ratio=ratio, infeasible_by_upper_bound=infeasible,
        conclusion=('infeasible_by_upper_bound' if infeasible else 'not_ruled_out_by_upper_bound'),
        assumptions=list(UPPER_BOUND_ASSUMPTIONS), per_slot_upper_bound_bits=per_slot_bits,
        selected_uav_count_per_slot=selected_counts)


def candidate_loads(report, ratios=(.10, .25, .50)):
    """Return exploratory equal-size loads without writing or changing the input dataset."""
    upper = float(report['bs_capacity_upper_bound_bits'])
    count = int(report['data_unit_count'])
    per_unit_upper_bits = upper / count
    rows = []
    for fraction in ratios:
        if not np.isfinite(fraction) or fraction <= 0:
            raise ValueError('candidate ratios must be positive and finite')
        amount_bytes = int(np.floor(per_unit_upper_bits * fraction / 8))
        if amount_bytes < 1:
            warnings.warn(f'{fraction:.0%} candidate is below 1 byte and was not generated', RuntimeWarning)
            continue
        amount_bits = amount_bytes * 8
        total_bits = amount_bits * count
        rows.append(dict(fraction_of_per_unit_upper_bound=float(fraction),
            data_per_collection_bytes=amount_bytes, data_per_collection_bits=amount_bits,
            total_data_bits=total_bits,
            load_to_bs_upper_bound_ratio=float(total_bits / upper)))
    return rows


def print_diagnostic(report, candidates):
    config = report['channel_config']
    print('Base-station receive-capacity upper-bound diagnostic')
    print('ChannelConfig:')
    for name, value in config.items():
        print(f'  {name}={value}')
    print(f'total_data={report["total_data_bits"]:.0f} bit')
    print(f'U_BS={report["bs_capacity_upper_bound_bits"]:.6f} bit')
    print(f'load/U_BS={report["load_to_upper_bound_ratio"]:.6f}')
    print(f'conclusion={report["conclusion"]}')
    print('Upper-bound assumptions and scope:')
    for assumption in report['assumptions']:
        print(f'  - {assumption}')
    print('\nExploratory candidate loads (necessary condition only):')
    print('fraction  bytes/item  bits/item  total_bits  total/U_BS')
    for row in candidates:
        print(f'{row["fraction_of_per_unit_upper_bound"]:>7.0%}  '
              f'{row["data_per_collection_bytes"]:>10d}  {row["data_per_collection_bits"]:>9d}  '
              f'{row["total_data_bits"]:>10d}  {row["load_to_bs_upper_bound_ratio"]:>10.6f}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(Path(__file__).resolve().parents[1] /
                                               'paper2_baselines' / 'direct_return' / 'input' /
                                               'input_ref_5to10mbit_v10'))
    args = parser.parse_args(argv)
    from .experiment_config import communication_configs
    channel_config, _ = communication_configs()
    report = base_station_capacity_upper_bound(Paper2Dataset(args.input), channel_config)
    rows = candidate_loads(report)
    print_diagnostic(report, rows)
    return report, rows


if __name__ == '__main__':
    main()
