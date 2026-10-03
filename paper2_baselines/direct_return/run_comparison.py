"""Compare three direct-to-base schedulers on one saved input and one channel snapshot set."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from paper1.paper2_dataset import Paper2Dataset
from paper2_model.channel import ChannelModel
from paper2_model.experiment_config import communication_configs
from paper2_model.return_simulation import ReturnSimulation, load_data_units, write_simulation_result


ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / 'paper2_baselines' / 'direct_return'
INPUT_DIR = BASELINE_DIR / 'input' / 'input_ref_5to10mbit_v10'
OUTPUT_DIR = BASELINE_DIR / 'results' / 'direct_scheduler_comparison_seed2026_data2027'
STRATEGIES = ('rate_greedy', 'round_robin', 'oldest_first')


def _hashes(folder):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(folder).iterdir()) if path.is_file()}


def _row(name, result):
    return dict(strategy=name, complete_units=result['complete_data_unit_count'],
        delay_sample_count=result['complete_data_unit_count'],
        complete_unit_rate=result['complete_data_unit_rate'],
        delivered_bit_ratio=result['delivered_bit_ratio'],
        completed_mean_delay_s=result['completed_data_conditional_mean_delay_s'],
        completed_p95_delay_s=result['completed_data_p95_delay_s'],
        completed_max_delay_s=result['completed_data_max_delay_s'],
        pre_first_send_mean_s=result['completed_pre_first_send_mean_s'],
        pre_first_send_p95_s=result['completed_pre_first_send_p95_s'],
        post_first_send_to_complete_mean_s=result['completed_post_first_send_mean_s'],
        post_first_send_to_complete_p95_s=result['completed_post_first_send_p95_s'],
        final_undelivered_bits=result['undelivered_data_bits'],
        total_backlog_peak_bits=result['total_backlog_peak_bits'],
        max_single_uav_peak_buffer_bits=result['max_single_uav_peak_buffer_bits'],
        buffer_constraint_violated=result['buffer_constraint_violated'],
        conservation_passed=result['conservation_passed'])


def _write_comparison(results, input_hashes):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [_row(name, results[name]) for name in STRATEGIES]
    with (OUTPUT_DIR / 'scheduler_comparison.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    common = np.logical_and.reduce([results[name]['complete'] for name in STRATEGIES])
    common_rows = []
    for name in STRATEGIES:
        delays = results[name]['delays_s'][common]
        common_rows.append(dict(strategy=name, common_complete_count=int(common.sum()),
            mean_delay_s=float(delays.mean()) if len(delays) else None,
            p95_delay_s=float(np.percentile(delays, 95)) if len(delays) else None,
            max_delay_s=float(delays.max()) if len(delays) else None))
    with (OUTPUT_DIR / 'common_completed_delay_comparison.csv').open(
            'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(common_rows[0]))
        writer.writeheader(); writer.writerows(common_rows)
    config = dict(input_folder=str(INPUT_DIR), input_sha256=input_hashes,
        command=r'.\.runtime\python\python.exe -m paper2_baselines.direct_return.run_comparison',
        channel_config=results['rate_greedy']['channel_config'],
        simulation_config=results['rate_greedy']['simulation_config'],
        strategy_rules={
            'rate_greedy': ('At each slot, among full-slot eligible UAVs with nonempty start-of-slot '
                'queues, compute the interference-free single-subchannel direct rate, select up to four '
                'highest rates, and assign distinct subchannels.'),
            'round_robin': ('Scan stable UAV IDs from a cyclic pointer, skip ineligible/empty UAVs, select '
                'up to four once each, then advance past the last selected UAV.'),
            'oldest_first': ('Rank each eligible/nonempty UAV by the age of its FIFO-head unit at slot '
                'start; select up to four oldest, breaking equal ages by UAV ID.'),
        }, same_channel_snapshots=True,
        common_completed_count=int(common.sum()),
        common_set_limitation=('Common-set delays describe only units completed by all strategies and do '
            'not replace all-unit completion-rate comparisons.'), rows=rows, common_rows=common_rows)
    (OUTPUT_DIR / 'comparison_summary.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.runtime' / 'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {'rate_greedy': 'Rate greedy', 'round_robin': 'Round robin',
              'oldest_first': 'Oldest first'}
    total = len(results['rate_greedy']['data_units'])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name in STRATEGIES:
        delays = np.sort(results[name]['delays_s'][results[name]['complete']])
        ax.step(delays, np.arange(1, len(delays) + 1) / total, where='post', label=labels[name])
    ax.set(xlabel='Completed end-to-end delay (s)', ylabel='Fraction of all 8000 data units', ylim=(0, 1.02))
    ax.grid(True, alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'all_unit_delay_cdf.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name in STRATEGIES:
        records = results[name]['slot_records']
        ax.plot([(row['slot_zero_based'] + 1) * results[name]['slot_seconds'] for row in records],
                [row['system_undelivered_bits'] for row in records], label=labels[name])
    ax.set(xlabel='Time (s)', ylabel='System undelivered data (bit)')
    ax.grid(True, alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'scheduler_backlog_comparison.png', dpi=160); plt.close(fig)

    uavs = np.arange(results['rate_greedy']['num_uavs'] if 'num_uavs' in results['rate_greedy'] else 20)
    width = .25
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for offset, name in enumerate(STRATEGIES):
        result = results[name]
        rates, means = [], []
        for uav in uavs:
            ids = np.array([k for k, unit in enumerate(result['data_units'])
                            if unit.source_uav_id == uav], int)
            complete = result['complete'][ids]
            rates.append(float(complete.mean()))
            means.append(float(result['delays_s'][ids[complete]].mean()) if complete.any() else np.nan)
        x = uavs + (offset - 1) * width
        axes[0].bar(x, np.array(rates) * 100, width, label=labels[name])
        axes[1].bar(x, means, width, label=labels[name])
    axes[0].set(ylabel='Complete delivery rate (%)', ylim=(0, 105))
    axes[1].set(xlabel='UAV ID', ylabel='Mean completed delay (s)', xticks=uavs)
    for ax in axes: ax.grid(True, axis='y', alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / 'uav_delivery_delay_comparison.png', dpi=160); plt.close(fig)


def run_comparison():
    channel_config, simulation_config = communication_configs()
    dataset = Paper2Dataset(INPUT_DIR)
    model = ChannelModel(dataset, channel_config)
    units = load_data_units(dataset)
    simulation = ReturnSimulation(dataset, model, units, simulation_config)
    runners = dict(rate_greedy=simulation.run_direct_rate_greedy,
                   round_robin=simulation.run_round_robin,
                   oldest_first=simulation.run_oldest_data_first)
    input_hashes = _hashes(INPUT_DIR)
    results = {}
    for name in STRATEGIES:
        result = runners[name]()
        result.update(experiment_name=f'direct scheduler comparison: {name}',
            input_folder=str(INPUT_DIR), output_folder=str(OUTPUT_DIR / name),
            source_input_folder=str(INPUT_DIR), input_sha256=input_hashes,
            actual_uav_count=int(dataset.scenario['num_uavs']),
            data_size_seed=int(dataset.scenario['data_size_seed']))
        write_simulation_result(result, OUTPUT_DIR / name)
        results[name] = result
    frozen = json.loads((BASELINE_DIR / 'verification' /
                         'baseline_frozen.json').read_text(encoding='utf-8'))
    baseline = frozen['regression_reference']
    reproduced = results['rate_greedy']
    if not (reproduced['complete_data_unit_rate'] == baseline['complete_data_unit_rate'] and
            np.isclose(reproduced['completed_data_conditional_mean_delay_s'],
                       baseline['all_data_mean_delay_s'], atol=1e-9, rtol=0)):
        raise RuntimeError('rate-greedy result did not reproduce the validated baseline')
    _write_comparison(results, input_hashes)
    return results


if __name__ == '__main__':
    results = run_comparison()
    print(json.dumps({name: _row(name, results[name]) for name in STRATEGIES},
                     ensure_ascii=False, indent=2))
