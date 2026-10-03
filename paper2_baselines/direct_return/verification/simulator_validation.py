"""Reproduce the current rate-greedy baseline in an isolated verification folder."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from paper1.paper2_export import validate_paper2_inputs

from paper2_model.experiment_config import communication_configs
from paper2_model.return_simulation import run_yard_baseline


ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR = ROOT / 'paper2_baselines' / 'direct_return'
INPUT = BASELINE_DIR / 'input' / 'input_ref_5to10mbit_v10'
FROZEN = BASELINE_DIR / 'verification' / 'baseline_frozen.json'
OUTPUT = BASELINE_DIR / 'verification' / 'simulator_validation_seed2026_data2027'
EXPECTED = {
    'data_unit_count': 8000,
    'total_generated_bits': 60_034_387_504.0,
    'complete_data_unit_rate': 1.0,
    'all_data_mean_delay_s': 63.58476096944617,
    'buffer_constraint_violated': False,
    'conservation_passed': True,
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _compare(actual, reference):
    rows = {}
    for key, expected in reference.items():
        value = actual[key]
        if isinstance(expected, bool) or isinstance(expected, int):
            passed = value == expected
            difference = None
        else:
            difference = float(value) - float(expected)
            passed = abs(difference) <= 1e-9 * max(1.0, abs(float(expected)))
        rows[key] = {'actual': value, 'reference': expected,
                     'difference': difference, 'passed': bool(passed)}
    return rows


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    input_report = validate_paper2_inputs(INPUT, OUTPUT / 'input_validation.json')
    channel_config, simulation_config = communication_configs()
    result_dir = OUTPUT / 'rate_greedy'
    result = run_yard_baseline(INPUT, result_dir, channel_config, simulation_config)
    if FROZEN.exists():
        reference = json.loads(FROZEN.read_text(encoding='utf-8'))['regression_reference']
        reference_source = str(FROZEN)
    else:
        reference = EXPECTED
        reference_source = 'bootstrap constants in direct_return verification'
    expected_comparison = _compare(result, EXPECTED)
    frozen_comparison = _compare(result, {key: reference[key] for key in EXPECTED})
    report = {
        'passed': (input_report['passed'] and
                   all(row['passed'] for row in expected_comparison.values()) and
                   all(row['passed'] for row in frozen_comparison.values())),
        'input_folder': str(INPUT),
        'result_folder': str(result_dir),
        'input_hashes_sha256': {
            name: _sha256(INPUT / name) for name in (
                'scenario.json', 'collection_events.csv', 'trajectories.npz',
                'node_states.npz', 'environment.npz', 'slot_arrivals.npz')
        },
        'input_validation': input_report,
        'expected_comparison': expected_comparison,
        'reference_source': reference_source,
        'frozen_comparison': frozen_comparison,
        'channel_config': result['channel_config'],
        'simulation_config': result['simulation_config'],
    }
    (OUTPUT / 'verification_summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['passed']:
        raise SystemExit('simulator verification failed')


if __name__ == '__main__':
    main()
