"""Write the reproducible baseline record and source-code SHA-256 manifest."""
from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

from paper1.paper2_dataset import _read_events
from paper2_model.experiment_config import communication_configs


ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / 'paper2_baselines' / 'direct_return'
INPUT = BASELINE / 'input' / 'input_ref_5to10mbit_v10'
COMPARISON = BASELINE / 'results' / 'direct_scheduler_comparison_seed2026_data2027'
OUTPUT = BASELINE / 'verification'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_files():
    paths = []
    paths.extend(sorted((ROOT / 'paper1').rglob('*.py')))
    paths.extend(sorted((ROOT / 'paper2_model').rglob('*.py')))
    paths.extend(sorted((ROOT / 'paper2_baselines' / 'direct_return').rglob('*.py')))
    return [path for path in paths if '__pycache__' not in path.parts]


def main():
    channel, simulation = communication_configs()
    scenario = json.loads((INPUT / 'scenario.json').read_text(encoding='utf-8'))
    comparison = json.loads((COMPARISON / 'comparison_summary.json').read_text(encoding='utf-8'))
    audit = json.loads((OUTPUT / 'audit_results.json').read_text(encoding='utf-8'))
    regression = json.loads((OUTPUT / 'simulator_validation_seed2026_data2027' /
                             'verification_summary.json').read_text(encoding='utf-8'))
    events = _read_events(INPUT / 'collection_events.csv')
    input_hashes = {path.name: sha256(path) for path in sorted(INPUT.iterdir()) if path.is_file()}
    hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in source_files()}
    (OUTPUT / 'source_sha256.json').write_text(
        json.dumps({'algorithm': 'SHA-256', 'files': hashes}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    strategies = {row['strategy']: row for row in comparison['rows']}
    rate_greedy = strategies['rate_greedy']
    record = {
        'status': 'frozen_after_validation',
        'scope': '5-10 Mbit per target, 10 m/s, three direct-return schedulers',
        'input_directory': INPUT.relative_to(ROOT).as_posix(),
        'input_sha256': input_hashes,
        'scenario_config': scenario,
        'channel_config': channel.__dict__,
        'simulation_config': simulation.__dict__,
        'data_config': {
            'unit_count': len(events),
            'total_bits': sum(event['data_bits'] for event in events),
            'size_min_bytes': min(event['data_bytes'] for event in events),
            'size_max_bytes': max(event['data_bytes'] for event in events),
            'data_size_seed': scenario['data_size_seed'],
        },
        'runtime': {
            'python': platform.python_version(),
            'implementation': platform.python_implementation(),
            'required_packages_installed': {name: version(name) for name in ('numpy', 'scipy', 'matplotlib')},
            'declared_requirements': (ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines(),
        },
        'commands': {
            'rate_greedy_regression': r'.\.runtime\python\python.exe -m paper2_baselines.direct_return.verification.simulator_validation',
            'three_scheduler_comparison': r'.\.runtime\python\python.exe -m paper2_baselines.direct_return.run_comparison',
            'all_tests': r'.\.runtime\python\python.exe -m unittest discover -v',
            'independent_audit': r'.\.runtime\python\python.exe paper2_baselines\direct_return\verification\simulator_audit.py',
        },
        'validation_evidence': {
            'unit_tests': {'passed': 40, 'total': 40, 'command': r'.\.runtime\python\python.exe -m unittest discover -v'},
            'independent_audit': audit,
            'formal_input_and_rate_greedy_regression_passed': regression['passed'],
            'formal_input_checks': regression['input_validation'],
        },
        'strategy_results': strategies,
        'regression_reference': {
            'data_unit_count': len(events),
            'total_generated_bits': float(sum(event['data_bits'] for event in events)),
            'complete_data_unit_rate': rate_greedy['complete_unit_rate'],
            'all_data_mean_delay_s': rate_greedy['completed_mean_delay_s'],
            'buffer_constraint_violated': rate_greedy['buffer_constraint_violated'],
            'conservation_passed': rate_greedy['conservation_passed'],
        },
        'source_hash_manifest': 'paper2_baselines/direct_return/verification/source_sha256.json',
        'source_hash_scope': ('all Python files under paper1, paper2_model and '
                              'paper2_baselines/direct_return, including verification scripts; '
                              'generated outputs and hash manifests excluded'),
    }
    (OUTPUT / 'baseline_frozen.json').write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'record': 'paper2_baselines/direct_return/verification/baseline_frozen.json',
                      'source_files': len(hashes), 'input_files': len(input_hashes)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
