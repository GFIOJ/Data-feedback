"""Paper 1 launcher. Edit CONFIG, then run this module with the bundled Python."""
from __future__ import annotations

import os
from pathlib import Path


CONFIG = {
    # Paper 1 scenario
    'scene_width_m': 2000,
    'scene_length_m': 2000,
    'boundary_margin_m': 50.0,
    'blocks_x': 10,
    'blocks_y': 10,
    'ground_slots': 60000,
    'average_stack_tiers': 5,
    'aggregate_strips_per_region': 4,
    'num_effective_tags': 8000,
    'num_uavs': 20,
    'uav_speed_mps': 10.0,
    'depot_altitude_m': 15.0,
    'safety_radius_m': 5.0,
    'grid_resolution_m': 20.0,
    'max_scan_segment_m': 250.0,
    'max_task_load_ratio': 1.0,
    'seed': 1,
    # Paper 2 known inputs
    'slot_seconds': 1.0,
    'data_per_collection_mb': 2.0,
    'base_station_position_m': [1000.0, 1000.0, 30.0],
    'output_dir': 'paper1/outputs/paper2_input_10mps',
    # Runtime
    'quick': True,
    'visualize': True,
}


def main(config=None):
    cfg = dict(CONFIG if config is None else config)
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault('MPLCONFIGDIR', str(root / '.runtime' / 'matplotlib'))
    if not cfg['visualize']:
        import matplotlib
        matplotlib.use('Agg')
    from paper1.main_stage1_hover_planner import main_stage1_hover_planner
    from paper1.main_stage2_uav_planner import main_stage2_uav_planner
    from paper1.paper2_export import export_paper2_inputs

    stage1 = main_stage1_hover_planner(
        SceneWidth=cfg['scene_width_m'], SceneLength=cfg['scene_length_m'],
        NumBlocksX=cfg['blocks_x'], NumBlocksY=cfg['blocks_y'], AggregateYard=True,
        BoundaryMargin=cfg['boundary_margin_m'],
        GroundSlots=cfg['ground_slots'], StackTiers=cfg['average_stack_tiers'],
        StripsPerRegion=cfg['aggregate_strips_per_region'], NumTags=cfg['num_effective_tags'],
        TagSeed=cfg['seed'], LayoutSeed=cfg['seed'],
        RFIDOverrides={'maxScanSegmentLength': cfg['max_scan_segment_m']},
        EnableFlyByRFID=True, Visualize=cfg['visualize'], Verbose=True)
    budget = dict(GameRestarts=1, GameMaxIter=3, MaxImproveIters=1,
                  TopKExactRefine=1, MaxExactRefineCalls=2,
                  EnableConflictFeedback=False) if cfg['quick'] else {}
    depot = [cfg['scene_width_m'] / 2, cfg['scene_length_m'] / 2, cfg['depot_altitude_m']]
    stage2 = main_stage2_uav_planner(stage1, NumUAVs=cfg['num_uavs'], DepotPos=depot,
        UAVSpeed=cfg['uav_speed_mps'], SafetyRadius=cfg['safety_radius_m'],
        GridRes=cfg['grid_resolution_m'], InflateRadius=0., Seed=cfg['seed'],
        MaxTaskLoadRatio=cfg['max_task_load_ratio'],
        LegacyCompatibility=False, Visualize=cfg['visualize'], Verbose=True, **budget)
    if not stage2.validation.passed:
        raise RuntimeError(f'Path validation failed: {stage2.validation.reasons}')
    cfg.update(depot_position_m=depot, ground_utilization=stage1.scene.ground_utilization)
    output = root / cfg['output_dir']
    scenario = export_paper2_inputs(stage1, stage2, output,
        slot_seconds=cfg['slot_seconds'],
        data_per_collection_bytes=round(cfg['data_per_collection_mb'] * 1_000_000),
        base_station_position_m=cfg['base_station_position_m'],
        scenario_config=cfg)
    print(f"Paper 2 inputs saved to {output}")
    print(f"scan tasks={scenario['num_scan_tasks']}, target coverage={scenario['inspection_target_coverage_rate']:.2%}, "
          f"makespan={scenario['makespan_seconds']:.2f}s")
    import csv
    with (output / 'uav_metadata.csv').open(encoding='utf-8-sig') as file:
        for row in csv.DictReader(file):
            print(f"UAV {row['uav_id']}: takeoff={float(row['takeoff_time_s']):.3f}s, "
                  f"return={float(row['return_time_s']):.3f}s, collections={row['collection_count']}, "
                  f"data={int(row['generated_data_bits']) / 1e6:.3f} Mbit")
    print(f"System arrival distribution saved to {output / 'system_arrivals.csv'}")
    if cfg['visualize']:
        import matplotlib.pyplot as plt
        figures = root / 'paper1' / 'figures'
        figures.mkdir(exist_ok=True)
        for number in plt.get_fignums():
            figure = plt.figure(number)
            figure.savefig(figures / f'{figure.get_label() or number}.png', dpi=160)
        if 'agg' not in plt.get_backend().lower():
            plt.show()
    return stage1, stage2


if __name__ == '__main__':
    main()
