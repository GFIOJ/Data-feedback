"""Stage 2 public entry point."""
from time import perf_counter
import numpy as np
from .common import Struct, options
from .uav_mission_planner import (PLANNER_DEFAULTS, build_library, task_assignment,
                                 optimize_2opt, resolve_conflict, evaluate)


def main_stage2_uav_planner(stage1=None, *args, **kwargs):
    opts = options(dict(PLANNER_DEFAULTS, NumUAVs=5, UAVSpeed=5., SafetyRadius=1.,
        GridRes=3., InflateRadius=0., GameRestarts=1, Visualize=True), args, kwargs)
    if stage1 is None:
        import main_stage1_hover_planner as first
        stage1 = first._last_stage1
        if stage1 is None:
            stage1 = first.main_stage1_hover_planner(Visualize=False, Verbose=opts.Verbose)
    for key in ('Containers', 'Tags', 'scene', 'rfid', 'hoverPoints', 'hoverIdx', 'cov_prob', 'coverageRate', 'Khover', 'Ntag'):
        if key not in stage1:
            raise ValueError(f'Stage 1 result is missing {key}')
    start = perf_counter()
    params = {k: opts[k] for k in PLANNER_DEFAULTS}
    params['ServicePaths'] = stage1.get('scanServicePaths') or None
    if params['ServicePaths']:
        params['LegacyCompatibility'] = False
    state = build_library(stage1.hoverPoints, stage1.Containers, stage1.scene, stage1.rfid, opts.NumUAVs, **params)
    if opts.LegacyCompatibility and '_rng' in stage1:
        state.rng = stage1['_rng']
    for label, step in [('task assignment', task_assignment), ('2-opt', optimize_2opt), ('conflict resolution', resolve_conflict)]:
        if opts.Verbose:
            print(f'Stage 2: {label}...', flush=True)
        step(state)
    solution, makespan, stats = evaluate(state)
    elapsed = perf_counter() - start
    load = float(np.ptp(solution.uavCosts) / max(solution.uavCosts.mean(), 1e-15) * 100)
    result = Struct(stage1=stage1, planner_state=state, solution=solution, validation=solution.validation,
        makespan=makespan, alignedObjective=stats.alignedObjective, objectiveName=stats.objectiveName,
        objectiveMode=stats.objectiveMode, cbs_stats=stats, numUAVs=opts.NumUAVs, uavSpeed=opts.UAVSpeed,
        safetyR=opts.SafetyRadius, gridRes=opts.GridRes, inflateR=opts.InflateRadius, depotPos=state.depot,
        loadImbalance=load, stage2Time=elapsed, totalTime=stage1.get('stage1Time', 0) + elapsed,
        time_to_first_feasible_sec=None, timing_boundary='stage 2 only; stage 1 excluded',
        ablationSwitches=Struct({k: state.pm[k] for k in ('AblationMethod', 'EnableTaskTransfer',
            'EnableTaskExchange', 'EnableSeqAdjust', 'EnableConflictFeedback', 'EnableLocalSearch')}))
    if opts.Visualize:
        from .visualization import visualize_stage2
        visualize_stage2(result)
    if opts.Verbose:
        print(f'Stage 2 completed: makespan={makespan:.3f}s, SOC={solution.soc:.3f}s, '
              f'load imbalance={load:.1f}%, objective={solution.objective:.3f}, '
              f'validation={solution.validation.passed}, elapsed={elapsed:.2f}s', flush=True)
    return result


if __name__ == '__main__':
    main_stage2_uav_planner()
    import matplotlib.pyplot as plt
    plt.show()
