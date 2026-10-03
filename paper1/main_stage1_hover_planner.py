"""Stage 1: environment -> sparse RFID probabilities -> coverage game."""
from time import perf_counter
import numpy as np
from .common import Struct, options
from .plot_environment import ENV_DEFAULTS, plot_environment
from .rfid_module import defaults, batch_read_prob, batch_scan_prob
from .game_hover_solver import (generate_candidates, generate_scan_candidates,
                               prune_candidates, solve_game, coverage_probability)

_last_stage1 = None


def main_stage1_hover_planner(*args, **kwargs):
    global _last_stage1
    opts = options(dict(ENV_DEFAULTS, CloseEnvironmentFigure=True, CommonParamOverrides={}, EnableFlyByRFID=True,
                        GameParams={}, RFIDOverrides={}, Verbose=True), args, kwargs)
    opts = options(opts, kwargs=opts.CommonParamOverrides)
    start = perf_counter()
    def report(message):
        if opts.Verbose:
            print(message, flush=True)
    report('Stage 1: generating the port environment...')
    env = {k: opts[k] for k in ENV_DEFAULTS}
    env['Visualize'] = opts.Visualize and not opts.CloseEnvironmentFigure
    containers, tags, scene = plot_environment(**env)
    rfid = defaults()
    rfid.update(opts.RFIDOverrides)
    pos = np.column_stack((tags.x, tags.y, tags.z))
    scan_segments, service_paths = None, None
    if opts.EnableFlyByRFID:
        candidates, scan_segments, service_paths = generate_scan_candidates(scene, tags, rfid)
        P, prob_stats = batch_scan_prob(scan_segments, pos, rfid)
        prune_stats = Struct(N_before=len(candidates), N_after=len(candidates), n_zero=0,
            n_dominated=0, n_merged=0, cov_before=1., cov_after=1., cov_delta=0., prune_time=0.)
    else:
        candidates = generate_candidates(scene, containers, tags, rfid)
        P, prob_stats = batch_read_prob(candidates, pos, rfid)
    report(f'  Containers={len(containers)}, tags={len(tags.x)}, candidates={len(candidates)}')
    reachable = np.asarray(P.max(axis=0).toarray()).ravel() > .01 if P.shape[0] else np.zeros(len(tags.x), bool)
    if not opts.EnableFlyByRFID:
        report(f'  Probability matrix: {P.shape}, nonzeros={P.nnz}; pruning...')
        P, candidates, prune_stats = prune_candidates(P, candidates)
    if not len(candidates):
        raise ValueError('All candidates have zero coverage; check the scene and RFID parameters')
    report(f'  Retained candidates={len(candidates)}; solving the coverage game...')
    game_params = Struct(target_coverage=1.0, max_players=250, n_restarts=12,
        max_iter_brd=150, min_marginal_cov=.001, verbose=opts.Verbose, seed=opts.TagSeed)
    game_params.update(opts.GameParams)
    from .legacy_planner import LegacyRandom
    rng = LegacyRandom(game_params.seed)
    # The source's two benchmark_gain_time_ calls consume Ntag uniforms each.
    rng.random.rand(2 * len(tags.x))
    game_params['_rng'] = rng
    hover, ids, cov_rate, history = solve_game(P, candidates, game_params, len(tags.x))
    selected_segments = scan_segments[ids] if scan_segments is not None else np.empty((0, 9))
    selected_paths = [service_paths[i] for i in ids] if service_paths is not None else []
    cov = coverage_probability(P, ids)
    bins = Struct(high=int((cov >= .8).sum()), mid=int(((cov >= .5) & (cov < .8)).sum()),
                  low=int(((cov > 0) & (cov < .5)).sum()), zero=int((cov == 0).sum()))
    stage1 = Struct(Containers=containers, Tags=tags, scene=scene, rfid=rfid,
        Ntag=len(pos), num_tags=len(pos), scene_name=scene.name, fixed_scene_layout=opts.FixedSceneLayout,
        tag_seed=opts.TagSeed, tag_distribution_mode=opts.TagDistributionMode, tag_cluster_level=opts.TagClusterLevel,
        layout_seed=opts.LayoutSeed, num_effective_targets=int(reachable.sum()),
        num_hover_points=0 if opts.EnableFlyByRFID else len(ids), num_scan_tasks=len(ids),
        flyByRFID=bool(opts.EnableFlyByRFID), scanSegments=selected_segments, scanServicePaths=selected_paths,
        tag_positions=pos, target_ids=np.flatnonzero(reachable),
        tagPos=pos, tagNormal=tags.normal, tag_z_unique=np.unique(np.round(pos[:, 2], 2)),
        candidates=candidates, Ncand=len(candidates), P=P,
        prob_stats=prob_stats, prune_stats=prune_stats, hoverPoints=hover,
        hoverIdx=ids, coverageRate=cov_rate, cov_prob=cov, history=history, game_params=game_params,
        Khover=len(ids), coverageBins=bins, stage1Time=perf_counter() - start)
    stage1['_rng'] = rng
    for name in ('irregularity_level', 'position_jitter', 'height_jitter', 'fill_jitter', 'sparse_bay_prob'):
        stage1['stack_' + name] = scene['stack_' + name]
    distances = np.linalg.norm(hover[:, :3] - [75, 10, 5], axis=1)
    stage1.scenarioInfo = Struct(scenario_name=scene.name, num_tags=len(pos), Khover=len(ids),
        num_hover_points=stage1.num_hover_points, num_scan_tasks=len(ids), candidate_hover_count=len(candidates), coverageRate=cov_rate,
        mean_distance_to_depot=float(distances.mean()), estimated_total_task_distance=float(distances.sum()), notes='')
    if opts.Visualize:
        from .visualization import visualize_stage1
        visualize_stage1(stage1)
    _last_stage1 = stage1
    report(f'Stage 1 completed: scan tasks={len(ids)}, actual hovers={stage1.num_hover_points}, '
           f'coverage={cov_rate:.2%}, elapsed={stage1.stage1Time:.2f}s')
    return stage1


if __name__ == '__main__':
    main_stage1_hover_planner()
    import matplotlib.pyplot as plt
    plt.show()
