"""Candidate generation and minimum RFID set cover."""
from __future__ import annotations
from time import perf_counter
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import Bounds, LinearConstraint, milp
from .common import Struct, round_half_away, positive


def generate_candidates(scene, containers, tags, rfid):
    """Generate the finite exact candidate set for planar circular coverage."""
    if not containers or not len(tags.x):
        return np.empty((0, 6))
    radius = float(rfid.maxReadRange)
    half_angle = np.deg2rad(float(rfid.maxScanAngleDeg) / 2)
    offset = max(.1, radius * max(0., np.cos(half_angle)))
    footprint = radius * np.sin(min(half_angle, np.pi / 2))
    candidates = []
    for sign in (-1, 1):
        faces = np.unique(round_half_away(tags.y[tags.normal[:, 1] * sign > 0], 2))
        for face in faces:
            ids = (tags.normal[:, 1] * sign > 0) & (round_half_away(tags.y, 2) == face)
            points = np.unique(round_half_away(np.column_stack((tags.x[ids], tags.z[ids])), 8), axis=0)
            centers = list(points)
            for i, point in enumerate(points):
                delta = points[i + 1:] - point
                distances = np.linalg.norm(delta, axis=1)
                for vector, distance in zip(delta[(distances > 1e-9) & (distances <= 2 * footprint + 1e-9)],
                                            distances[(distances > 1e-9) & (distances <= 2 * footprint + 1e-9)]):
                    middle = point + vector / 2
                    height = np.sqrt(max(0., footprint ** 2 - distance ** 2 / 4))
                    perpendicular = np.array([-vector[1], vector[0]]) / distance
                    centers.extend((middle + height * perpendicular, middle - height * perpendicular))
            for x, z in centers:
                y = face + sign * offset
                if scene.x_range[0] <= x <= scene.x_range[1] and scene.y_range[0] <= y <= scene.y_range[1] and .3 <= z <= scene.z_range[1]:
                    candidates.append([x, y, z, 0., -sign, 0.])
    if not candidates:
        return np.empty((0, 6))
    return np.unique(round_half_away(np.asarray(candidates), 6), axis=0)


def generate_scan_candidates(scene, tags, rfid):
    """Create continuous aisle scans; split a face only when its height exceeds one cone footprint."""
    radius = float(rfid.maxReadRange)
    half_angle = np.deg2rad(float(rfid.maxScanAngleDeg) / 2)
    offset = max(.1, radius * max(0., np.cos(half_angle)))
    footprint = radius * np.sin(min(half_angle, np.pi / 2))
    max_length = positive(rfid.get('maxScanSegmentLength', 250.), 'maxScanSegmentLength')
    poses, segments, paths = [], [], []
    for sign in (-1, 1):
        for face in np.unique(round_half_away(tags.y[tags.normal[:, 1] * sign > 0], 2)):
            ids = np.flatnonzero((tags.normal[:, 1] * sign > 0) & (round_half_away(tags.y, 2) == face))
            remaining = ids[np.argsort(tags.z[ids], kind='stable')].tolist()
            while remaining:
                low = tags.z[remaining[0]]
                band = [i for i in remaining if tags.z[i] <= low + 2 * footprint + 1e-9]
                remaining = [i for i in remaining if i not in set(band)]
                ordered = sorted(band, key=lambda i: tags.x[i])
                chunks = [[ordered[0]]]
                for index in ordered[1:]:
                    if tags.x[index] - tags.x[chunks[-1][-1]] > 2 * footprint or tags.x[index] - tags.x[chunks[-1][0]] > max_length:
                        chunks.append([])
                    chunks[-1].append(index)
                for chunk in chunks:
                    x0, x1 = float(tags.x[chunk].min()), float(tags.x[chunk].max())
                    z = float((tags.z[chunk].min() + tags.z[chunk].max()) / 2)
                    y = float(face + sign * offset)
                    start, end = np.array([x0, y, z]), np.array([x1, y, z])
                    if len(poses) % 2:
                        start, end = end, start
                    axis = np.array([0., -sign, 0.])
                    poses.append(np.r_[start, axis])
                    segments.append(np.r_[start, end, axis])
                    paths.append(np.vstack((start, end)))
    return np.asarray(poses), np.asarray(segments), paths


def coverage_probability(P, selected):
    if len(selected) == 0:
        return np.zeros(P.shape[1])
    rows = P[selected].toarray() if hasattr(P, 'toarray') else np.asarray(P)[selected]
    return 1 - np.prod(1 - rows, axis=0)


def prune_candidates(P, candidates):
    """Source thresholds: zero .01, dominance active .01, merge grid .35m."""
    start = perf_counter()
    P = csr_matrix(P)
    before = P.shape[0]
    def union(matrix):
        logs = matrix.copy()
        logs.data = np.log(np.maximum(1 - logs.data, 1e-15))
        return float(np.mean(1 - np.exp(np.asarray(logs.sum(0)).ravel()))) if matrix.shape[1] else 0.
    cov_before = union(P)
    row_sum = np.asarray(P.sum(1)).ravel()
    keep = row_sum > .01
    n_zero = int((~keep).sum())
    P, candidates, row_sum = P[keep], candidates[keep], row_sum[keep]
    csc = P.tocsc()
    removed = np.zeros(P.shape[0], bool)
    for i in np.argsort(row_sum, kind='stable'):
        row = P.getrow(i)
        active = row.data > .01
        columns, vals = row.indices[active], row.data[active]
        if not len(columns):
            removed[i] = True
            continue
        potential = None
        for t in np.argsort(-vals):
            c, val = columns[t], vals[t]
            lo, hi = csc.indptr[c:c + 2]
            ids = csc.indices[lo:hi][csc.data[lo:hi] >= val - 1e-9]
            potential = ids if potential is None else np.intersect1d(potential, ids, assume_unique=True)
            if not len(potential):
                break
        possible = potential[(potential != i) & (~removed[potential])]
        if np.any(row_sum[possible] >= row_sum[i] + 1e-6):
            removed[i] = True
    n_dominated = int(removed.sum())
    P, candidates, row_sum = P[~removed], candidates[~removed], row_sum[~removed]
    groups = {}
    for i, c in enumerate(candidates):
        key = tuple(np.r_[round_half_away(c[:3] / .35), c[3:]])
        previous = groups.get(key)
        if previous is None or row_sum[i] > row_sum[previous]:
            groups[key] = i
    ids = sorted(groups.values())
    n_merged = len(candidates) - len(ids)
    P, candidates = P[ids], candidates[ids]
    cov_after = union(P)
    stats = Struct(N_before=before, N_after=len(ids), n_zero=n_zero, n_dominated=n_dominated,
                   n_merged=n_merged, cov_before=cov_before, cov_after=cov_after,
                   cov_delta=cov_after - cov_before, prune_time=perf_counter() - start)
    return P, candidates, stats


def compress_cover(P, selected, target_coverage=1., max_pair_checks=5000):
    """Delete redundant rows, then try replacing two selected rows with one."""
    P = csr_matrix(P).astype(bool).astype(np.int8)
    selected = list(map(int, selected))
    required = int(np.ceil(P.shape[1] * target_coverage - 1e-12))
    counts = np.asarray(P[selected].sum(axis=0)).ravel() if selected else np.zeros(P.shape[1])
    for row in selected.copy():
        trial = counts - P.getrow(row).toarray().ravel()
        if np.count_nonzero(trial) >= required:
            selected.remove(row)
            counts = trial
    unselected = np.setdiff1d(np.arange(P.shape[0]), selected)
    checks = 0
    for a in range(len(selected)):
        for b in range(a + 1, len(selected)):
            if checks >= max_pair_checks:
                return np.asarray(selected, int)
            checks += 1
            needed = counts - P.getrow(selected[a]).toarray().ravel() - P.getrow(selected[b]).toarray().ravel() <= 0
            if needed.sum() > P[unselected].getnnz(axis=1).max(initial=0):
                continue
            replacement = unselected[np.asarray(P[unselected][:, needed].getnnz(axis=1)).ravel() == needed.sum()]
            if len(replacement):
                old_a, old_b = selected[a], selected[b]
                selected[a] = int(replacement[0])
                selected.pop(b)
                counts += (P.getrow(selected[a]) - P.getrow(old_a) - P.getrow(old_b)).toarray().ravel()
                return compress_cover(P, selected, target_coverage, max_pair_checks - checks)
    return np.asarray(selected, int)


def solve_game(P, candidates, params=None, Ntag_total=None):
    P = csr_matrix(P, dtype=float)
    candidates = np.asarray(candidates, float)
    if P.shape[0] != len(candidates) or np.any(~np.isfinite(P.data)) or np.any((P.data < 0) | (P.data > 1)):
        raise ValueError('P must be a finite probability matrix matching candidates')
    pm = Struct(target_coverage=1.0, max_players=250, n_restarts=12, max_iter_brd=150,
        min_marginal_cov=.001, verbose=True, seed=1, fixed_point_penalty=.10,
        service_time_penalty_weight=.20, path_proxy_penalty_weight=.15, service_time_base=1.,
        milp_time_limit=120.)
    pm.update(params or {})
    for key in ('max_players', 'n_restarts', 'max_iter_brd'):
        positive(pm[key], key, integer=True)
    if not 0 <= pm.target_coverage <= 1:
        raise ValueError('target_coverage must be in [0, 1]')
    n, nt = P.shape
    total = nt if Ntag_total is None else positive(Ntag_total, 'Ntag_total', integer=True)
    history = Struct(K_values=[], coverage=[], phi_values=[], convergence=[], time_per_K=[])
    if not n or not nt:
        return candidates[:0], np.zeros(0, int), 0., history
    if pm.target_coverage == 1 and total == nt and np.all(P.getnnz(axis=0)):
        started = perf_counter()
        result = milp(np.ones(n), integrality=np.ones(n), bounds=Bounds(0, 1),
            constraints=LinearConstraint(P.T, 1, np.inf),
            options={'time_limit': float(pm.milp_time_limit), 'mip_rel_gap': 0.})
        if result.x is not None:
            ids = np.flatnonzero(result.x > .5)
            if not result.success:
                ids = compress_cover(P, ids, 1.)
            coverage = float(coverage_probability(P, ids).sum() / total)
            if coverage == 1.:
                history.update(method='exact_set_cover', optimal=bool(result.success),
                    mip_gap=float(getattr(result, 'mip_gap', np.nan)), solver_message=result.message)
                history.K_values.append(len(ids)); history.coverage.append(coverage)
                history.phi_values.append(-float(len(ids))); history.convergence.append([])
                history.time_per_K.append(perf_counter() - started)
                if pm.verbose:
                    print(f'  Exact set cover: K={len(ids)}, coverage=100.00%, optimal={result.success}', flush=True)
                return candidates[ids], ids, coverage, history
    power = np.asarray(P.sum(1)).ravel()
    z = candidates[:, 2]
    zstd = np.std(z, ddof=1) if n > 1 else 0.
    service = pm.service_time_base + .7 * power / max(power.mean(), 1e-9) + .3 * abs(z - z.mean()) / max(zstd, 1e-9)
    anchor = np.asarray(pm.get('path_proxy_anchor', candidates[:, :3].mean(0)))[:3]
    distance = np.linalg.norm(candidates[:, :3] - anchor, axis=1)
    penalty = pm.fixed_point_penalty + pm.service_time_penalty_weight * service + pm.path_proxy_penalty_weight * distance / max(distance.mean(), 1e-9)
    from .legacy_planner import LegacyRandom
    rng = pm.get('_rng', LegacyRandom(pm.seed))
    solved = {}
    def uncov(selected):
        return 1 - coverage_probability(P, selected)
    def greedy(k, previous=()):
        selected = list(previous[:k])
        remaining = uncov(selected)
        while len(selected) < k:
            gains = np.asarray(P @ remaining).ravel() - penalty
            gains[selected] = -np.inf
            idx = int(np.argmax(gains))
            selected.append(idx)
            remaining *= 1 - P.getrow(idx).toarray().ravel()
        return selected
    def solve_k(k):
        if k in solved:
            return solved[k][1]
        t0 = perf_counter()
        previous_ks = [s for s in solved if s < k]
        previous = solved[max(previous_ks)][0] if previous_ks else []
        restarts = min(pm.n_restarts, 4 if k >= 80 else 6 if k >= 40 else pm.n_restarts)
        best_phi = -np.inf
        for restart in range(restarts):
            if restart == 0 and len(previous):
                selected = greedy(k, previous)
            elif restart == 1:
                selected = greedy(k)
            elif restart == 2:
                weights = power + 1e-15
                selected = rng.choice(n, k, replace=False, p=weights / weights.sum()).tolist()
            else:
                selected = rng.choice(n, k, replace=False).tolist()
            remaining = uncov(selected)
            trace = []
            for _ in range(pm.max_iter_brd):
                trace.append(float((1 - remaining).sum() - penalty[selected].sum()))
                changed = False
                for player in rng.permutation(k):
                    old = selected[player]
                    row = P.getrow(old).toarray().ravel()
                    others = selected[:player] + selected[player + 1:]
                    without = uncov(others) if np.any(row >= 1 - 1e-12) else remaining / (1 - row)
                    gains = np.asarray(P @ without).ravel() - penalty
                    gains[others] = -np.inf
                    idx = int(gains.argmax())
                    if gains[idx] > np.dot(row, without) - penalty[old] + 1e-12:
                        selected[player] = idx
                        remaining = without * (1 - P.getrow(idx).toarray().ravel())
                        changed = True
                if not changed:
                    break
            phi = float((1 - remaining).sum() - penalty[selected].sum())
            trace.append(phi)
            if phi > best_phi:
                best_phi, best_s, best_trace = phi, selected.copy(), trace
        cov = float(coverage_probability(P, best_s).sum() / max(total, 1))
        solved[k] = (best_s, cov, best_phi, best_trace, perf_counter() - t0)
        if pm.verbose:
            print(f'  K={k:3d}, coverage={cov:.2%}, objective={best_phi:.3f}', flush=True)
        return cov
    max_k = min(n, int(pm.max_players))
    low, high = 0, None
    for k in range(1, min(5, max_k) + 1):
        if solve_k(k) >= pm.target_coverage:
            high = k
            break
        low = k
    if high is None and low < max_k:
        probe = min(max_k, max(10, low + 1))
        while True:
            if solve_k(probe) >= pm.target_coverage:
                high = probe
                break
            low = probe
            if probe == max_k:
                break
            probe = min(max_k, probe * 2)
    if high is not None:
        while high - low > 1:
            mid = (low + high) // 2
            if solve_k(mid) >= pm.target_coverage:
                high = mid
            else:
                low = mid
        best = high
    else:
        best = max(sorted(solved), key=lambda k: solved[k][1])
    for k in sorted(solved):
        _, cov, phi, trace, elapsed = solved[k]
        for key, value in zip(history, (k, cov, phi, trace, elapsed)):
            history[key].append(value)
    ids = np.array(solved[best][0], int)
    return candidates[ids], ids, solved[best][1], history


def game_hover_solver(action, *args, **kwargs):
    if action.lower() == 'candidates':
        return generate_candidates(*args, **kwargs)
    if action.lower() == 'solve':
        return solve_game(*args, **kwargs)
    if action.lower() == 'visualize':
        from .visualization import visualize_game
        return visualize_game(*args, **kwargs)
    raise ValueError(f'Unknown game action: {action}')
