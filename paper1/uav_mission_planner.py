"""Multi-UAV task game, cached 3D A*, priority search and timed reservations.

The objective is makespan + Lambda * SOC.
Routes use node 0 for the depot, nodes 1..K for hover points. Assignment and
hover indices use 0..K-1. Timed paths are [x, y, z, seconds].
"""
from __future__ import annotations
import heapq
import itertools
from time import perf_counter
import numpy as np
from scipy.spatial.distance import cdist
from .common import Struct, options, bounds, segment_hits, positive


PLANNER_DEFAULTS = dict(DepotPos=None, UAVSpeed=2., SafetyRadius=3., GridRes=1.,
    InflateRadius=1., Lambda=.05, GameRestarts=8, GameMaxIter=100, PBSMaxNodes=120,
    EnableLocalImprove=True, MaxImproveIters=8, MaxCandidatePerIter=80,
    TopKExactRefine=10, MaxExactRefineCalls=120, EnableRelocate=True, EnableSwap=True,
    EnableTwoOptStar=False, EnableTaskTransfer=True, EnableTaskExchange=True,
    EnableSeqAdjust=True, EnableConflictFeedback=True, EnableLocalSearch=True,
    EnableFeedbackAudit=False, EnableMaxTaskConstraint=True, MaxTaskLoadRatio=1.2,
    RiskPointsPerUAV=3, ReservationWaitStep=.5, ReservationMaxDelay=120.,
    ReservationNodeGuard=None, ReservationTimeEps=1e-9, ConnectivityMode=26,
    Verbose=True, Seed=1, AblationMethod='CF_SIPP_FULL', LegacyCompatibility=True,
    ServicePaths=None)


class PlanningError(RuntimeError):
    """No feasible path/schedule was found within the configured budget."""


class RouteLibrary:
    """Per-planner cache; independent runs cannot overwrite each other's state."""
    def __init__(self, nodes, containers, scene, pm):
        self.nodes, self.pm = nodes, pm
        self.lo, self.hi = bounds(containers, pm.InflateRadius)
        ranges = np.array([scene.x_range, scene.y_range, scene.z_range])
        self.origin, self.upper = ranges[:, 0], ranges[:, 1]
        self.dims = np.ceil((self.upper - self.origin) / pm.GridRes).astype(int)
        self.occ = np.zeros(tuple(self.dims), bool)
        # Mark actual inflated-box intersection with voxel centers. Every edge
        # and every endpoint connector is additionally checked continuously.
        for lo, hi in zip(self.lo, self.hi):
            a = np.maximum(0, np.ceil((lo - self.origin) / pm.GridRes - .5).astype(int))
            b = np.minimum(self.dims - 1, np.floor((hi - self.origin) / pm.GridRes - .5).astype(int))
            if np.all(a <= b):
                self.occ[tuple(slice(l, h + 1) for l, h in zip(a, b))] = True
        for axis in range(3):
            bad = self.origin[axis] + (np.arange(self.dims[axis]) + .5) * pm.GridRes > self.upper[axis]
            selector = [slice(None)] * 3
            selector[axis] = bad
            self.occ[tuple(selector)] = True
        self.directions = [d for d in itertools.product((-1, 0, 1), repeat=3)
                           if any(d) and (pm.ConnectivityMode == 26 or sum(abs(v) for v in d) == 1)]
        self.cache, self.edges, self.connectors = {}, {}, {}
        self.astar_calls = 0
        self.info = Struct(origin=self.origin, dims=self.dims, res=pm.GridRes,
                           xr=scene.x_range, yr=scene.y_range, zr=scene.z_range)

    def world(self, g):
        return self.origin + (np.asarray(g) + .5) * self.pm.GridRes

    def grid(self, p):
        return tuple(np.clip(np.floor((np.asarray(p) - self.origin) / self.pm.GridRes).astype(int), 0, self.dims - 1))

    def free_segment(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        if np.any(a < self.origin) or np.any(b < self.origin) or np.any(a > self.upper) or np.any(b > self.upper):
            return False
        broad = np.all(self.hi >= np.minimum(a, b), axis=1) & np.all(self.lo <= np.maximum(a, b), axis=1)
        return not segment_hits(a, b, self.lo[broad], self.hi[broad]).any()

    def connector(self, node):
        if node in self.connectors:
            return self.connectors[node]
        p = self.nodes[node]
        if not self.free_segment(p, p):
            raise PlanningError(f'Node {node} is inside an inflated obstacle or outside the scene')
        base = np.array(self.grid(p))
        for radius in range(max(self.dims)):
            choices = []
            for d in itertools.product(range(-radius, radius + 1), repeat=3):
                if radius and max(abs(v) for v in d) != radius:
                    continue
                g = base + d
                if np.any(g < 0) or np.any(g >= self.dims) or self.occ[tuple(g)]:
                    continue
                w = self.world(g)
                if self.free_segment(p, w):
                    choices.append((np.linalg.norm(w - p), tuple(g)))
            if choices:
                result = min(choices)[1]
                self.connectors[node] = result
                return result
        raise PlanningError(f'Node {node} has no collision-free connector to the grid')

    def edge_free(self, a, b):
        key = (a, b) if a < b else (b, a)
        if key not in self.edges:
            self.edges[key] = self.free_segment(self.world(a), self.world(b))
        return self.edges[key]

    def path(self, i, j):
        if (i, j) in self.cache:
            return self.cache[i, j]
        a, b = self.nodes[i], self.nodes[j]
        if i == j or np.linalg.norm(a - b) < 1e-12:
            path = a[None].copy()
        else:
            self.astar_calls += 1
            start, goal = self.connector(i), self.connector(j)
            queue = [(np.linalg.norm(np.subtract(start, goal)), 0., start)]
            distances, parent = {start: 0.}, {}
            found = False
            while queue:
                _, cost, current = heapq.heappop(queue)
                if cost > distances.get(current, np.inf) + 1e-12:
                    continue
                if current == goal:
                    found = True
                    break
                for d in self.directions:
                    nxt = tuple(current[k] + d[k] for k in range(3))
                    if any(nxt[k] < 0 or nxt[k] >= self.dims[k] for k in range(3)) or self.occ[nxt]:
                        continue
                    new = cost + np.sqrt(sum(v * v for v in d))
                    if new >= distances.get(nxt, np.inf) - 1e-12 or not self.edge_free(current, nxt):
                        continue
                    distances[nxt], parent[nxt] = new, current
                    h = np.linalg.norm(np.subtract(nxt, goal))
                    heapq.heappush(queue, (new + h, new, nxt))
            if not found:
                raise PlanningError(f'No grid path between nodes {i} and {j}')
            cells = [goal]
            while cells[-1] != start:
                cells.append(parent[cells[-1]])
            path = np.vstack((a, [self.world(g) for g in reversed(cells)], b))
            path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-10]]
        self.cache[i, j], self.cache[j, i] = path, path[::-1].copy()
        return path

    def distance(self, i, j):
        return float(np.linalg.norm(np.diff(self.path(i, j), axis=0), axis=1).sum())


def objective(costs, pm):
    costs = np.asarray(costs)
    return float(costs.max(initial=0) + pm.Lambda * costs.sum()) if np.all(np.isfinite(costs)) else np.inf


def route_cost(seq, state, exact=False):
    if exact:
        distance = sum(np.linalg.norm(np.diff(transition_path(state, i, j), axis=0), axis=1).sum()
                       for i, j in zip(seq[:-1], seq[1:]))
    else:
        distance = state.distMat[seq[:-1], seq[1:]].sum()
    service = state.serviceTimes[np.asarray(seq[1:-1], int) - 1].sum() if len(seq) > 2 else 0.
    return float(distance / state.pm.UAVSpeed + service)


def insert_cheapest(seq, node, dist):
    increments = dist[seq[:-1], node] + dist[node, seq[1:]] - dist[seq[:-1], seq[1:]]
    position = int(np.argmin(increments)) + 1
    return seq[:position] + [node] + seq[position:]


def transition_path(state, source, target):
    source_node = state.Khover + source if source and state.openServicePaths else source
    return state.library.path(source_node, target)


def build_library(hoverPoints, Containers, scene, rfid, numUAVs, *args, **kwargs):
    pm = options(PLANNER_DEFAULTS, args, kwargs)
    n = positive(numUAVs, 'numUAVs', integer=True)
    for key in ('UAVSpeed', 'GridRes', 'ReservationWaitStep', 'ReservationTimeEps'):
        positive(pm[key], key)
    for key in ('SafetyRadius', 'InflateRadius', 'ReservationMaxDelay'):
        positive(pm[key], key, allow_zero=True)
    for key in ('GameRestarts', 'GameMaxIter', 'PBSMaxNodes', 'MaxImproveIters',
                'MaxCandidatePerIter', 'TopKExactRefine', 'MaxExactRefineCalls', 'RiskPointsPerUAV'):
        pm[key] = positive(pm[key], key, integer=True)
    if not np.isfinite(pm.Lambda) or pm.Lambda < 0:
        raise ValueError('Lambda must be finite and nonnegative')
    if not np.isfinite(pm.MaxTaskLoadRatio) or pm.MaxTaskLoadRatio < 1:
        raise ValueError('MaxTaskLoadRatio must be >= 1')
    if pm.ConnectivityMode not in (6, 26):
        raise ValueError('ConnectivityMode must be 6 or 26')
    if not pm.EnableLocalSearch:
        pm.EnableTaskTransfer = pm.EnableTaskExchange = pm.EnableSeqAdjust = pm.EnableTwoOptStar = False
    pm.EnableRelocate = pm.EnableRelocate and pm.EnableTaskTransfer
    pm.EnableSwap = pm.EnableSwap and pm.EnableTaskExchange
    if pm.DepotPos is None:
        pm.DepotPos = [scene.x_range[0] + 5, scene.y_range[1] - 5, 15.]
    depot = np.asarray(pm.DepotPos, float).ravel()[:3]
    hp = np.asarray(hoverPoints, float)
    if depot.shape != (3,) or not np.all(np.isfinite(depot)) or hp.ndim != 2 or hp.shape[1] < 3 or not np.all(np.isfinite(hp)):
        raise ValueError('Expected finite 3D depot and Nx3/Nx6 hover points')
    service_paths = [np.asarray(path, float) for path in (pm.ServicePaths or [])]
    if service_paths and (len(service_paths) != len(hp) or any(path.ndim != 2 or path.shape[1] != 3 or
            len(path) < 2 or not np.all(np.isfinite(path)) or
            not np.allclose(path[0], hp[i, :3])
            for i, path in enumerate(service_paths))):
        raise ValueError('ServicePaths must contain one finite Nx3 path starting at each task')
    open_service_paths = bool(service_paths)
    if open_service_paths:
        pm.LegacyCompatibility = False
    if not service_paths:
        service_paths = [point[:3][None] for point in hp]
    service_times = np.array([np.linalg.norm(np.diff(path, axis=0), axis=1).sum() / pm.UAVSpeed
                              for path in service_paths]) + positive(rfid['dwellTime'], 'dwellTime', allow_zero=True) * np.array([len(path) == 1 for path in service_paths])
    pm.MaxTasksPerUAV = max(1, int(np.ceil(pm.MaxTaskLoadRatio * len(hp) / n))) if pm.EnableMaxTaskConstraint else np.inf
    entries = np.vstack((depot, hp[:, :3]))
    exits = np.vstack((depot, [path[-1] for path in service_paths]))
    nodes = np.vstack((entries, exits[1:])) if open_service_paths else entries
    if pm.LegacyCompatibility:
        from .legacy_planner import LegacyRouteLibrary
        lib = LegacyRouteLibrary(nodes, Containers, scene, pm)
    else:
        lib = RouteLibrary(nodes, Containers, scene, pm)
    return Struct(hoverPoints=hp, Containers=Containers, scene=scene, rfid=rfid,
        numUAVs=n, pm=pm, depot=depot, Khover=len(hp), hpXYZ=hp[:, :3], openServicePaths=open_service_paths,
        dwellT=positive(rfid['dwellTime'], 'dwellTime', allow_zero=True), servicePaths=service_paths,
        serviceTimes=service_times,
        occGrid=lib.occ, gridInfo=lib.info, allNodes=nodes, distMat=cdist(exits, entries),
        pathLib=lib.cache, library=lib, startTime=perf_counter(), astarCalls=0)


def move_candidates(seqs, state, focus=None, include_sequence=False):
    pm, dist = state.pm, state.distMat
    sources = range(len(seqs)) if focus is None else sorted(focus)
    for u in sources:
        for i in range(1, len(seqs[u]) - 1):
            node = seqs[u][i]
            if pm.EnableRelocate:
                for v in range(len(seqs)):
                    if v == u or len(seqs[v]) - 2 >= pm.MaxTasksPerUAV:
                        continue
                    trial = [s.copy() for s in seqs]
                    trial[u].pop(i)
                    trial[v] = insert_cheapest(trial[v], node, dist)
                    yield 'relocate', trial, (u, v)
            if pm.EnableSwap:
                for v in range(u + 1, len(seqs)):
                    for j in range(1, len(seqs[v]) - 1):
                        trial = [s.copy() for s in seqs]
                        trial[u][i], trial[v][j] = trial[v][j], trial[u][i]
                        yield 'swap', trial, (u, v)
            if include_sequence and pm.EnableSeqAdjust:
                for j in range(i + 1, len(seqs[u]) - 1):
                    trial = [s.copy() for s in seqs]
                    trial[u][i:j + 1] = trial[u][i:j + 1][::-1]
                    yield 'sequence', trial, (u,)
            if pm.EnableTwoOptStar:
                for v in range(u + 1, len(seqs)):
                    for j in range(1, len(seqs[v])):
                        trial = [s.copy() for s in seqs]
                        trial[u] = seqs[u][:i] + seqs[v][j:]
                        trial[v] = seqs[v][:j] + seqs[u][i:]
                        if max(len(trial[u]), len(trial[v])) - 2 <= pm.MaxTasksPerUAV:
                            yield 'two_opt_star', trial, (u, v)


def task_assignment(state):
    if state.pm.LegacyCompatibility:
        from .legacy_planner import task_assignment as compatible_assignment
        return compatible_assignment(state)
    pm, n, k = state.pm, state.numUAVs, state.Khover
    rng = np.random.RandomState(int(pm.Seed))
    best_score, best_seqs = np.inf, None
    accepted = Struct(relocate=0, swap=0, two_opt_star=0)
    for restart in range(pm.GameRestarts):
        seqs, costs = [[0, 0] for _ in range(n)], np.zeros(n)
        remaining = list(range(1, k + 1))
        if restart:
            rng.shuffle(remaining)
            for t, node in enumerate(remaining):
                u = t % n
                seqs[u] = insert_cheapest(seqs[u], node, state.distMat)
            costs = np.array([route_cost(s, state) for s in seqs])
        else:
            while remaining:
                best = None
                for node in remaining:
                    for u in range(n):
                        if len(seqs[u]) - 2 >= pm.MaxTasksPerUAV:
                            continue
                        trial = insert_cheapest(seqs[u], node, state.distMat)
                        trial_cost = route_cost(trial, state)
                        new_costs = costs.copy()
                        new_costs[u] = trial_cost
                        score = objective(new_costs, pm)
                        if best is None or score < best[0]:
                            best = (score, node, u, trial, trial_cost)
                if best is None:
                    raise PlanningError('Task capacity prevents a complete assignment')
                _, node, u, seqs[u], costs[u] = best
                remaining.remove(node)
        if pm.EnableLocalSearch:
            for _ in range(pm.GameMaxIter):
                current, change = objective(costs, pm), None
                # Best response over task transfer/exchange neighborhoods.
                for kind, trial, changed in move_candidates(seqs, state):
                    tc = costs.copy()
                    for u in changed:
                        tc[u] = route_cost(trial[u], state)
                    score = objective(tc, pm)
                    if score < current - 1e-9:
                        current, change = score, (kind, trial, tc)
                if change is None:
                    break
                kind, seqs, costs = change
                accepted[kind] += 1
        score = objective(costs, pm)
        if score < best_score:
            best_score, best_seqs, best_costs = score, seqs, costs
    state.bestSeqs, state.bestUavCosts = best_seqs, best_costs
    state.bestAssign = [np.array(s[1:-1], int) - 1 for s in best_seqs]
    state.gameTaskAssignmentStats = accepted
    state.localImproveStats = Struct(exactRefineCalls=0, accepted=0)
    if pm.EnableLocalImprove and pm.EnableLocalSearch and k:
        for _ in range(pm.MaxImproveIters):
            candidates = []
            for counter, (kind, trial, changed) in enumerate(move_candidates(state.bestSeqs, state)):
                costs = np.array([route_cost(s, state) for s in trial])
                candidates.append((objective(costs, pm), counter, trial, changed))
            candidates.sort(key=lambda x: x[:2])
            candidates = candidates[:pm.MaxCandidatePerIter][:pm.TopKExactRefine]
            if not candidates:
                break
            current_costs = np.array([route_cost(s, state, True) for s in state.bestSeqs])
            best_obj, change = objective(current_costs, pm), None
            for _, _, trial, changed in candidates:
                if state.localImproveStats.exactRefineCalls >= pm.MaxExactRefineCalls:
                    break
                state.localImproveStats.exactRefineCalls += 1
                tc = current_costs.copy()
                try:
                    for u in changed:
                        tc[u] = route_cost(trial[u], state, True)
                except PlanningError:
                    continue
                score = objective(tc, pm)
                if score < best_obj - 1e-9:
                    best_obj, change = score, (trial, tc)
            if change is None:
                break
            state.bestSeqs, state.bestUavCosts = change
            state.localImproveStats.accepted += 1
        state.bestAssign = [np.array(s[1:-1], int) - 1 for s in state.bestSeqs]
    return state


def optimize_2opt(state):
    if state.pm.LegacyCompatibility:
        from .legacy_planner import optimize_2opt as compatible_2opt
        return compatible_2opt(state)
    if state.pm.EnableSeqAdjust:
        for u, seq in enumerate(state.bestSeqs):
            best = route_cost(seq, state)
            while True:
                change = None
                for i in range(1, len(seq) - 2):
                    for j in range(i + 1, len(seq) - 1):
                        trial = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                        cost = route_cost(trial, state)
                        if cost < best - 1e-9:
                            change, best = trial, cost
                if change is None:
                    break
                seq = change
            state.bestSeqs[u] = seq
    state.bestUavCosts = np.array([route_cost(s, state) for s in state.bestSeqs])
    state.bestAssign = [np.array(s[1:-1], int) - 1 for s in state.bestSeqs]
    return state


def first_path_conflict(path_a, path_b, separation):
    """Exact closest approach of overlapping linear motion/hover segments.

    Vehicles are active only between their launch and return timestamps.
    A zero SafetyRadius still disallows coincident centers.
    """
    a, b = np.asarray(path_a), np.asarray(path_b)
    if len(a) < 2 or len(b) < 2:
        return None
    threshold = max(float(separation), 1e-7)
    va = np.diff(a[:, :3], axis=0) / np.maximum(np.diff(a[:, 3]), 1e-15)[:, None]
    vb = np.diff(b[:, :3], axis=0) / np.maximum(np.diff(b[:, 3]), 1e-15)[:, None]
    # Search only overlapping time ranges, then solve the relative-motion
    # quadratic analytically; no coarse time sampling can miss a crossing.
    for i in range(len(a) - 1):
        start = max(0, int(np.searchsorted(b[:, 3], a[i, 3], side='left')) - 1)
        end = min(len(b) - 1, int(np.searchsorted(b[:, 3], a[i + 1, 3], side='right')))
        if end <= start:
            continue
        ids = np.arange(start, end)
        lo = np.maximum(a[i, 3], b[ids, 3])
        hi = np.minimum(a[i + 1, 3], b[ids + 1, 3])
        active = hi >= lo - 1e-10
        relative = a[i, :3] + (lo - a[i, 3])[:, None] * va[i] - b[ids, :3] - (lo - b[ids, 3])[:, None] * vb[ids]
        velocity = va[i] - vb[ids]
        speed2 = (velocity * velocity).sum(1)
        closest = np.clip(-(relative * velocity).sum(1) / np.maximum(speed2, 1e-20), 0, np.maximum(0, hi - lo))
        distance = np.linalg.norm(relative + closest[:, None] * velocity, axis=1)
        hit = np.flatnonzero(active & (distance < threshold - 1e-10))
        if len(hit):
            j = hit[0]
            return Struct(time=float(lo[j] + closest[j]), segmentA=i, segmentB=int(ids[j]),
                          distance=float(distance[j]), blockerEnd=float(b[ids[j] + 1, 3]))
    return None


def timed_leg(path, start, speed, dwell=0):
    times = start + np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)) / speed]
    result = np.column_stack((path, times))
    if dwell > 0:
        result = np.vstack((result, np.r_[path[-1], times[-1] + dwell]))
    return result


class Reservations:
    def __init__(self, state):
        self.state, self.nodes, self.edges, self.paths = state, {}, {}, []
        self.guard = state.pm.ReservationNodeGuard
        if self.guard is None:
            self.guard = max(.2, .25 * state.pm.SafetyRadius / state.pm.UAVSpeed)
        positive(self.guard, 'ReservationNodeGuard', allow_zero=True)

    def intervals(self, path):
        """Reserve voxels at actual timestamps and undirected edge intervals."""
        if len(path) < 2:
            return
        lib = self.state.library
        previous = None
        for a, b in zip(path[:-1], path[1:]):
            length = np.linalg.norm(b[:3] - a[:3])
            if length < 1e-10:
                yield 'node', lib.grid(a[:3]), a[3] - self.guard, b[3] + self.guard
                continue
            samples = max(1, int(np.ceil(length / (self.state.pm.GridRes * .4))))
            for t in np.linspace(0, 1, samples + 1):
                p = a[:3] + t * (b[:3] - a[:3])
                stamp = a[3] + t * (b[3] - a[3])
                voxel = lib.grid(p)
                yield 'node', voxel, stamp - self.guard, stamp + self.guard
                if previous is not None and previous[0] != voxel:
                    key = tuple(sorted((previous[0], voxel)))
                    yield 'edge', key, previous[1], stamp
                previous = voxel, stamp

    def conflict(self, path):
        eps = self.state.pm.ReservationTimeEps
        first = None
        for kind, key, start, end in self.intervals(path):
            table = self.nodes if kind == 'node' else self.edges
            for lo, hi, owner in table.get(key, ()):
                if start < hi - eps and end > lo + eps:
                    shift = hi - start + 10 * eps
                    candidate = Struct(blockerUAV=owner, conflictType=kind, conflictTime=max(start, lo), shift=shift)
                    if first is None or candidate.conflictTime < first.conflictTime:
                        first = candidate
        for owner, other in self.paths:
            hit = first_path_conflict(path, other, self.state.pm.SafetyRadius)
            if hit is not None:
                # Advance by the configured time step until the exact continuous
                # collision disappears. Voxel conflicts supply interval-end hints.
                candidate = Struct(blockerUAV=owner, conflictType='separation',
                    conflictTime=hit.time, shift=self.state.pm.ReservationWaitStep)
                if first is None or candidate.conflictTime < first.conflictTime:
                    first = candidate
        return first

    def add(self, path, owner):
        for kind, key, lo, hi in self.intervals(path):
            table = self.nodes if kind == 'node' else self.edges
            table.setdefault(key, []).append((lo, hi, owner))
        self.paths.append((owner, path))


def simulate_route(seq, state, reservations, uav):
    pm, lib = state.pm, state.library
    if len(seq) <= 2:
        return Struct(timedPath=np.empty((0, 4)), finishTime=0., serviceWindows=np.empty((0, 3)),
            legInfo=[], isFeasible=True, uavId=uav, lowLevelStats=Struct(waitEvents=[], totalWaitSec=0., waitResolvedCount=0))
    start_shift = 0.
    first_failure = None
    # If a leg must wait in a location crossed by a higher-priority UAV, restart
    # from a later launch instead of inserting an unsafe stationary wait.
    while start_shift <= pm.ReservationMaxDelay + 1e-9:
        t, chunks, services, legs, events = start_shift, [], [], [], []
        failed = False
        for index, (source, target) in enumerate(zip(seq[:-1], seq[1:])):
            spatial = transition_path(state, source, target)
            service = state.servicePaths[target - 1] if target else spatial[-1:]
            moving_service = target and len(service) > 1
            combined = np.vstack((spatial, service[1:])) if moving_service else spatial
            dwell = state.dwellT if target and not moving_service else 0.
            earliest, departure = t, t
            first = None
            while departure - earliest <= pm.ReservationMaxDelay + 1e-9:
                candidate = timed_leg(combined, departure, pm.UAVSpeed, dwell)
                check = candidate
                if index and departure > earliest + 1e-10:
                    check = np.vstack((np.r_[spatial[0], earliest], candidate))
                hit = reservations.conflict(check)
                if hit is None:
                    break
                if first is None:
                    first = hit
                departure += max(pm.ReservationTimeEps * 10, hit.shift)
            else:
                failed = True
                first_failure = first or first_failure
                break
            if first is not None:
                events.append(Struct(uav=uav, blockerUAV=first.blockerUAV,
                    conflictType=first.conflictType, conflictTime=first.conflictTime,
                    waitSec=departure - earliest, fromNode=source, toNode=target,
                    taskNode=target if target else source))
            if index == 0:
                # Before launch, the UAV is in the depot's off-map staging area.
                chunks.append(candidate)
            else:
                if departure > earliest + 1e-10:
                    chunks.append(np.array([np.r_[spatial[0], departure]]))
                chunks.append(candidate[1:])
            arrival = departure + np.linalg.norm(np.diff(spatial, axis=0), axis=1).sum() / pm.UAVSpeed
            if target:
                services.append([target, arrival, candidate[-1, 3]])
            legs.append(Struct(fromNode=source, toNode=target, startTime=departure,
                              endTime=arrival, waitApplied=departure - earliest))
            t = candidate[-1, 3]
        if not failed:
            path = np.vstack(chunks)
            if start_shift > 0 and first_failure is not None:
                events.append(Struct(uav=uav, blockerUAV=first_failure.blockerUAV,
                    conflictType='launch', conflictTime=first_failure.conflictTime,
                    waitSec=start_shift, fromNode=0, toNode=seq[1], taskNode=seq[1]))
            return Struct(timedPath=path, finishTime=float(t), serviceWindows=np.asarray(services).reshape(-1, 3),
                legInfo=legs, isFeasible=True, uavId=uav, lowLevelStats=Struct(waitEvents=events,
                    totalWaitSec=sum(e.waitSec for e in events), waitResolvedCount=len(events)))
        start_shift += max(pm.ReservationWaitStep, (first_failure.shift if first_failure else 0))
    error = PlanningError(f'UAV {uav}: no safe schedule within ReservationMaxDelay')
    error.blocker = first_failure.blockerUAV if first_failure is not None else None
    error.uav = uav
    raise error


def priority_order(graph, base):
    remaining, order = set(base), []
    while remaining:
        ready = [u for u in base if u in remaining and not any(graph[v, u] for v in remaining)]
        if not ready:
            return None
        u = ready[0]
        order.append(u)
        remaining.remove(u)
    return order


def plan_pbs(seqs, state):
    n = state.numUAVs
    estimates = np.array([route_cost(s, state) for s in seqs])
    base = list(np.argsort(-estimates, kind='stable'))
    queue, seen = [(0, np.zeros((n, n), bool))], set()
    stats = Struct(expanded=0, branched=0, solved=False, num_conflicts_resolved=0)
    last_error = None
    # Like the source's plan_with_priority_graph_, all previously scheduled
    # UAVs reserve space, even without an explicit graph edge.
    while queue and stats.expanded < state.pm.PBSMaxNodes:
        depth, graph = queue.pop(0)
        order = priority_order(graph, base)
        if order is None or tuple(order) in seen:
            continue
        seen.add(tuple(order))
        stats.expanded += 1
        reservations, timeline = Reservations(state), [None] * n
        try:
            for u in order:
                sim = simulate_route(seqs[u], state, reservations, u)
                timeline[u] = sim
                reservations.add(sim.timedPath, u)
            stats.solved = True
            stats.priorityOrder = order
            stats.num_conflicts_resolved = sum(t.lowLevelStats.waitResolvedCount for t in timeline)
            return timeline, stats
        except PlanningError as exc:
            last_error = exc
            low, blocker = getattr(exc, 'uav', None), getattr(exc, 'blocker', None)
            if low is None:
                raise
            # Reverse the blocking priority; also try putting the failed UAV
            # before each earlier UAV. Cyclic branches are rejected.
            targets = ([blocker] if blocker is not None else []) + order[:order.index(low)]
            for high in dict.fromkeys(targets):
                child = graph.copy()
                child[low, high] = True
                if priority_order(child, base) is not None:
                    queue.append((depth + 1, child))
                    stats.branched += 1
    raise PlanningError(f'Priority search exhausted ({stats.expanded} orders): {last_error}')


def resolve_conflict(state):
    if state.pm.LegacyCompatibility:
        from .legacy_planner import resolve_conflict as compatible_conflicts
        return compatible_conflicts(state)
    seqs = [s.copy() for s in state.bestSeqs]
    timeline, stats = plan_pbs(seqs, state)
    costs = np.array([s.finishTime for s in timeline])
    audit = Struct(enabled=state.pm.EnableConflictFeedback, feedback_round_count=0,
        feedback_trigger_count=0, feedback_candidates_generated=0, task_layer_replan_count=0,
        feedback_accept_count=0, feedback_assignment_changed_count=0,
        feedback_sequence_changed_count=0, candidate_events=[],
        current_objective_trace=[objective(costs, state.pm)], stop_reason='disabled_by_ablation')
    if state.pm.EnableConflictFeedback:
        for round_id in range(state.pm.MaxImproveIters):
            events = [e for t in timeline for e in t.lowLevelStats.waitEvents]
            if not events:
                audit.stop_reason = 'no_low_level_conflict_event'
                break
            audit.feedback_round_count += 1
            audit.feedback_trigger_count += len(events)
            focus = {e.uav for e in events} | {e.blockerUAV for e in events}
            candidates = []
            for kind, trial, changed in move_candidates(seqs, state, focus, include_sequence=True):
                fast = objective([route_cost(s, state) for s in trial], state.pm)
                candidates.append((fast, kind, trial))
            candidates.sort(key=lambda c: c[0])
            candidates = candidates[:state.pm.MaxCandidatePerIter][:state.pm.TopKExactRefine]
            audit.feedback_candidates_generated += len(candidates)
            best_obj, change = objective(costs, state.pm), None
            for _, kind, trial in candidates:
                audit.task_layer_replan_count += 1
                try:
                    ts, ps = plan_pbs(trial, state)
                except PlanningError:
                    audit.candidate_events.append(Struct(round=round_id, type=kind, status='infeasible'))
                    continue
                cs = np.array([t.finishTime for t in ts])
                score = objective(cs, state.pm)
                audit.candidate_events.append(Struct(round=round_id, type=kind, objective=score,
                    status='improving' if score < best_obj - 1e-9 else 'not_improving'))
                if score < best_obj - 1e-9:
                    best_obj, change = score, (trial, ts, ps, cs, kind)
            if change is None:
                audit.stop_reason = 'all_feedback_candidates_rejected' if candidates else 'no_enabled_candidate'
                break
            old = [set(s[1:-1]) for s in seqs]
            seqs, timeline, stats, costs, kind = change
            audit.feedback_accept_count += 1
            audit.feedback_sequence_changed_count += 1
            audit.feedback_assignment_changed_count += int(old != [set(s[1:-1]) for s in seqs])
            audit.current_objective_trace.append(best_obj)
        else:
            audit.stop_reason = 'feedback_round_budget_exhausted'
    audit.final_wait_events = [e for t in timeline for e in t.lowLevelStats.waitEvents]
    audit.wait_time_after_feedback = sum(t.lowLevelStats.totalWaitSec for t in timeline)
    stats.feedbackAudit = audit
    state.bestSeqs, state.timelinePerUAV, state.pbsStats = seqs, timeline, stats
    state.paths, state.uavCosts = [t.timedPath for t in timeline], costs
    state.bestAssign = [np.array(s[1:-1], int) - 1 for s in seqs]
    state.lowLevelStats = Struct(waitEvents=audit.final_wait_events,
        totalWaitSec=audit.wait_time_after_feedback, waitResolvedCount=len(audit.final_wait_events),
        lowLevelInfeasibleCount=0)
    return state


def validate_solution(solution, state):
    if state.pm.LegacyCompatibility:
        from .legacy_planner import validate
        return validate(solution, state)
    reasons, obstacle_violations, conflicts = [], 0, []
    assigned = np.concatenate(solution.assignment) if solution.assignment else np.zeros(0, int)
    counts = np.bincount(assigned, minlength=state.Khover) if len(assigned) else np.zeros(state.Khover, int)
    missing, duplicated = np.flatnonzero(counts == 0), np.flatnonzero(counts > 1)
    if len(missing) or len(duplicated) or len(assigned) != state.Khover or len(counts) != state.Khover:
        reasons.append('Missing, duplicate, or invalid task assignments')
    for u, (path, seq, sim) in enumerate(zip(solution.paths, solution.seqPerUAV, solution.timelinePerUAV)):
        if len(seq) - 2 > state.pm.MaxTasksPerUAV:
            reasons.append(f'UAV {u}: task capacity exceeded')
        if not len(path):
            if len(seq) > 2:
                reasons.append(f'UAV {u}: empty task path')
            continue
        if not np.all(np.isfinite(path)) or np.any(np.diff(path[:, 3]) < -1e-10):
            reasons.append(f'UAV {u}: invalid timestamps')
        duration = np.diff(path[:, 3])
        length = np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1)
        if np.any(length > state.pm.UAVSpeed * duration + 1e-7):
            reasons.append(f'UAV {u}: speed violation')
        if not np.allclose(path[0, :3], state.depot) or not np.allclose(path[-1, :3], state.depot):
            reasons.append(f'UAV {u}: incorrect depot endpoints')
        if any(not state.library.free_segment(a[:3], b[:3]) for a, b in zip(path[:-1], path[1:])):
            obstacle_violations += 1
        services = sim.serviceWindows
        if list(services[:, 0].astype(int)) != seq[1:-1]:
            reasons.append(f'UAV {u}: service sequence mismatch')
        for node, start, end in services:
            matches = (np.linalg.norm(path[:, :3] - state.allNodes[int(node)], axis=1) < 1e-7)
            if not np.any(matches & (abs(path[:, 3] - start) < 1e-7)) or end - start < state.serviceTimes[int(node) - 1] - 1e-7:
                reasons.append(f'UAV {u}: missing hover service')
        if abs(path[-1, 3] - solution.uavCosts[u]) > 1e-7:
            reasons.append(f'UAV {u}: completion time mismatch')
    for u in range(state.numUAVs):
        for v in range(u + 1, state.numUAVs):
            hit = first_path_conflict(solution.paths[u], solution.paths[v], state.pm.SafetyRadius)
            if hit:
                conflicts.append(Struct(u=u, v=v, **hit))
    if obstacle_violations:
        reasons.append(f'{obstacle_violations} paths hit obstacles')
    if conflicts:
        reasons.append(f'{len(conflicts)} UAV pairs violate separation')
    if not np.isclose(solution.makespan, max(solution.uavCosts, default=0)) or not np.isclose(solution.soc, sum(solution.uavCosts)):
        reasons.append('Inconsistent makespan/SOC')
    return Struct(passed=not reasons, reasons=reasons, numMissingTargets=len(missing),
        numDuplicatedTargets=len(duplicated), numObstacleViolations=obstacle_violations,
        numConflicts=len(conflicts), firstConflict=conflicts[0] if conflicts else None,
        objectiveCalc=objective(solution.uavCosts, state.pm),
        note='Continuous segment/AABB and UAV separation checks; off-map depot staging')


def evaluate(state):
    costs = state.uavCosts
    solution = Struct(paths=state.paths, assignment=state.bestAssign, seqPerUAV=state.bestSeqs,
        timelinePerUAV=state.timelinePerUAV, uavCosts=costs, makespan=float(max(costs, default=0)),
        soc=float(sum(costs)), numUAVs=state.numUAVs, hoverPoints=state.hoverPoints,
        depotPos=state.depot, occGrid=state.occGrid, gridInfo=state.gridInfo,
        distMat=state.distMat.copy(), pathLib=dict(state.library.cache), objective=objective(costs, state.pm))
    for (i, j), path in state.library.cache.items():
        if state.openServicePaths:
            if i > state.Khover and j <= state.Khover:
                solution.distMat[i - state.Khover, j] = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
            elif i == 0 and 0 < j <= state.Khover:
                solution.distMat[0, j] = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
        elif i < solution.distMat.shape[0] and j < solution.distMat.shape[1]:
            solution.distMat[i, j] = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    solution.validation = validate_solution(solution, state)
    stats = Struct(state.pbsStats)
    stats.update(astarCalls=state.library.astar_calls, criticalUAV=int(np.argmax(costs)),
        alignedObjective=solution.objective, objectiveName='makespan_plus_lambda_soc',
        objectiveMode='cf_sipp', validation=solution.validation, lowLevelStats=state.lowLevelStats,
        cachedPairs=len(state.library.cache) // 2, localImproveStats=state.localImproveStats)
    return solution, solution.makespan, stats


def uav_mission_planner(*args, **kwargs):
    if args and isinstance(args[0], str):
        action, args = args[0].lower(), args[1:]
    else:
        action = 'plan'
    dispatch = dict(build_library=build_library, task_assignment=task_assignment,
        optimize_2opt=optimize_2opt, resolve_conflict=resolve_conflict, evaluate=evaluate)
    if action in dispatch:
        return dispatch[action](*args, **kwargs)
    if action != 'plan':
        raise ValueError(f'Unknown planner action: {action}')
    state = build_library(*args, **kwargs)
    for step in (task_assignment, optimize_2opt, resolve_conflict):
        step(state)
    return evaluate(state)
