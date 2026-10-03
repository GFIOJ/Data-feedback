"""Legacy-compatible stage-2 algorithms.

This module retains the original voxel endpoints, interval-based collision
definition, first-improvement neighborhoods and time-axis semantics.
It is separate from the continuous-geometry implementation to keep the two
models explicit and avoid comparing different objective inputs.
"""
from __future__ import annotations
from itertools import product
import math
import numpy as np
from .common import Struct, bounds


class LegacyRandom:
    """MT19937 uniform stream and full randperm implemented as sorted uniforms."""
    def __init__(self, seed=1):
        self.random = np.random.RandomState(5489 if seed == 0 else int(seed))

    def permutation(self, n):
        return np.argsort(self.random.rand(n), kind='stable')

    def rand(self):
        return self.random.rand()

    def choice(self, n, size, replace=False, p=None):
        if replace:
            raise ValueError('Only sampling without replacement is supported')
        if p is None:
            return self.permutation(n)[:size]
        weights=np.array(p,float,copy=True)
        selected=[]
        for _ in range(size):
            total=weights.sum()
            if total<1e-12:
                remaining=np.setdiff1d(np.arange(n),selected)
                selected.extend(remaining[self.permutation(len(remaining))][:size-len(selected)])
                break
            index=min(n-1,int(np.searchsorted(np.cumsum(weights)/total,self.rand(),side='left')))
            selected.append(index)
            weights[index]=0
        return np.array(selected,int)


class LegacyRouteLibrary:
    def __init__(self, nodes, containers, scene, pm):
        self.nodes, self.pm = nodes, pm
        self.origin = np.array([scene.x_range[0], scene.y_range[0], scene.z_range[0]])
        self.upper = np.array([scene.x_range[1], scene.y_range[1], scene.z_range[1]])
        self.dims = np.ceil((self.upper - self.origin) / pm.GridRes).astype(int)
        self.occ = np.zeros(tuple(self.dims), bool)
        lo, hi = bounds(containers)
        inflation = int(np.ceil(pm.InflateRadius / pm.GridRes))
        for a, b in zip(lo, hi):
            lower = np.maximum(0, np.floor((a - self.origin) / pm.GridRes).astype(int) - inflation)
            upper = np.minimum(self.dims - 1, np.floor((b - self.origin) / pm.GridRes).astype(int) + inflation)
            if np.all(lower <= upper):
                self.occ[tuple(slice(x, y + 1) for x, y in zip(lower, upper))] = True
        self.info = Struct(origin=self.origin, dims=self.dims, res=pm.GridRes,
                           xr=scene.x_range, yr=scene.y_range, zr=scene.z_range)
        self.cache, self.records, self.edges = {}, {}, {}
        self.astar_calls = 0
        self.directions = ([(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
            if pm.ConnectivityMode == 6 else [d for d in product((-1,0,1), repeat=3) if any(d)])
        self.costs = np.sqrt(np.sum(np.array(self.directions)**2, axis=1))
        self.edge_offsets = {}
        for d,c in zip(self.directions,self.costs):
            ns=max(2,int(np.ceil(c*pm.GridRes/max(pm.GridRes*.5,.2)))+1)
            self.edge_offsets[d]=list(dict.fromkeys(tuple(int(np.floor(.5+k/ns*v)) for v in d) for k in range(ns+1)))

    def grid(self, p):
        return tuple(np.clip(np.floor((np.asarray(p) - self.origin) / self.pm.GridRes).astype(int), 0, self.dims - 1))

    def world(self, g):
        return self.origin + (np.asarray(g) + .5) * self.pm.GridRes

    def ensure_free(self, g):
        if not self.occ[g]:
            return g
        for radius in range(1, int(max(self.dims)) + 1):
            for d in product(range(-radius, radius + 1), repeat=3):
                if all(abs(v) < radius for v in d):
                    continue
                p = tuple(g[k] + d[k] for k in range(3))
                if all(0 <= p[k] < self.dims[k] for k in range(3)) and not self.occ[p]:
                    return p
        return g

    def edge_free(self, a, b):
        # Sampled-voxel check used by the original planner.
        key = (a, b)
        if key not in self.edges:
            d=tuple(b[k]-a[k] for k in range(3))
            self.edges[key]=all(not self.occ[tuple(a[k]+offset[k] for k in range(3))]
                               for offset in self.edge_offsets[d])
        return self.edges[key]

    def astar(self, start, end):
        # Preserve the original unsorted min/replace-last removal. Heap
        # tie-breaking changes equal-cost routes and subsequent reservations.
        shape = tuple(self.dims)
        count = int(np.prod(self.dims))
        def linear(g):
            return g[0] + self.dims[0] * (g[1] + self.dims[1] * g[2])
        def cell(i):
            z, xy = divmod(int(i), int(self.dims[0] * self.dims[1]))
            y, x = divmod(xy, int(self.dims[0]))
            return x, y, z
        source, target = linear(start), linear(end)
        scores, parent, closed = np.full(count, np.inf), np.full(count, -1, int), np.zeros(count, bool)
        keys, values = np.empty(max(count, 16)), np.empty(max(count, 16), int)
        keys[0], values[0], size = np.linalg.norm(np.subtract(end, start)), source, 1
        scores[source] = 0.
        while size:
            index = int(np.argmin(keys[:size]))
            current = values[index]
            size -= 1
            keys[index], values[index] = keys[size], values[size]
            if current == target:
                result = [cell(current)]
                while parent[current] >= 0:
                    current = parent[current]
                    result.append(cell(current))
                return list(reversed(result))
            if closed[current]:
                continue
            closed[current] = True
            g = cell(current)
            for d, cost in zip(self.directions, self.costs):
                h = tuple(g[k] + d[k] for k in range(3))
                if any(h[k] < 0 or h[k] >= shape[k] for k in range(3)) or self.occ[h]:
                    continue
                nxt = linear(h)
                if closed[nxt] or (len(self.directions) == 26 and not self.edge_free(g, h)):
                    continue
                new = scores[current] + cost
                if new < scores[nxt]:
                    scores[nxt], parent[nxt] = new, current
                    if size == len(keys):
                        keys = np.r_[keys, np.empty(len(keys))]
                        values = np.r_[values, np.empty(len(values), int)]
                    keys[size], values[size] = new + math.sqrt(sum((end[k]-h[k])**2 for k in range(3))), nxt
                    size += 1
        return []

    def path(self, i, j):
        if (i, j) in self.cache:
            return self.cache[i, j]
        if i == j:
            p = np.vstack((self.nodes[i], self.nodes[j]))
        else:
            self.astar_calls += 1
            start, end = self.ensure_free(self.grid(self.nodes[i])), self.ensure_free(self.grid(self.nodes[j]))
            cells = self.astar(start, end)
            if not cells:
                z = min(self.dims[2] - 1, max(start[2], end[2]) + 5)
                a, b = self.ensure_free((start[0], start[1], z)), self.ensure_free((end[0], end[1], z))
                legs = [self.astar(start, a), self.astar(a, b), self.astar(b, end)]
                if all(legs):
                    cells = legs[0] + legs[1][1:] + legs[2][1:]
            p = np.array([self.world(g) for g in cells]).reshape(-1, 3)
        self.cache[i, j], self.cache[j, i] = p, p[::-1].copy()
        return p

    def distance(self, i, j):
        path = self.path(i, j)
        return sum(float(np.linalg.norm(b - a)) for a, b in zip(path[:-1], path[1:])) if len(path) else np.inf


def cost(seq, state, exact=False):
    distance = 0.
    for i, j in zip(seq[:-1], seq[1:]):
        distance += state.library.distance(i, j) if exact else state.distMat[i, j]
    return distance / state.pm.UAVSpeed + (len(seq) - 2) * state.dwellT


def objective(costs, pm):
    return max(costs, default=0.) + pm.Lambda * sum(costs) if np.all(np.isfinite(costs)) else np.inf


def insert(seq, node, state):
    best, output = np.inf, None
    for position in range(1, len(seq)):
        trial = seq[:position] + [node] + seq[position:]
        score = cost(trial, state)
        if score < best:
            best, output = score, trial
    return output


def rebuild(assign, state):
    seq = [0, 0]
    for node in assign:
        seq = insert(seq, node, state)
    return seq


def task_assignment(state):
    pm, n, k = state.pm, state.numUAVs, state.Khover
    rng = state.get('rng', LegacyRandom(pm.Seed))
    state.rng = rng
    best_obj = np.inf
    accepted = Struct(accepted_transfer_count=0, accepted_exchange_count=0, accepted_total_count=0)
    for restart in range(pm.GameRestarts):
        assign, seqs, costs = [[] for _ in range(n)], [[0, 0] for _ in range(n)], np.zeros(n)
        if restart == 0:
            remaining = list(range(1, k + 1))
            while remaining:
                best = np.inf
                info = None
                for node in remaining:
                    for u in range(n):
                        if len(assign[u]) >= pm.MaxTasksPerUAV:
                            continue
                        for position in range(1, len(seqs[u])):
                            seq = seqs[u][:position] + [node] + seqs[u][position:]
                            value = cost(seq, state)
                            trial = costs.copy()
                            trial[u] = value
                            score = objective(trial, pm)
                            if score < best:
                                best, info = score, (node, u, seq, value)
                if info is None:
                    raise RuntimeError('Incomplete greedy assignment')
                node, u, seqs[u], costs[u] = info
                assign[u].append(node)
                remaining.remove(node)
        else:
            for index, node in enumerate(rng.permutation(k) + 1):
                u = index % n
                assign[u].append(int(node))
                seqs[u] = insert(seqs[u], int(node), state)
                costs[u] = cost(seqs[u], state)
        for _ in range(pm.GameMaxIter):
            changed = False
            for u in rng.permutation(n):
                current = objective(costs, pm)
                moves = []
                light, heavy = int(np.argmin(costs)), int(np.argmax(costs))
                if pm.EnableTaskTransfer:
                    if len(assign[u]) > 1 and light != u and len(assign[light]) < pm.MaxTasksPerUAV:
                        moves.extend(('transfer', u, light, node, None) for node in assign[u])
                    if heavy != u and len(assign[heavy]) > 1 and len(assign[u]) < pm.MaxTasksPerUAV:
                        moves.extend(('transfer', heavy, u, node, None) for node in assign[heavy])
                found = False
                def attempt(kind, source, target, node, other):
                    aa = [a.copy() for a in assign]
                    ss = [s.copy() for s in seqs]
                    if kind == 'transfer':
                        aa[source].remove(node)
                        aa[target].append(node)
                        ss[source] = rebuild(aa[source], state)
                        ss[target] = insert(seqs[target], node, state)
                    else:
                        aa[source][aa[source].index(node)] = other
                        aa[target][aa[target].index(other)] = node
                        ss[source], ss[target] = rebuild(aa[source], state), rebuild(aa[target], state)
                    cc = costs.copy()
                    cc[source], cc[target] = cost(ss[source], state), cost(ss[target], state)
                    return (aa, ss, cc) if objective(cc, pm) < current - 1e-9 else None
                for move in moves:
                    answer = attempt(*move)
                    if answer is not None:
                        assign, seqs, costs = answer
                        accepted.accepted_transfer_count += 1
                        found = True
                        break
                if not found and pm.EnableTaskExchange:
                    for v in rng.permutation(n):
                        if v == u or not assign[u] or not assign[v]:
                            continue
                        for node in assign[u]:
                            for other in assign[v]:
                                answer = attempt('swap', u, v, node, other)
                                if answer is not None:
                                    assign, seqs, costs = answer
                                    accepted.accepted_exchange_count += 1
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                changed |= found
                accepted.accepted_total_count += int(found)
            if not changed:
                break
        score = objective(costs, pm)
        if score < best_obj:
            best_obj, best = score, (assign, seqs, costs)
    state.bestAssign = [np.array(a, int) - 1 for a in best[0]]
    state.bestSeqs, state.bestUavCosts = best[1:]
    state.gameTaskAssignmentStats = accepted
    state.preLocalSeqs = [s.copy() for s in state.bestSeqs]
    if pm.Verbose:
        print(f'  Assignment game: makespan={max(state.bestUavCosts):.3f}, SOC={sum(state.bestUavCosts):.3f}; building exact route costs...',flush=True)
    return local_improve(state)


def local_candidates(seqs, exact_costs, state):
    pm, dist = state.pm, state.distMat
    critical = int(np.argmax(exact_costs))
    def picks(u, extra):
        seq = seqs[u]
        positions = list(range(1, len(seq) - 1))
        if not positions:
            return []
        marginal = [dist[seq[p-1],seq[p]] + dist[seq[p],seq[p+1]] - dist[seq[p-1],seq[p+1]] for p in positions]
        order = np.argsort(-np.array(marginal), kind='stable')
        keep = order[:pm.RiskPointsPerUAV].tolist()
        if extra and u == critical and len(order) > len(keep):
            keep = sorted(set(keep + [int(order[len(keep)])]))
        other = [node for v, s in enumerate(seqs) if v != u for node in s[1:-1]]
        if other:
            boundary = int(np.argmin([min(dist[seq[p], other]) for p in positions]))
            keep = sorted(set(keep + [boundary]))
        return [positions[i] for i in keep]
    if pm.EnableRelocate:
        for u in range(len(seqs)):
            for p in picks(u, True):
                node = seqs[u][p]
                for v in range(len(seqs)):
                    if v == u or len(seqs[v]) - 2 >= pm.MaxTasksPerUAV:
                        continue
                    seq = seqs[v]
                    increments = [dist[a,node] + dist[node,b] - dist[a,b] for a,b in zip(seq[:-1],seq[1:])]
                    q = int(np.argmin(increments)) + 1
                    trial = [s.copy() for s in seqs]
                    trial[u].pop(p)
                    trial[v].insert(q, node)
                    yield 'relocate', trial, (u,v)
    if pm.EnableSwap:
        picked = [picks(u, False) for u in range(len(seqs))]
        for u in range(len(seqs) - 1):
            for v in range(u + 1, len(seqs)):
                for p in picked[u]:
                    for q in picked[v]:
                        trial = [s.copy() for s in seqs]
                        trial[u][p], trial[v][q] = trial[v][q], trial[u][p]
                        yield 'swap', trial, (u,v)


def local_improve(state):
    pm = state.pm
    stats = Struct(totalExactRefineCalls=0, acceptedMoves=0, history=[])
    state.localImproveStats = stats
    if not pm.EnableLocalImprove:
        return state
    seqs, approx = state.bestSeqs, state.bestUavCosts
    exact = np.array([cost(s,state,True) for s in seqs])
    current = objective(exact, pm)
    if pm.Verbose:
        print(f'  Exact initial routes: makespan={max(exact):.3f}, SOC={sum(exact):.3f}',flush=True)
    for iteration in range(pm.MaxImproveIters):
        candidates = []
        for kind, trial, affected in local_candidates(seqs, exact, state):
            cc = approx.copy()
            for u in affected:
                cc[u] = cost(trial[u],state)
            score = objective(cc,pm)
            if np.isfinite(score):
                candidates.append((score,kind,trial,affected))
        candidates.sort(key=lambda c:c[0])
        best, change = current, None
        exhausted = False
        for _,kind,trial,affected in candidates[:pm.MaxCandidatePerIter][:pm.TopKExactRefine]:
            cc = exact.copy()
            for u in sorted(affected):
                if stats.totalExactRefineCalls >= pm.MaxExactRefineCalls:
                    exhausted = True
                    break
                cc[u] = cost(trial[u],state,True)
                stats.totalExactRefineCalls += 1
            if exhausted:
                break
            score = objective(cc,pm)
            if score < best - 1e-9:
                best,change = score,(kind,trial,cc)
        if change is None:
            break
        kind,seqs,exact = change
        current = best
        approx = np.array([cost(s,state) for s in seqs])
        stats.acceptedMoves += 1
        stats.history.append(Struct(iteration=iteration,type=kind,objective=best))
        if pm.Verbose:
            print(f'  Local improvement {iteration+1}: {kind}, objective={best:.3f}',flush=True)
        if exhausted or stats.totalExactRefineCalls >= pm.MaxExactRefineCalls:
            break
    state.bestSeqs, state.bestUavCosts = seqs, approx
    state.bestAssign = [np.array(s[1:-1],int)-1 for s in seqs]
    return state


def optimize_2opt(state):
    costs = np.array([cost(s,state) for s in state.bestSeqs])
    if state.pm.EnableSeqAdjust:
        for u in np.argsort(-costs,kind='stable'):
            seq = state.bestSeqs[u]
            if len(seq) <= 4:
                continue
            for _ in range(30):
                improved = False
                for i in range(1,len(seq)-2):
                    for j in range(i+1,len(seq)-1):
                        trial = seq[:i]+seq[i:j+1][::-1]+seq[j+1:]
                        value = cost(trial,state)
                        if value < costs[u] - .05:
                            seq,costs[u],improved = trial,value,True
                            break
                    if improved:
                        break
                if not improved:
                    break
            state.bestSeqs[u] = seq
    state.bestUavCosts = costs
    state.bestAssign = [np.array(s[1:-1],int)-1 for s in state.bestSeqs]
    return state


def empty_reservations():
    return {'node': {}, 'edge': {}, 'hover': {}}


def add_interval(table, kind, key, start, end, owner):
    table[kind].setdefault(key, []).append((float(start), float(end), owner))


def overlap(table, kind, key, start, end, eps):
    for a,b,owner in table[kind].get(key, ()):
        if a < end-eps and start < b-eps:
            return owner,max(a,start)
    return None


def segment_window(table, voxels, fly_time, earliest, state):
    pm = state.pm
    guard = pm.ReservationNodeGuard
    guard = max(.2,.25*pm.SafetyRadius/pm.UAVSpeed) if guard is None else guard
    eps = pm.ReservationTimeEps
    offsets = np.arange(len(voxels)) * (fly_time/max(len(voxels)-1,1))
    start, first = earliest, None
    while start <= earliest+pm.ReservationMaxDelay+eps:
        next_start, detail, hit_any = np.inf, None, False
        checks = [('node',v,offset-guard,offset+guard) for v,offset in zip(voxels,offsets)]
        checks += [('edge',tuple(sorted((a,b))),offsets[i],offsets[i+1]) for i,(a,b) in enumerate(zip(voxels[:-1],voxels[1:]))]
        for kind,key,lo,hi in checks:
            for a,b,owner in table[kind].get(key, ()):
                if a < start+hi-eps and start+lo < b-eps:
                    hit_any = True
                    candidate = b-lo+eps
                    if candidate > start+eps and candidate < next_start:
                        next_start = candidate
                        detail = Struct(blockerUAV=owner,type=kind,time=max(start+lo,a),key=key)
        if not hit_any:
            return start, start+offsets, first
        if first is None and detail is not None:
            first = detail
        start = start+pm.ReservationWaitStep if not np.isfinite(next_start) else max(start+eps,next_start)
    return None,None,first


def simulate(seq,state,global_table,uav):
    pm,lib = state.pm,state.library
    local = empty_reservations()
    guard = max(.2,.25*pm.SafetyRadius/pm.UAVSpeed) if pm.ReservationNodeGuard is None else pm.ReservationNodeGuard
    eps = pm.ReservationTimeEps
    path,services,legs,events = np.empty((0,4)),[],[],[]
    t = 0.
    def failed():
        return Struct(timedPath=path,finishTime=np.inf,isFeasible=False,resvLocal=local,
            serviceWindows=np.asarray(services).reshape(-1,3),legInfo=legs,uavId=uav,
            lowLevelStats=Struct(waitEvents=events,totalWaitSec=sum(e.waitSec for e in events),waitResolvedCount=len(events)))
    for si,(source,target) in enumerate(zip(seq[:-1],seq[1:])):
        if si>0 and source>0:
            voxel = lib.grid(state.allNodes[source])
            earliest,service_start,first = t,t,None
            while True:
                end = service_start+state.dwellT
                h = overlap(global_table,'hover',source,service_start,end,eps)
                v = overlap(global_table,'node',voxel,service_start,end,eps)
                if h is None and v is None:
                    break
                if first is None:
                    owner,stamp = h if h is not None else v
                    first=Struct(blockerUAV=owner,type='hover_hover' if h else 'hover_node',time=stamp)
                service_start += pm.ReservationWaitStep
                if service_start-earliest > pm.ReservationMaxDelay:
                    return failed()
            if first is not None:
                events.append(Struct(uav=uav,blockerUAV=first.blockerUAV,type=first.type,time=first.time,
                    waitSec=service_start-earliest,fromNode=source,toNode=source,taskNode=source))
            if len(path):
                path[-1,3]=end
            else:
                path=np.array([np.r_[state.allNodes[source],end]])
            services.append([source,service_start,end])
            add_interval(local,'hover',source,service_start,end,uav)
            add_interval(local,'node',voxel,service_start,end,uav)
            t=end
        spatial=lib.path(source,target)
        distance=lib.distance(source,target)
        if not len(spatial) or not np.isfinite(distance):
            return failed()
        voxels=[]
        for point in spatial:
            v=lib.grid(point)
            if not voxels or v!=voxels[-1]:
                voxels.append(v)
        earliest=t
        departure,times,first=segment_window(global_table,voxels,distance/pm.UAVSpeed,t,state)
        if departure is None:
            return failed()
        if first is not None and departure-t>eps:
            events.append(Struct(uav=uav,blockerUAV=first.blockerUAV,type=first.type,time=first.time,
                waitSec=departure-t,fromNode=source,toNode=target,taskNode=target if target else source))
        if len(path) and departure>t:
            path[-1,3]=departure
        stamps=[departure]
        for a,b in zip(spatial[:-1],spatial[1:]):
            stamps.append(stamps[-1]+float(np.linalg.norm(b-a))/pm.UAVSpeed)
        timed=np.column_stack((spatial,stamps))
        path=timed if not len(path) else np.vstack((path,timed[1:]))
        for v,stamp in zip(voxels,times):
            add_interval(local,'node',v,stamp-guard,stamp+guard,uav)
        for i,(a,b) in enumerate(zip(voxels[:-1],voxels[1:])):
            add_interval(local,'edge',tuple(sorted((a,b))),times[i],times[i+1],uav)
        legs.append(Struct(fromNode=source,toNode=target,startTime=departure,endTime=stamps[-1],waitApplied=departure-earliest))
        t=stamps[-1]
    if not len(path):
        path=np.array([np.r_[state.allNodes[seq[0]],0.]])
    return Struct(timedPath=path,finishTime=t,isFeasible=True,resvLocal=local,
        serviceWindows=np.asarray(services).reshape(-1,3),legInfo=legs,uavId=uav,
        lowLevelStats=Struct(waitEvents=events,totalWaitSec=sum(e.waitSec for e in events),waitResolvedCount=len(events)))


def reservation_conflict(a,b,eps):
    for kind in ('hover','edge','node'):
        # Preserve lexically ordered string keys.
        def key_text(key):
            if kind=='hover':
                return f'H{key+1}'
            if kind=='node':
                return '_'.join(str(v+1) for v in key)
            return str(key)
        for key in sorted(a[kind],key=key_text):
            for start,end,_ in a[kind][key]:
                hit=overlap(b,kind,key,start,end,eps)
                if hit is not None:
                    return Struct(type=kind,key=key,time=hit[1])
    return None


def plan(seqs,state):
    from .uav_mission_planner import priority_order
    n=state.numUAVs
    base=list(np.argsort(-np.array([cost(s,state) for s in seqs]),kind='stable'))
    def with_graph(graph):
        order=priority_order(graph,base)
        if order is None:
            return None
        table=empty_reservations()
        sims=[None]*n
        for u in order:
            sims[u]=simulate(seqs[u],state,table,u)
            if not sims[u].isFeasible:
                return None
            for kind in table:
                for key,ivs in sims[u].resvLocal[kind].items():
                    table[kind].setdefault(key,[]).extend(ivs)
        return sims
    root=np.zeros((n,n),bool)
    sims=with_graph(root)
    stats=Struct(expanded=0,branched=0,solved=False,num_conflicts_resolved=0)
    if sims is None:
        return None,stats
    queue=[(objective([s.finishTime for s in sims],state.pm),0,root,sims)]
    best=sims
    while queue and stats.expanded<state.pm.PBSMaxNodes:
        index=min(range(len(queue)),key=lambda i:queue[i][0])
        score,depth,graph,sims=queue.pop(index)
        stats.expanded+=1
        conflict=None
        for u in range(n-1):
            for v in range(u+1,n):
                if reservation_conflict(sims[u].resvLocal,sims[v].resvLocal,state.pm.ReservationTimeEps):
                    conflict=(u,v)
                    break
            if conflict:
                break
        if conflict is None:
            stats.solved=True
            return sims,stats
        stats.num_conflicts_resolved+=1
        for high,low in (conflict,conflict[::-1]):
            child=graph.copy()
            child[high,low]=True
            trial=with_graph(child)
            if trial is not None:
                values=[s.finishTime for s in trial]
                child_score=objective(values,state.pm)+1e-6*(depth+1)
                queue.append((child_score,depth+1,child,trial))
                stats.branched+=1
    return best,stats


def feedback_candidates(seqs,timeline,state):
    unique={}
    for sim in timeline:
        for event in sim.lowLevelStats.waitEvents:
            key=(event.uav,event.blockerUAV,event.taskNode,event.type)
            if key not in unique or event.waitSec>unique[key].waitSec:
                unique[key]=event
    events=sorted(unique.values(),key=lambda e:-e.waitSec)
    seen=set()
    def nearest_task(sim,t):
        for node,start,end in sim.serviceWindows:
            if start<=t<=end:
                return int(node)
        best,node=np.inf,0
        for leg in sim.legInfo:
            d=max(leg.startTime-t,t-leg.endTime,0.)
            task=leg.toNode if leg.toNode else leg.fromNode
            if task>0 and d<best:
                best,node=d,task
        return node
    candidates=[]
    for event in events:
        u,v=event.uav,event.blockerUAV
        if u==v or not 0<=v<len(seqs):
            continue
        task_u=event.taskNode
        if task_u<=0 or task_u not in seqs[u]:
            task_u=event.toNode if event.toNode>0 and event.toNode in seqs[u] else event.fromNode
        task_v=nearest_task(timeline[v],event.time)
        if task_v<=0 and len(seqs[v])>2:
            task_v=seqs[v][1]
        moves=[]
        if state.pm.EnableTaskTransfer:
            for source,target,node in ((u,v,task_u),(v,u,task_v)):
                if node>0 and node in seqs[source]:
                    p=seqs[source].index(node)
                    s=seqs[target]
                    dist=state.distMat
                    q=int(np.argmin([dist[a,node]+dist[node,b]-dist[a,b] for a,b in zip(s[:-1],s[1:])]))+1
                    moves.append(('relocate',source,target,p,q,node))
        if state.pm.EnableTaskExchange and task_u>0 and task_v>0 and task_u!=task_v:
            moves.append(('swap',u,v,seqs[u].index(task_u),seqs[v].index(task_v),0))
        if state.pm.EnableSeqAdjust and task_u>0 and task_u in seqs[u]:
            p=seqs[u].index(task_u)
            for q in sorted(set((max(1,p-1),min(len(seqs[u])-2,p+1)))):
                if q!=p:
                    moves.append(('sequence_reinsert',u,u,p,q,task_u))
        for move in moves:
            if move in seen:
                continue
            seen.add(move)
            kind,source,target,p,q,node=move
            trial=[s.copy() for s in seqs]
            if kind=='swap':
                trial[source][p],trial[target][q]=trial[target][q],trial[source][p]
            else:
                trial[source].pop(p)
                trial[target].insert(q,node)
            if max(len(s)-2 for s in trial)<=state.pm.MaxTasksPerUAV:
                candidates.append((kind,trial))
    return events,candidates


def resolve_conflict(state):
    from .uav_mission_planner import PlanningError
    seqs=[s.copy() for s in state.bestSeqs]
    timeline,stats=plan(seqs,state)
    if timeline is None:
        raise PlanningError('Legacy reservation planner found no feasible timeline')
    costs=np.array([s.finishTime for s in timeline])
    audit=Struct(feedback_round_count=0,feedback_trigger_count=0,feedback_candidates_generated=0,
        task_layer_replan_count=0,feedback_accept_count=0,current_objective_trace=[objective(costs,state.pm)])
    if state.pm.EnableConflictFeedback:
        for _ in range(state.pm.MaxImproveIters):
            events,candidates=feedback_candidates(seqs,timeline,state)
            if not events:
                break
            audit.feedback_round_count+=1
            audit.feedback_trigger_count+=len(events)
            audit.feedback_candidates_generated+=len(candidates)
            best,change=objective(costs,state.pm),None
            for kind,trial in candidates:
                ts,ps=plan(trial,state)
                audit.task_layer_replan_count+=1
                if ts is None or not ps.solved:
                    continue
                cs=np.array([s.finishTime for s in ts])
                score=objective(cs,state.pm)
                if score<best-1e-9:
                    best,change=score,(trial,ts,ps,cs)
            if change is None:
                break
            seqs,timeline,stats,costs=change
            audit.feedback_accept_count+=1
            audit.current_objective_trace.append(best)
            if state.pm.Verbose:
                print(f'  Conflict feedback {audit.feedback_round_count}: objective={best:.3f}',flush=True)
    stats.feedbackAudit=audit
    state.bestSeqs,state.timelinePerUAV,state.pbsStats=seqs,timeline,stats
    state.paths,state.uavCosts=[s.timedPath for s in timeline],costs
    state.bestAssign=[np.array(s[1:-1],int)-1 for s in seqs]
    events=[e for s in timeline for e in s.lowLevelStats.waitEvents]
    state.lowLevelStats=Struct(waitEvents=events,totalWaitSec=sum(e.waitSec for e in events),
        waitResolvedCount=len(events),lowLevelInfeasibleCount=0)
    return state


def validate(solution,state):
    reasons=[]
    tasks=np.concatenate(solution.assignment)
    counts=np.bincount(tasks,minlength=state.Khover)
    missing=int((counts==0).sum())
    duplicated=int((counts>1).sum())
    if missing or duplicated or len(tasks)!=state.Khover:
        reasons.append('Task visit mismatch')
    if any(len(a)>state.pm.MaxTasksPerUAV for a in solution.assignment):
        reasons.append('Task capacity violation')
    violations=0
    for path in solution.paths:
        hit=False
        for a,b in zip(path[:-1,:3],path[1:,:3]):
            ns=max(2,int(np.ceil(np.linalg.norm(b-a)/max(state.pm.GridRes*.5,.2)))+1)
            if any(state.library.occ[state.library.grid(a+(i/ns)*(b-a))] for i in range(ns+1)):
                hit=True
                break
        violations+=int(hit)
    conflicts=0
    for u in range(state.numUAVs-1):
        for v in range(u+1,state.numUAVs):
            conflicts+=int(reservation_conflict(solution.timelinePerUAV[u].resvLocal,
                solution.timelinePerUAV[v].resvLocal,state.pm.ReservationTimeEps) is not None)
    if violations:
        reasons.append('Occupied-voxel path violation')
    if conflicts:
        reasons.append('Node/edge/hover reservation conflicts')
    if not state.pbsStats.solved:
        reasons.append('PBS search did not solve all conflicts')
    for sim,value in zip(solution.timelinePerUAV,solution.uavCosts):
        if not sim.isFeasible or not np.isfinite(value) or abs(sim.finishTime-value)>1e-5:
            reasons.append('Infeasible or inconsistent timeline')
    return Struct(passed=not reasons,reasons=reasons,numMissingTargets=missing,
        numDuplicatedTargets=duplicated,numObstacleViolations=violations,numConflicts=conflicts,
        objectiveCalc=objective(solution.uavCosts,state.pm),
        note='Legacy voxel/reservation validation, not continuous-space physical validation')
