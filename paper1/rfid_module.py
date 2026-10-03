"""Binary RFID recognition based only on range and reader scan angle."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from .common import Struct


def defaults():
    return Struct(maxReadRange=8.0, maxScanAngleDeg=120.0, maxScanSegmentLength=250.0, dwellTime=0.30)


def _limits(rfid):
    radius = float(rfid['maxReadRange'])
    angle = float(rfid['maxScanAngleDeg'])
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('maxReadRange must be positive and finite')
    if not np.isfinite(angle) or not 0 < angle <= 360:
        raise ValueError('maxScanAngleDeg must be in (0, 360]')
    return radius, np.cos(np.deg2rad(angle / 2))


def read_prob(reader_pos, boresight, tag_pos, rfid):
    """Return 1.0 when a tag lies inside the reader cone, otherwise 0.0."""
    reader = np.asarray(reader_pos, float)
    direction = np.asarray(tag_pos, float) - reader
    boresight = np.asarray(boresight, float)
    distance = np.linalg.norm(direction)
    axis_norm = np.linalg.norm(boresight)
    radius, cos_half_angle = _limits(rfid)
    if distance > radius or axis_norm < 1e-12:
        return 0.0
    if distance < 1e-12:
        return 1.0
    return float(np.dot(direction / distance, boresight / axis_norm) >= cos_half_angle - 1e-12)


def batch_read_prob(candidates, tag_pos, rfid):
    """Build a sparse binary candidate-by-tag recognition matrix."""
    candidates = np.asarray(candidates, float)
    tags = np.asarray(tag_pos, float)
    if candidates.ndim != 2 or candidates.shape[1] < 6:
        raise ValueError('candidates must be an Nx6 array [x,y,z,bx,by,bz]')
    if tags.ndim != 2 or tags.shape[1] != 3:
        raise ValueError('tag_pos must be an Mx3 array')
    radius, cos_half_angle = _limits(rfid)
    tree = cKDTree(tags)
    rows, cols = [], []
    stats = Struct(totalPairs=len(candidates) * len(tags), distFiltered=0,
                   angleFiltered=0, readable=0)
    for row, candidate in enumerate(candidates):
        axis = candidate[3:6]
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            stats.distFiltered += len(tags)
            continue
        ids = np.asarray(tree.query_ball_point(candidate[:3], radius), dtype=int)
        stats.distFiltered += len(tags) - len(ids)
        if not len(ids):
            continue
        vectors = tags[ids] - candidate[:3]
        distances = np.linalg.norm(vectors, axis=1)
        cosines = np.ones(len(ids))
        nonzero = distances > 1e-12
        cosines[nonzero] = vectors[nonzero] @ (axis / axis_norm) / distances[nonzero]
        visible = cosines >= cos_half_angle - 1e-12
        stats.angleFiltered += int((~visible).sum())
        selected = ids[visible]
        rows.extend([row] * len(selected))
        cols.extend(selected.tolist())
    values = np.ones(len(rows), dtype=float)
    matrix = csr_matrix((values, (rows, cols)), shape=(len(candidates), len(tags)))
    stats.readable = stats.nnz = matrix.nnz
    return matrix, stats


def batch_scan_prob(segments, tag_pos, rfid):
    """Build binary coverage for continuous [start,end,boresight] scan segments."""
    segments, tags = np.asarray(segments, float), np.asarray(tag_pos, float)
    if segments.ndim != 2 or segments.shape[1] != 9 or tags.ndim != 2 or tags.shape[1] != 3:
        raise ValueError('segments must be Nx9 and tag_pos must be Mx3')
    radius, cos_half_angle = _limits(rfid)
    rows, cols = [], []
    for row, segment in enumerate(segments):
        start, end, axis = segment[:3], segment[3:6], segment[6:9]
        line = end - start
        length2 = float(line @ line)
        fraction = np.clip((tags - start) @ line / max(length2, 1e-15), 0., 1.)
        vectors = tags - (start + fraction[:, None] * line)
        distances = np.linalg.norm(vectors, axis=1)
        cosines = np.ones(len(tags))
        nonzero = distances > 1e-12
        cosines[nonzero] = vectors[nonzero] @ (axis / np.linalg.norm(axis)) / distances[nonzero]
        ids = np.flatnonzero((distances <= radius + 1e-12) & (cosines >= cos_half_angle - 1e-12))
        rows.extend([row] * len(ids)); cols.extend(ids.tolist())
    matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(segments), len(tags)))
    return matrix, Struct(totalPairs=len(segments) * len(tags), distFiltered=0,
                          angleFiltered=0, readable=matrix.nnz, nnz=matrix.nnz)


def diagnostic(rfid=None):
    rfid = defaults() if rfid is None else rfid
    cases = np.array([[8, 0, 0], [4, 4 * np.sqrt(3), 0], [0, 8, 0], [8.01, 0, 0]])
    return np.column_stack((cases, [read_prob([0, 0, 0], [1, 0, 0], p, rfid) for p in cases]))


def rfid_module(action, *args, **kwargs):
    dispatch = dict(defaults=defaults, batchreadprob=batch_read_prob, batchscanprob=batch_scan_prob,
                    readprob=read_prob, diagnostic=diagnostic)
    try:
        function = dispatch[action.lower()]
    except KeyError as exc:
        raise ValueError(f'Unknown RFID action: {action}') from exc
    return function(*args, **kwargs)
