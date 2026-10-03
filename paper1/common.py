"""Shared data structures and geometry. All Python indices are zero based."""
from __future__ import annotations

import numpy as np


class Struct(dict):
    """Dictionary with convenient attribute-style field access."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def options(defaults, args=(), kwargs=None):
    if len(args) % 2:
        raise TypeError("Options must be name/value pairs")
    supplied = dict(zip(args[::2], args[1::2]))
    supplied.update(kwargs or {})
    lookup = {k.lower().replace('_', ''): k for k in defaults}
    out = Struct(defaults)
    for key, value in supplied.items():
        canonical = lookup.get(key.lower().replace('_', ''))
        if canonical is None:
            raise TypeError(f"Unknown option: {key}")
        out[canonical] = value
    return out


def bounds(containers, inflate=0):
    groups = {}
    for c in containers:
        key = tuple(round(float(c[k]), 9) for k in ('x', 'y', 'width', 'length'))
        groups.setdefault(key, []).append((float(c.z), float(c.z + c.height)))
    boxes = []
    for (x, y, width, length), intervals in groups.items():
        intervals.sort()
        start, end = intervals[0]
        for low, high in intervals[1:]:
            if low <= end + 1e-9:
                end = max(end, high)
            else:
                boxes.append((x, y, start, width, length, end - start))
                start, end = low, high
        boxes.append((x, y, start, width, length, end - start))
    boxes = np.asarray(boxes, float).reshape(-1, 6)
    lo, size = boxes[:, :3], boxes[:, 3:]
    return lo - inflate, lo + size + inflate


def segment_hits(a, b, lo, hi):
    """Vectorized closed-segment/AABB slab intersection (including tangencies)."""
    a, b = np.asarray(a), np.asarray(b)
    if not len(lo):
        return np.zeros(0, bool)
    d = b - a
    enter, leave = np.zeros(len(lo)), np.ones(len(lo))
    hit = np.ones(len(lo), bool)
    for axis in range(3):
        if abs(d[axis]) < 1e-12:
            hit &= (a[axis] >= lo[:, axis]) & (a[axis] <= hi[:, axis])
        else:
            t1, t2 = (lo[:, axis] - a[axis]) / d[axis], (hi[:, axis] - a[axis]) / d[axis]
            enter = np.maximum(enter, np.minimum(t1, t2))
            leave = np.minimum(leave, np.maximum(t1, t2))
    return hit & (enter <= leave)


def round_half_away(x, decimals=0):
    scale = 10.0 ** decimals
    x = np.asarray(x) * scale
    return np.copysign(np.floor(np.abs(x) + .5), x) / scale


def positive(value, name, integer=False, allow_zero=False):
    if not np.isscalar(value) or not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} and finite")
    if integer and int(value) != value:
        raise ValueError(f"{name} must be an integer")
    return int(value) if integer else float(value)
