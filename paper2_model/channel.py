"""Per-slot channel snapshots based only on the formulas in paper section 2.1."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paper1.common import segment_hits


LIGHT_SPEED_MPS = 299_792_458.0


def line_is_obstructed(start, end, obstacle_min_m, obstacle_max_m):
    """Closed AABB convention: crossing or merely touching an obstacle is NLoS."""
    return bool(segment_hits(np.asarray(start, float), np.asarray(end, float),
                             np.asarray(obstacle_min_m, float),
                             np.asarray(obstacle_max_m, float)).any())


def sample_rician_coefficient(rng, k_linear, phase_rad, size=None):
    """Normalized Rician coefficient; K=0 is normalized Rayleigh fading."""
    if k_linear < 0:
        raise ValueError('k_linear must be nonnegative')
    w = (rng.normal(size=size) + 1j * rng.normal(size=size)) / np.sqrt(2)
    return (np.sqrt(k_linear / (k_linear + 1)) * np.exp(1j * phase_rad) +
            np.sqrt(1 / (k_linear + 1)) * w)


@dataclass(frozen=True)
class ChannelConfig:
    """Uncalibrated debug defaults; transmit_power_w is one fixed P for every UAV and slot."""
    carrier_frequency_hz: float = 2.4e9
    reference_distance_m: float = 1.0
    reference_path_loss_db: float = 40.0
    path_loss_exponent: float = 2.2
    obstruction_loss_db: float = 20.0
    shadowing_std_db: float = 4.0
    k_los: float = 4.0
    k_los_unit: str = 'linear'
    transmit_power_w: float = 0.1
    total_bandwidth_hz: float = 10e6
    num_subchannels: int = 4
    noise_psd_w_per_hz: float = 4e-21
    random_seed: int = 2026
    valid_distance_min_m: float = 1.0
    valid_distance_max_m: float = 5000.0

    def validated(self):
        positive = ('carrier_frequency_hz', 'reference_distance_m', 'path_loss_exponent',
                    'transmit_power_w', 'total_bandwidth_hz', 'noise_psd_w_per_hz',
                    'valid_distance_min_m', 'valid_distance_max_m')
        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.valid_distance_max_m <= self.valid_distance_min_m:
            raise ValueError('valid_distance_max_m must exceed valid_distance_min_m')
        if not all(np.isfinite(value) for value in
                   (self.reference_path_loss_db, self.obstruction_loss_db,
                    self.shadowing_std_db, self.k_los)):
            raise ValueError('path-loss, shadowing and K parameters must be finite')
        if self.obstruction_loss_db < 0 or self.shadowing_std_db < 0:
            raise ValueError('loss and shadowing standard deviation must be nonnegative')
        if int(self.num_subchannels) != self.num_subchannels or self.num_subchannels <= 0:
            raise ValueError('num_subchannels must be a positive integer')
        if self.k_los_unit not in ('linear', 'dB'):
            raise ValueError("k_los_unit must be 'linear' or 'dB'")
        if self.k_los_unit == 'linear' and self.k_los < 0:
            raise ValueError('linear k_los must be nonnegative')
        if int(self.random_seed) != self.random_seed or self.random_seed < 0:
            raise ValueError('random_seed must be a nonnegative integer')
        return self

    @property
    def k_los_linear(self):
        return float(self.k_los if self.k_los_unit == 'linear' else 10 ** (self.k_los / 10))

    @property
    def subchannel_bandwidth_hz(self):
        return self.total_bandwidth_hz / self.num_subchannels


@dataclass(frozen=True)
class ChannelSnapshot:
    """Matrices are (M+1,M+1); base station node id is M; unavailable entries are NaN/-1."""
    slot_index: int
    base_station_id: int
    config: ChannelConfig
    positions_m: np.ndarray
    scheduling_eligible: np.ndarray
    distance_m: np.ndarray
    obstruction_state: np.ndarray
    path_loss_db: np.ndarray
    shadowing_db: np.ndarray
    large_scale_gain: np.ndarray
    small_scale_coefficient: np.ndarray
    power_gain: np.ndarray

    def __post_init__(self):
        # A frozen dataclass does not freeze ndarray contents. Copy and seal every array.
        for name in ('positions_m', 'scheduling_eligible', 'distance_m', 'obstruction_state',
                     'path_loss_db', 'shadowing_db', 'large_scale_gain',
                     'small_scale_coefficient', 'power_gain'):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


class ChannelModel:
    """Create and cache one reciprocal block-fading snapshot per slot."""
    def __init__(self, dataset, config=None):
        self.dataset = dataset
        self.config = (config or ChannelConfig()).validated()
        self.num_uavs = int(dataset.scenario['num_uavs'])
        self.base_station_id = self.num_uavs
        self.obstacle_min_m = np.asarray(dataset.environment['obstacle_min_m'], float)
        self.obstacle_max_m = np.asarray(dataset.environment['obstacle_max_m'], float)
        self.base_station_position_m = np.asarray(dataset.environment['base_station_position_m'], float)
        self._cache = {}

    def snapshot(self, slot_index):
        slot = int(slot_index)
        if slot != slot_index or not 0 <= slot < self.dataset.scenario['slot_count']:
            raise IndexError('slot_index is outside the exported zero-based slot range')
        if slot not in self._cache:
            self._cache[slot] = self._build_snapshot(slot)
        return self._cache[slot]

    def clear_cache(self):
        self._cache.clear()

    def _build_snapshot(self, slot):
        states = self.dataset.node_states
        uav_positions = np.asarray(states['positions_at_slot_midpoint_m'][slot], float)
        uav_eligible = np.asarray(states['scheduling_eligible'][slot], bool)
        positions = np.vstack((uav_positions, self.base_station_position_m))
        eligible = np.r_[uav_eligible, True]
        if np.isnan(positions[eligible]).any():
            raise ValueError('eligible node has a NaN midpoint position')
        size = self.num_uavs + 1
        distance = np.full((size, size), np.nan)
        state = np.full((size, size), -1, np.int8)
        loss = np.full((size, size), np.nan)
        shadow = np.full((size, size), np.nan)
        beta = np.full((size, size), np.nan)
        h = np.full((size, size), np.nan + 1j * np.nan, complex)
        gain = np.full((size, size), np.nan)
        rng = np.random.default_rng(np.random.SeedSequence([self.config.random_seed, slot]))
        active = np.flatnonzero(eligible)
        for offset, i in enumerate(active):
            for j in active[offset + 1:]:
                d = float(np.linalg.norm(positions[i] - positions[j]))
                if d == 0:
                    raise ValueError(f'nodes {i} and {j} overlap in slot {slot}')
                if not self.config.valid_distance_min_m <= d <= self.config.valid_distance_max_m:
                    raise ValueError(f'distance {d:.3f} m for nodes {i},{j} is outside the configured model range')
                blocked = line_is_obstructed(positions[i], positions[j],
                                             self.obstacle_min_m, self.obstacle_max_m)
                x_db = float(rng.normal(0., self.config.shadowing_std_db))
                path_loss = (self.config.reference_path_loss_db +
                    10 * self.config.path_loss_exponent * np.log10(d / self.config.reference_distance_m) +
                    int(blocked) * self.config.obstruction_loss_db + x_db)
                large = 10 ** (-path_loss / 10)
                k = 0. if blocked else self.config.k_los_linear
                phase = np.remainder(2 * np.pi * self.config.carrier_frequency_hz * d / LIGHT_SPEED_MPS,
                                     2 * np.pi)
                small = sample_rician_coefficient(rng, k, phase)
                total = large * abs(small) ** 2
                for a, b in ((i, j), (j, i)):
                    distance[a, b], state[a, b], loss[a, b] = d, int(blocked), path_loss
                    shadow[a, b], beta[a, b], h[a, b], gain[a, b] = x_db, large, small, total
        return ChannelSnapshot(slot, self.base_station_id, self.config, positions, eligible, distance, state,
                               loss, shadow, beta, h, gain)


def evaluate_schedule(snapshot, schedule, config=None, candidate_links=None):
    """Evaluate a user-supplied [(tx,rx,subchannel), ...] on one fixed snapshot."""
    config = (config or snapshot.config).validated()
    if config != snapshot.config:
        raise ValueError('schedule evaluation must use the configuration stored in the snapshot')
    links = [tuple(item) for item in schedule]
    size, base, channels = len(snapshot.positions_m), snapshot.base_station_id, config.num_subchannels
    candidates = None if candidate_links is None else {tuple(map(int, link)) for link in candidate_links}
    seen_links, used_uavs, base_channels = set(), set(), set()
    normalized = []
    for raw in links:
        if len(raw) != 3:
            raise ValueError('each scheduled link must be (transmitter, receiver, subchannel)')
        tx, rx, channel = raw
        if any(int(value) != value for value in raw):
            raise ValueError('node and subchannel identifiers must be integers')
        tx, rx, channel = int(tx), int(rx), int(channel)
        if not 0 <= tx < size or not 0 <= rx < size:
            raise ValueError('scheduled node identifier is out of range')
        if not 0 <= channel < channels:
            raise ValueError('subchannel identifier is out of range')
        if tx == rx:
            raise ValueError('self-links are not allowed')
        if tx == base:
            raise ValueError('the base station is receive-only')
        if not snapshot.scheduling_eligible[tx] or not snapshot.scheduling_eligible[rx]:
            raise ValueError('offline or partial-slot UAVs cannot be scheduled')
        if (tx, rx) in seen_links:
            raise ValueError('duplicate scheduled link')
        if candidates is not None and (tx, rx) not in candidates:
            raise ValueError('scheduled link is not in candidate_links')
        involved = {node for node in (tx, rx) if node != base}
        if involved & used_uavs:
            raise ValueError('a UAV may participate in at most one link and one subchannel per slot')
        if rx == base and channel in base_channels:
            raise ValueError('at most one base-station reception is allowed per subchannel')
        seen_links.add((tx, rx)); used_uavs.update(involved)
        if rx == base:
            base_channels.add(channel)
        normalized.append((tx, rx, channel))

    per_channel_rate = np.zeros((size, size, channels), float)
    total_rate = np.zeros((size, size), float)
    details = []
    bandwidth = config.subchannel_bandwidth_hz
    noise = config.noise_psd_w_per_hz * bandwidth
    for tx, rx, channel in normalized:
        signal = config.transmit_power_w * snapshot.power_gain[tx, rx]
        interference = sum(config.transmit_power_w * snapshot.power_gain[other_tx, rx]
                           for other_tx, _, other_channel in normalized
                           if other_channel == channel and other_tx != tx)
        sinr = signal / (noise + interference)
        rate = bandwidth * np.log2(1 + sinr)
        per_channel_rate[tx, rx, channel] = rate
        total_rate[tx, rx] += rate
        details.append(dict(transmitter=tx, receiver=rx, subchannel=channel,
            signal_power_w=float(signal), interference_power_w=float(interference),
            noise_power_w=float(noise), sinr=float(sinr), rate_bps=float(rate)))
    return dict(links=details, per_subchannel_rate_bps=per_channel_rate,
                link_total_rate_bps=total_rate)
