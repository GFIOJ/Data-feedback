"""Current communication experiment parameters and small shared validation helpers."""
from __future__ import annotations

import numpy as np

from paper2_model.channel import ChannelConfig
from paper2_model.return_simulation import SimulationConfig


BUFFER_BITS = 1_000_000_000 * 8


def communication_configs():
    channel = ChannelConfig(carrier_frequency_hz=2.4e9, reference_distance_m=1.0,
        reference_path_loss_db=40.0, path_loss_exponent=2.2, obstruction_loss_db=20.0,
        shadowing_std_db=4.0, k_los=4.0, k_los_unit='linear', transmit_power_w=0.1,
        total_bandwidth_hz=20_000_000.0, num_subchannels=4,
        noise_psd_w_per_hz=4e-21, random_seed=2026)
    return channel.validated(), SimulationConfig(buffer_capacity_bits=BUFFER_BITS).validated()


def event_identity(event):
    return {key: value for key, value in event.items() if key not in ('data_bytes', 'data_bits')}


def trajectory_speed_check(dataset, requested_speed):
    moving_speeds = []
    for path in dataset.paths:
        dt = np.diff(path[:, 3])
        distance = np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1)
        if np.any(dt < -1e-12) or np.any((dt <= 1e-12) & (distance > 1e-8)):
            return False, float('nan')
        moving_speeds.extend((distance[dt > 1e-12] / dt[dt > 1e-12]).tolist())
    maximum = max(moving_speeds, default=0.)
    return bool(maximum <= requested_speed + 1e-7), float(maximum)
