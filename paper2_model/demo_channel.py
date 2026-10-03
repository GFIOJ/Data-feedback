"""Two channel demonstrations; neither chooses routes or schedules."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from paper1.paper2_dataset import Paper2Dataset
from paper2_model.channel import ChannelConfig, ChannelModel, ChannelSnapshot, evaluate_schedule


def deterministic_demo():
    """Known gains make every signal/interference/rate value hand-checkable."""
    config = ChannelConfig(transmit_power_w=1., total_bandwidth_hz=4., num_subchannels=2,
                           noise_psd_w_per_hz=1., shadowing_std_db=0., valid_distance_min_m=.1)
    size, base = 5, 4
    gain = np.ones((size, size))
    gain[0, base] = gain[base, 0] = 2.
    gain[1, base] = gain[base, 1] = 1.
    gain[1, 2] = gain[2, 1] = 3.
    gain[0, 2] = gain[2, 0] = .5
    snapshot = ChannelSnapshot(0, base, config, np.zeros((size, 3)), np.ones(size, bool),
        np.ones((size, size)), np.zeros((size, size), np.int8), np.zeros((size, size)),
        np.zeros((size, size)), np.ones((size, size)), np.ones((size, size), complex), gain)
    result = evaluate_schedule(snapshot, [(0, base, 0), (1, 2, 0)])
    expected_sinr = 2 / (2 + 1)
    expected_rate = 2 * np.log2(1 + expected_sinr)
    assert np.isclose(result['links'][0]['sinr'], expected_sinr)
    assert np.isclose(result['links'][0]['rate_bps'], expected_rate)
    print('Deterministic demo: b=2 Hz, N0*b=2 W')
    print(f'link 0->{base}: signal=2 W, interference=1 W, SINR={expected_sinr:.6f}, '
          f'rate={expected_rate:.6f} bit/s')
    return result


def yard_demo(input_folder=None):
    """Evaluate legal same-channel and orthogonal arrangements on one cached yard snapshot."""
    folder = Path(input_folder or Path(__file__).resolve().parents[1] /
                  'paper2_baselines' / 'direct_return' / 'input' /
                  'input_ref_5to10mbit_v10')
    data = Paper2Dataset(folder)
    model = ChannelModel(data, ChannelConfig())
    eligible = data.node_states['scheduling_eligible']
    slot = int(np.flatnonzero(eligible.sum(axis=1) >= 3)[0])
    snapshot = model.snapshot(slot)
    u0, u1, u2 = np.flatnonzero(snapshot.scheduling_eligible[:model.num_uavs])[:3]
    base = model.base_station_id
    same_schedule = [(int(u0), base, 0), (int(u1), int(u2), 0)]
    other_schedule = [(int(u0), base, 0), (int(u1), int(u2), 1)]
    same = evaluate_schedule(snapshot, same_schedule)
    other = evaluate_schedule(snapshot, other_schedule)
    target_same, target_other = same['links'][0], other['links'][0]
    print(f'Yard demo: zero-based slot={slot}, base_station_id={base}, eligible_uavs={int(eligible[slot].sum())}')
    for tx, rx, channel in same_schedule:
        state = 'NLoS' if snapshot.obstruction_state[tx, rx] else 'LoS'
        print(f'link {tx}->{rx}, f={channel}: d={snapshot.distance_m[tx,rx]:.3f} m, {state}, '
              f'L={snapshot.path_loss_db[tx,rx]:.3f} dB, g={snapshot.power_gain[tx,rx]:.6e}')
    print(f'same-channel target: I={target_same["interference_power_w"]:.6e} W, '
          f'SINR={target_same["sinr"]:.6e}, R={target_same["rate_bps"]:.3f} bit/s')
    print(f'other-channel target: I={target_other["interference_power_w"]:.6e} W, '
          f'SINR={target_other["sinr"]:.6e}, R={target_other["rate_bps"]:.3f} bit/s')
    assert model.snapshot(slot) is snapshot
    assert target_same['rate_bps'] < target_other['rate_bps']
    return snapshot, same, other


if __name__ == '__main__':
    deterministic_demo()
    yard_demo()
