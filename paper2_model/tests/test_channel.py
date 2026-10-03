import unittest
import tempfile
from pathlib import Path

import numpy as np

from paper1.common import Struct
from paper2_model.channel import (ChannelConfig, ChannelModel, ChannelSnapshot,
    evaluate_schedule, line_is_obstructed, sample_rician_coefficient)
from paper2_model.capacity_diagnostic import base_station_capacity_upper_bound, candidate_loads


def dataset(positions, eligible=None, obstacles=None, base=(20., 0., 0.)):
    positions = np.asarray(positions, float)[None, :, :]
    count = positions.shape[1]
    eligible = np.ones((1, count), bool) if eligible is None else np.asarray(eligible, bool)[None, :]
    obstacles = obstacles or (np.empty((0, 3)), np.empty((0, 3)))
    return Struct(scenario={'num_uavs': count, 'slot_count': 1},
        environment={'obstacle_min_m': np.asarray(obstacles[0], float).reshape(-1, 3),
                     'obstacle_max_m': np.asarray(obstacles[1], float).reshape(-1, 3),
                     'base_station_position_m': np.asarray(base, float)},
        node_states={'positions_at_slot_midpoint_m': positions,
                     'scheduling_eligible': eligible})


def manual_snapshot(config, eligible=None, base_gains=None, slot=0):
    size, base = 5, 4
    shape = (size, size)
    gain = np.ones(shape)
    gain[0, base] = gain[base, 0] = 2.
    gain[1, base] = gain[base, 1] = 1.
    gain[1, 2] = gain[2, 1] = 3.
    gain[0, 2] = gain[2, 0] = .5
    if base_gains is not None:
        for uav, value in enumerate(base_gains):
            gain[uav, base] = gain[base, uav] = value
    return ChannelSnapshot(slot, base, config, np.zeros((size, 3)),
        np.ones(size, bool) if eligible is None else np.asarray(eligible, bool),
        np.ones(shape), np.zeros(shape, np.int8), np.zeros(shape), np.zeros(shape),
        np.ones(shape), np.ones(shape, complex), gain)


class ChannelTests(unittest.TestCase):
    def test_geometry_los_blocked_above_and_tangent(self):
        low, high = np.array([[4., -1., -1.]]), np.array([[6., 1., 1.]])
        self.assertFalse(line_is_obstructed([0, 2, 0], [10, 2, 0], low, high))
        self.assertTrue(line_is_obstructed([0, 0, 0], [10, 0, 0], low, high))
        self.assertFalse(line_is_obstructed([0, 0, 2], [10, 0, 2], low, high))
        self.assertTrue(line_is_obstructed([0, 1, 0], [10, 1, 0], low, high))

    def test_large_scale_distance_and_obstruction_loss(self):
        config = ChannelConfig(shadowing_std_db=0., valid_distance_min_m=.1)
        clear = ChannelModel(dataset([[5, 0, 0], [10, 0, 0]], base=(30, 0, 0)), config).snapshot(0)
        self.assertGreater(clear.large_scale_gain[1, 2], clear.large_scale_gain[0, 2])
        obstacle = (np.array([[14., -1., -1.]]), np.array([[16., 1., 1.]]))
        blocked = ChannelModel(dataset([[10, 0, 0]], obstacles=obstacle, base=(20, 0, 0)), config).snapshot(0)
        unblocked = ChannelModel(dataset([[10, 0, 0]], base=(20, 0, 0)), config).snapshot(0)
        ratio = blocked.large_scale_gain[0, 1] / unblocked.large_scale_gain[0, 1]
        self.assertAlmostEqual(ratio, 10 ** (-config.obstruction_loss_db / 10), places=12)

    def test_normalized_rician_and_rayleigh_power(self):
        rng = np.random.default_rng(7)
        for k in (0., 4.):
            power = abs(sample_rician_coefficient(rng, k, .3, size=200_000)) ** 2
            self.assertAlmostEqual(power.mean(), 1., delta=.01)
            if k == 0:
                self.assertAlmostEqual(power.var(), 1., delta=.03)

    def test_snr_interference_orthogonality_and_total_rate(self):
        config = ChannelConfig(transmit_power_w=1., total_bandwidth_hz=4., num_subchannels=2,
                               noise_psd_w_per_hz=1., valid_distance_min_m=.1)
        snapshot = manual_snapshot(config)
        alone = evaluate_schedule(snapshot, [(0, 4, 0)])
        link = alone['links'][0]
        self.assertEqual(link['interference_power_w'], 0.)
        self.assertAlmostEqual(link['sinr'], 1.)
        self.assertAlmostEqual(link['rate_bps'], 2.)
        same = evaluate_schedule(snapshot, [(0, 4, 0), (1, 2, 0)])
        other = evaluate_schedule(snapshot, [(0, 4, 0), (1, 2, 1)])
        self.assertLess(same['links'][0]['rate_bps'], alone['links'][0]['rate_bps'])
        self.assertAlmostEqual(other['links'][0]['rate_bps'], alone['links'][0]['rate_bps'])
        self.assertAlmostEqual(same['link_total_rate_bps'][0, 4], same['links'][0]['rate_bps'])
        self.assertEqual(same['per_subchannel_rate_bps'][0, 4, 1], 0.)

    def test_illegal_schedules_and_offline_nodes_are_rejected(self):
        config = ChannelConfig(valid_distance_min_m=.1)
        snapshot = manual_snapshot(config, [True, True, True, False, True])
        invalid = [
            [(0, 0, 0)], [(4, 0, 0)], [(3, 4, 0)],
            [(0, 4, 0), (0, 2, 1)], [(0, 4, 0), (1, 4, 0)],
            [(0, 4, 0), (0, 4, 0)],
        ]
        for schedule in invalid:
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                evaluate_schedule(snapshot, schedule)
        with self.assertRaises(ValueError):
            evaluate_schedule(snapshot, [(0, 4, 0)], candidate_links={(1, 4)})

    def test_seed_reproducibility_cache_and_reciprocity(self):
        config = ChannelConfig(valid_distance_min_m=.1, random_seed=9)
        data = dataset([[5, 2, 3], [10, 4, 6]])
        first_model, second_model = ChannelModel(data, config), ChannelModel(data, config)
        first = first_model.snapshot(0)
        self.assertIs(first, first_model.snapshot(0))
        arrays = ('positions_m', 'scheduling_eligible', 'distance_m', 'obstruction_state',
                  'path_loss_db', 'shadowing_db', 'large_scale_gain',
                  'small_scale_coefficient', 'power_gain')
        self.assertTrue(all(not getattr(first, name).flags.writeable for name in arrays))
        with self.assertRaises(ValueError):
            first.power_gain[0, 1] = 0
        copied = first.power_gain.copy()
        copied[0, 1] = 0
        self.assertNotEqual(copied[0, 1], first.power_gain[0, 1])
        np.testing.assert_allclose(first.power_gain, second_model.snapshot(0).power_gain,
                                   equal_nan=True)
        np.testing.assert_allclose(first.power_gain, first.power_gain.T, equal_nan=True)
        one = evaluate_schedule(first, [(0, 2, 0)])
        two = evaluate_schedule(first, [(0, 2, 0)])
        self.assertEqual(one['links'], two['links'])
        before = {name: getattr(first, name).tobytes() for name in arrays}
        evaluate_schedule(first, [(1, 2, 1)])
        self.assertEqual(before, {name: getattr(first, name).tobytes() for name in arrays})

    def test_base_station_capacity_upper_bound_and_conclusions(self):
        config = ChannelConfig(transmit_power_w=1., total_bandwidth_hz=2., num_subchannels=2,
                               noise_psd_w_per_hz=1., valid_distance_min_m=.1)
        snapshots = [
            manual_snapshot(config, [True, True, True, False, True], [3., 2., 1., 100.], slot=0),
            manual_snapshot(config, [False, False, False, False, True], [100.] * 4, slot=1),
        ]
        data = Struct(scenario={'slot_count': 2, 'slot_seconds': 2.})
        model = Struct(config=config, num_uavs=4, base_station_id=4,
                       snapshot=lambda slot: snapshots[slot])
        expected = 2 * (np.log2(1 + 3) + np.log2(1 + 2))
        high = base_station_capacity_upper_bound(data, model=model,
            total_data_bits=int(np.ceil(expected + 1)), data_unit_count=4)
        low = base_station_capacity_upper_bound(data, model=model,
            total_data_bits=int(np.floor(expected - 1)), data_unit_count=4)
        self.assertAlmostEqual(high['bs_capacity_upper_bound_bits'], expected)
        self.assertEqual(high['selected_uav_count_per_slot'].tolist(), [2, 0])
        self.assertTrue(high['infeasible_by_upper_bound'])
        self.assertEqual(low['conclusion'], 'not_ruled_out_by_upper_bound')

    def test_candidate_loads_do_not_write_input_files(self):
        report = {'bs_capacity_upper_bound_bits': 8000., 'data_unit_count': 10}
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / 'input.bin'
            marker.write_bytes(b'unchanged')
            before = {p.name: p.read_bytes() for p in Path(folder).iterdir()}
            rows = candidate_loads(report)
            after = {p.name: p.read_bytes() for p in Path(folder).iterdir()}
        self.assertEqual(before, after)
        self.assertEqual([row['data_per_collection_bytes'] for row in rows], [10, 25, 50])


if __name__ == '__main__':
    unittest.main()
