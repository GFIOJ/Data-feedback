import unittest

import numpy as np

from paper1.common import Struct
from paper2_model.channel import ChannelConfig, ChannelSnapshot
from paper2_model.return_simulation import DataUnit, ReturnSimulation, SimulationConfig


class StaticChannelModel:
    def __init__(self, num_uavs, eligible_by_slot, config=None):
        self.config = config or ChannelConfig(transmit_power_w=1., total_bandwidth_hz=2.,
            num_subchannels=2, noise_psd_w_per_hz=1., valid_distance_min_m=.1)
        self.num_uavs = num_uavs
        self.base_station_id = num_uavs
        self.snapshots = []
        size = num_uavs + 1
        for slot, uav_eligible in enumerate(eligible_by_slot):
            eligible = np.r_[np.asarray(uav_eligible, bool), True]
            shape = (size, size)
            self.snapshots.append(ChannelSnapshot(slot, self.base_station_id, self.config,
                np.zeros((size, 3)), eligible, np.ones(shape), np.zeros(shape, np.int8),
                np.zeros(shape), np.zeros(shape), np.ones(shape), np.ones(shape, complex),
                np.ones(shape)))  # g=1 gives 1 bit/s on each 1 Hz subchannel

    def snapshot(self, slot):
        return self.snapshots[slot]


def simulation(num_uavs, slot_count, units, eligible=None, returns=None, buffer_bits=1e6,
               channel_config=None, takeoff=None, slot_seconds=1.):
    eligible = eligible or [[True] * num_uavs for _ in range(slot_count)]
    returns = returns or [float(slot_count)] * num_uavs
    node_states = {'return_time_s': np.asarray(returns, float)}
    if takeoff is not None:
        node_states['takeoff_time_s'] = np.asarray(takeoff, float)
    data = Struct(scenario={'num_uavs': num_uavs, 'slot_count': slot_count,
                            'slot_seconds': slot_seconds}, node_states=node_states)
    model = StaticChannelModel(num_uavs, eligible, channel_config)
    return ReturnSimulation(data, model, units, SimulationConfig(buffer_capacity_bits=buffer_bits)), model


class ReturnSimulationTests(unittest.TestCase):
    def test_allocation_identifiers_reject_nonintegers_before_queue_update(self):
        for bad in (.5, np.nan, np.inf, True):
            sim, model = simulation(1, 2, [DataUnit(0, 0, 1., 0)])
            with self.subTest(value=bad), self.assertRaisesRegex(ValueError, 'allocation.data_id'):
                sim.run({1: [(0, model.base_station_id, 0)]},
                    lambda slot, *_: [(0, model.base_station_id, bad, 1.)] if slot == 1 else [])

    def test_allocation_python_and_numpy_integer_identifiers_work(self):
        for identifier in (0, np.int64(0)):
            sim, model = simulation(1, 2, [DataUnit(0, 0, 1., 0)])
            result = sim.run({1: [(0, model.base_station_id, 0)]},
                lambda slot, *_: [(np.int64(0), model.base_station_id, identifier, 1.)]
                if slot == 1 else [])
            self.assertTrue(result['complete'][0])

    def test_data_unit_size_and_identifier_validation(self):
        for bad in (-1., 0., np.nan, np.inf):
            with self.subTest(bits=bad), self.assertRaisesRegex(ValueError, 'original_bits'):
                simulation(1, 2, [DataUnit(0, 0, bad, 0)])
        invalid_sets = [
            [DataUnit(0, 0, 1., 0), DataUnit(0, 0, 1., 0)],
            [DataUnit(1, 0, 1., 0)],
            [DataUnit(1, 0, 1., 0), DataUnit(0, 0, 1., 0)],
            [DataUnit(True, 0, 1., 0)],
        ]
        for units in invalid_sets:
            with self.subTest(units=units), self.assertRaisesRegex(ValueError, 'data_id'):
                simulation(1, 2, units)

    def test_collection_time_slot_boundary_and_online_interval_validation(self):
        with self.assertRaisesRegex(ValueError, 'belongs to slot 10'):
            simulation(1, 12, [DataUnit(0, 0, 1., 0, 10.)])
        simulation(1, 2, [DataUnit(0, 0, 1., 1, 1.)])
        with self.assertRaisesRegex(ValueError, 'belongs to slot 1'):
            simulation(1, 2, [DataUnit(0, 0, 1., 0, 1.)])
        for bad_time in (.2, 1.75, 1.9):
            with self.subTest(time=bad_time), self.assertRaisesRegex(ValueError, 'online interval'):
                simulation(1, 2, [DataUnit(0, 0, 1., int(bad_time), bad_time)],
                           takeoff=[.25], returns=[1.75])
        simulation(1, 2, [DataUnit(0, 0, 1., 0, .5)],
                   takeoff=[.25], returns=[1.75])

    def test_missing_exact_collection_time_uses_compatible_slot_end_proxy(self):
        sim, _ = simulation(1, 2, [DataUnit(0, 0, 1., 0, None)],
                            takeoff=[.25], returns=[.75])
        self.assertEqual(sim.collection_times_s[0], 1.)

    def test_simulation_dimensions_source_slot_and_time_scalars_are_validated(self):
        for bad_time in (-.1, np.nan, np.inf, True):
            with self.subTest(time=bad_time), self.assertRaisesRegex(ValueError,
                    'collection_complete_time_s'):
                simulation(1, 2, [DataUnit(0, 0, 1., 0, bad_time)])
        for unit, field in ((DataUnit(0, -1, 1., 0), 'source_uav_id'),
                            (DataUnit(0, 0, 1., 2), 'collection_slot_zero_based')):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                simulation(1, 2, [unit])
        for bad_dt in (0., np.nan, np.inf):
            with self.subTest(dt=bad_dt), self.assertRaisesRegex(ValueError, 'slot_seconds'):
                simulation(1, 2, [DataUnit(0, 0, 1., 0)], slot_seconds=bad_dt)
        with self.assertRaisesRegex(ValueError, 'slot_count'):
            simulation(1, True, [DataUnit(0, 0, 1., 0)])

    def test_arrival_at_slot_end_can_only_send_next_slot(self):
        sim, model = simulation(1, 2, [DataUnit(0, 0, 1., 0, .25)])
        result = sim.run({0: [(0, model.base_station_id, 0)],
                          1: [(0, model.base_station_id, 0)]})
        self.assertEqual(result['delivery_slot_paper_one_based'][0], 2)
        self.assertEqual(result['delays_s'][0], 1.75)
        self.assertEqual(result['paper_slot_delays_s'][0], 1.)
        self.assertEqual(result['slot_records'][0]['transmitted_bits'], 0.)
        self.assertAlmostEqual(result['delays_s'][0], result['pre_first_send_wait_s'][0] +
                               result['post_first_send_to_complete_s'][0])
        self.assertEqual(result['first_tx_time_s'][0], 1.)

    def test_single_hop_spans_multiple_slots(self):
        sim, model = simulation(1, 4, [DataUnit(0, 0, 3., 0)])
        schedule = {slot: [(0, model.base_station_id, 0)] for slot in range(4)}
        result = sim.run(schedule)
        self.assertEqual(result['delivery_slot_paper_one_based'][0], 4)
        self.assertEqual(result['delays_s'][0], 3.)
        self.assertTrue(result['conservation_passed'])

    def test_different_sized_units_complete_from_their_own_sizes(self):
        sim, model = simulation(1, 5, [DataUnit(0, 0, 1., 0), DataUnit(1, 0, 2., 0)])
        schedule = {slot: [(0, model.base_station_id, 0)] for slot in range(1, 5)}
        result = sim.run(schedule)
        np.testing.assert_array_equal(result['delivery_slot_paper_one_based'], [2, 4])
        np.testing.assert_allclose(result['base_received_bits'], [1., 2.])
        self.assertEqual(result['complete_data_unit_count'], 2)
        self.assertTrue(result['conservation_passed'])

    def test_two_hop_relay_cannot_forward_newly_received_data(self):
        sim, model = simulation(2, 3, [DataUnit(0, 0, 1., 0)])
        result = sim.run({1: [(0, 1, 0)], 2: [(1, model.base_station_id, 0)]})
        self.assertEqual(result['base_received_bits'][0], 1.)
        self.assertEqual(result['delivery_slot_paper_one_based'][0], 3)
        self.assertEqual(result['slot_records'][1]['base_received_cumulative_bits'], 0.)

    def test_fragments_on_two_paths_complete_only_after_both_arrive(self):
        sim, model = simulation(3, 4, [DataUnit(0, 0, 2., 0)])
        schedule = {1: [(0, 1, 0)],
                    2: [(0, 2, 0), (1, model.base_station_id, 1)],
                    3: [(2, model.base_station_id, 0)]}
        result = sim.run(schedule)
        self.assertEqual(result['slot_records'][2]['base_received_cumulative_bits'], 1.)
        self.assertEqual(result['delivery_slot_paper_one_based'][0], 4)
        self.assertEqual(result['delays_s'][0], 3.)

    def test_capacity_and_source_queue_limits_are_rejected(self):
        sim, model = simulation(1, 2, [DataUnit(0, 0, 1., 0)])
        schedule = {1: [(0, model.base_station_id, 0)]}
        with self.assertRaisesRegex(ValueError, 'capacity'):
            sim.run(schedule, lambda slot, state, snap, rates:
                    [(0, model.base_station_id, 0, 2.)] if slot == 1 else [])
        sim, model = simulation(1, 2, [DataUnit(0, 0, .5, 0)])
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            sim.run(schedule, lambda slot, state, snap, rates:
                    [(0, model.base_station_id, 0, .75)] if slot == 1 else [])

    def test_buffer_violation_exit_residual_and_partial_delay_reporting(self):
        units = [DataUnit(0, 0, 1., 0), DataUnit(1, 1, 2., 0)]
        eligible = [[True, False], [True, False]]
        sim, model = simulation(2, 2, units, eligible=eligible, returns=[2., 1.], buffer_bits=1.)
        result = sim.run({1: [(0, model.base_station_id, 0)]})
        self.assertTrue(result['buffer_constraint_violated'])
        self.assertFalse(result['all_data_complete'])
        self.assertIsNone(result['all_data_mean_delay_s'])
        self.assertEqual(result['completed_data_conditional_mean_delay_s'], 1.)
        self.assertEqual(result['incomplete_data_unit_count'], 1)
        exit_one = next(row for row in result['exit_stranded'] if row['uav_id'] == 1)
        self.assertEqual(exit_one['stranded_bits'], 2.)
        self.assertEqual(exit_one['stranded_data_unit_ids'], [1])
        self.assertEqual(result['residual_data_bits'], 2.)

    def test_round_robin_serves_persistent_backlogs_in_id_order(self):
        units = [DataUnit(u, u, 10., 0, .1) for u in range(5)]
        sim, _ = simulation(5, 4, units)
        result = sim.run(sim.make_round_robin_schedule())
        selected = [np.flatnonzero(row).tolist() for row in result['slot_uav_scheduled']]
        self.assertEqual(selected, [[], [0, 1], [2, 3], [0, 4]])

    def test_oldest_first_ties_offline_and_partial_age(self):
        config = ChannelConfig(transmit_power_w=1., total_bandwidth_hz=1., num_subchannels=1,
            noise_psd_w_per_hz=1., valid_distance_min_m=.1)
        units = [DataUnit(0, 0, 2., 0, .1), DataUnit(1, 1, 1., 0, .9)]
        sim, _ = simulation(2, 4, units, channel_config=config)
        result = sim.run(sim.oldest_data_first_schedule)
        selected = [np.flatnonzero(row).tolist() for row in result['slot_uav_scheduled']]
        self.assertEqual(selected, [[], [0], [0], [1]])
        self.assertEqual(result['positive_tx_slot_count'][0], 2)
        tied = [DataUnit(u, u, 1., 0, .1) for u in range(3)]
        offline_sim, model = simulation(3, 2, tied,
                                        eligible=[[True, False, True], [True, False, True]])
        state = type('State', (), {'node_total_buffer_bits': np.ones(3),
            'queues_bits': np.eye(3)})()
        schedule = offline_sim.oldest_data_first_schedule(1, state, model.snapshot(1))
        self.assertEqual(schedule, [(0, model.base_station_id, 0),
                                    (2, model.base_station_id, 1)])


if __name__ == '__main__':
    unittest.main()
