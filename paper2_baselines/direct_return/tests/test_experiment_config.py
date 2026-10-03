import unittest

from paper2_model.experiment_config import BUFFER_BITS, communication_configs
from paper2_baselines.direct_return.prepare_input import DATA_SIZE_SEED, MAX_BYTES, MIN_BYTES, sampled_sizes


class ReferenceExperimentConfigurationTests(unittest.TestCase):
    def test_requested_units_and_radio_resources(self):
        channel, simulation = communication_configs()
        self.assertEqual(channel.num_subchannels, 4)
        self.assertEqual(channel.subchannel_bandwidth_hz, 5_000_000.)
        self.assertEqual(channel.total_bandwidth_hz, 20_000_000.)
        self.assertEqual(channel.transmit_power_w, .1)
        self.assertAlmostEqual(channel.noise_psd_w_per_hz *
                               channel.subchannel_bandwidth_hz, 2e-14)
        self.assertEqual(BUFFER_BITS, 8_000_000_000)
        self.assertEqual(simulation.buffer_capacity_bits, BUFFER_BITS)

    def test_variable_sizes_are_inclusive_integer_and_reproducible(self):
        first = sampled_sizes(100_000)
        second = sampled_sizes(100_000, DATA_SIZE_SEED)
        self.assertEqual(first.dtype.kind, 'i')
        self.assertGreaterEqual(first.min(), MIN_BYTES)
        self.assertLessEqual(first.max(), MAX_BYTES)
        self.assertTrue((first == second).all())


if __name__ == '__main__':
    unittest.main()
