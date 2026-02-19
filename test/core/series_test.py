import unittest

import array_api_compat as aac
import numpy as np
import pytest

from bilby.core.utils import create_frequency_series, create_time_series
from bilby.core.series import CoupledTimeAndFrequencySeries


@pytest.mark.array_backend
@pytest.mark.usefixtures("xp_class")
class TestCoupledTimeAndFrequencySeries(unittest.TestCase):
    def setUp(self):
        self.duration = self.xp.asarray(2.0)
        self.sampling_frequency = self.xp.asarray(4096.0)
        self.start_time = self.xp.asarray(-1.0)
        self.series = CoupledTimeAndFrequencySeries(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            start_time=self.start_time,
        )

    def tearDown(self):
        del self.duration
        del self.sampling_frequency
        del self.start_time
        del self.series

    def test_repr(self):
        expected = (
            "CoupledTimeAndFrequencySeries(duration={}, sampling_frequency={},"
            " start_time={})".format(
                self.series.duration,
                self.series.sampling_frequency,
                self.series.start_time,
            )
        )
        self.assertEqual(expected, repr(self.series))

    def test_duration_from_init(self):
        self.assertEqual(self.duration, self.series.duration)

    def test_sampling_from_init(self):
        self.assertEqual(self.sampling_frequency, self.series.sampling_frequency)

    def test_start_time_from_init(self):
        self.assertEqual(self.start_time, self.series.start_time)

    def test_frequency_array_type(self):
        self.assertEqual(aac.get_namespace(self.series.frequency_array), self.xp)

    def test_time_array_type(self):
        self.assertEqual(aac.get_namespace(self.series.time_array), self.xp)

    def test_frequency_array_from_init(self):
        expected = create_frequency_series(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )
        self.assertTrue(np.array_equal(expected, self.series.frequency_array))

    def test_time_array_from_init(self):
        expected = create_time_series(
            sampling_frequency=self.sampling_frequency,
            duration=self.duration,
            starting_time=self.start_time,
        )
        self.assertTrue(np.array_equal(expected, self.series.time_array))

    def test_frequency_array_setter(self):
        new_sampling_frequency = self.xp.asarray(100.0)
        new_duration = self.xp.asarray(3.0)
        new_frequency_array = create_frequency_series(
            sampling_frequency=new_sampling_frequency, duration=new_duration
        )
        self.series.frequency_array = new_frequency_array
        self.assertTrue(
            np.array_equal(new_frequency_array, self.series.frequency_array)
        )
        self.assertLessEqual(
            np.abs(new_sampling_frequency - self.series.sampling_frequency), 1
        )
        self.assertAlmostEqual(new_duration, self.series.duration)
        self.assertAlmostEqual(self.start_time, self.series.start_time)

    def test_time_array_setter(self):
        new_sampling_frequency = self.xp.asarray(100.0)
        new_duration = self.xp.asarray(3.0)
        new_start_time = self.xp.asarray(4.0)
        new_time_array = create_time_series(
            sampling_frequency=new_sampling_frequency,
            duration=new_duration,
            starting_time=new_start_time,
        )
        self.series.time_array = new_time_array
        self.assertTrue(np.array_equal(new_time_array, self.series.time_array))
        self.assertAlmostEqual(
            np.asarray(new_sampling_frequency), np.asarray(self.series.sampling_frequency), places=1
        )
        self.assertAlmostEqual(np.asarray(new_duration), np.asarray(self.series.duration), places=1)
        self.assertAlmostEqual(np.asarray(new_start_time), np.asarray(self.series.start_time), places=1)

    def test_time_array_without_sampling_frequency(self):
        self.series.sampling_frequency = None
        self.series.duration = self.xp.asarray(4)
        with self.assertRaises(ValueError):
            _ = self.series.time_array

    def test_time_array_without_duration(self):
        self.series.sampling_frequency = self.xp.asarray(4096)
        self.series.duration = None
        with self.assertRaises(ValueError):
            _ = self.series.time_array

    def test_frequency_array_without_sampling_frequency(self):
        self.series.sampling_frequency = None
        self.series.duration = self.xp.asarray(4)
        with self.assertRaises(ValueError):
            _ = self.series.frequency_array

    def test_frequency_array_without_duration(self):
        self.series.sampling_frequency = self.xp.asarray(4096)
        self.series.duration = None
        with self.assertRaises(ValueError):
            _ = self.series.frequency_array


if __name__ == "__main__":
    unittest.main()
