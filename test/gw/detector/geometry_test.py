import unittest
from unittest import mock

import numpy as np

import bilby


class TestInterferometerGeometry(unittest.TestCase):
    def setUp(self):
        self.length = 30
        self.latitude = 1
        self.longitude = 2
        self.elevation = 3
        self.xarm_azimuth = 4
        self.yarm_azimuth = 5
        self.xarm_tilt = 0.0
        self.yarm_tilt = 0.0
        self.geometry = bilby.gw.detector.InterferometerGeometry(
            length=self.length,
            latitude=self.latitude,
            longitude=self.longitude,
            elevation=self.elevation,
            xarm_azimuth=self.xarm_azimuth,
            yarm_azimuth=self.yarm_azimuth,
            xarm_tilt=self.xarm_tilt,
            yarm_tilt=self.yarm_tilt,
        )

    def tearDown(self):
        del self.length
        del self.latitude
        del self.longitude
        del self.elevation
        del self.xarm_azimuth
        del self.yarm_azimuth
        del self.xarm_tilt
        del self.yarm_tilt
        del self.geometry

    def test_length_setting(self):
        self.assertEqual(self.geometry.length, self.length)

    def test_latitude_setting(self):
        self.assertEqual(self.geometry.latitude, self.latitude)

    def test_longitude_setting(self):
        self.assertEqual(self.geometry.longitude, self.longitude)

    def test_elevation_setting(self):
        self.assertEqual(self.geometry.elevation, self.elevation)

    def test_xarm_azi_setting(self):
        self.assertEqual(self.geometry.xarm_azimuth, self.xarm_azimuth)

    def test_yarm_azi_setting(self):
        self.assertEqual(self.geometry.yarm_azimuth, self.yarm_azimuth)

    def test_xarm_tilt_setting(self):
        self.assertEqual(self.geometry.xarm_tilt, self.xarm_tilt)

    def test_yarm_tilt_setting(self):
        self.assertEqual(self.geometry.yarm_tilt, self.yarm_tilt)

    def test_vertex_without_update(self):
        _ = self.geometry.vertex
        with mock.patch("bilby.gw.utils.get_vertex_position_geocentric") as m:
            m.return_value = np.array([1])
            self.assertFalse(np.array_equal(self.geometry.vertex, np.array([1])))

    def test_vertex_with_latitude_update(self):
        with mock.patch("bilby.gw.utils.get_vertex_position_geocentric") as m:
            m.return_value = np.array([1])
            self.geometry.latitude = 5
            self.assertEqual(self.geometry.vertex, np.array([1]))

    def test_vertex_with_longitude_update(self):
        with mock.patch("bilby.gw.utils.get_vertex_position_geocentric") as m:
            m.return_value = np.array([1])
            self.geometry.longitude = 5
            self.assertEqual(self.geometry.vertex, np.array([1]))

    def test_vertex_with_elevation_update(self):
        with mock.patch("bilby.gw.utils.get_vertex_position_geocentric") as m:
            m.return_value = np.array([1])
            self.geometry.elevation = 5
            self.assertEqual(self.geometry.vertex, np.array([1]))

    def test_x_without_update(self):
        _ = self.geometry.x
        self.geometry.unit_vector_along_arm = mock.MagicMock(return_value=np.array([1]))

        self.assertFalse(np.array_equal(self.geometry.x, np.array([1])))

    def test_x_with_xarm_tilt_update(self):
        self.geometry.unit_vector_along_arm = mock.MagicMock(return_value=np.array([1]))
        self.geometry.xarm_tilt = 0
        self.assertTrue(np.array_equal(self.geometry.x, np.array([1])))

    def test_x_with_xarm_azimuth_update(self):
        self.geometry.unit_vector_along_arm = mock.MagicMock(return_value=np.array([1]))
        self.geometry.xarm_azimuth = 0
        self.assertTrue(np.array_equal(self.geometry.x, np.array([1])))

    def test_x_with_longitude_update(self):
        self.geometry.unit_vector_along_arm = mock.MagicMock(return_value=np.array([1]))
        self.geometry.longitude = 0
        self.assertTrue(np.array_equal(self.geometry.x, np.array([1])))

    def test_x_with_latitude_update(self):
        self.geometry.unit_vector_along_arm = mock.MagicMock(return_value=np.array([1]))
        self.geometry.latitude = 0
        self.assertTrue(np.array_equal(self.geometry.x, np.array([1])))

    def test_y_without_update(self):
        _ = self.geometry.y
        self.geometry.unit_vector_along_arm = mock.MagicMock(return_value=np.array([1]))

        self.assertFalse(np.array_equal(self.geometry.y, np.array([1])))

    def test_y_with_yarm_tilt_update(self):
        self.geometry.unit_vector_along_arm = mock.MagicMock(return_value=np.array([1]))
        self.geometry.yarm_tilt = 0
        self.assertTrue(np.array_equal(self.geometry.y, np.array([1])))

    def test_y_with_yarm_azimuth_update(self):
        self.geometry.unit_vector_along_arm = mock.MagicMock(return_value=np.array([1]))
        self.geometry.yarm_azimuth = 0
        self.assertTrue(np.array_equal(self.geometry.y, np.array([1])))

    def test_y_with_longitude_update(self):
        self.geometry.unit_vector_along_arm = mock.MagicMock(return_value=np.array([1]))
        self.geometry.longitude = 0
        self.assertTrue(np.array_equal(self.geometry.y, np.array([1])))

    def test_y_with_latitude_update(self):
        self.geometry.unit_vector_along_arm = mock.MagicMock(return_value=np.array([1]))
        self.geometry.latitude = 0
        self.assertTrue(np.array_equal(self.geometry.y, np.array([1])))

    def test_detector_tensor_with_x_azimuth_update(self):
        original = self.geometry.detector_tensor
        self.geometry.xarm_azimuth += 1
        self.assertNotEqual(np.max(abs(self.geometry.detector_tensor - original)), 0)

    def test_detector_tensor_with_y_azimuth_update(self):
        original = self.geometry.detector_tensor
        self.geometry.yarm_azimuth += 1
        self.assertNotEqual(np.max(abs(self.geometry.detector_tensor - original)), 0)

    def test_detector_tensor_with_x_tilt_update(self):
        original = self.geometry.detector_tensor
        self.geometry.xarm_tilt += 1
        self.assertNotEqual(np.max(abs(self.geometry.detector_tensor - original)), 0)

    def test_detector_tensor_with_y_tilt_update(self):
        original = self.geometry.detector_tensor
        self.geometry.yarm_tilt += 1
        self.assertNotEqual(np.max(abs(self.geometry.detector_tensor - original)), 0)

    def test_detector_tensor_with_longitude_update(self):
        original = self.geometry.detector_tensor
        self.geometry.longitude += 1
        self.assertNotEqual(np.max(abs(self.geometry.detector_tensor - original)), 0)

    def test_detector_tensor_with_latitude_update(self):
        original = self.geometry.detector_tensor
        self.geometry.latitude += 1
        self.assertNotEqual(np.max(abs(self.geometry.detector_tensor - original)), 0)

    def test_unit_vector_along_arm_default(self):
        with self.assertRaises(ValueError):
            self.geometry.unit_vector_along_arm("z")

    def test_unit_vector_along_arm_x(self):
        self.geometry.longitude = 0
        self.geometry.latitude = 0
        self.geometry.xarm_tilt = 0
        self.geometry.xarm_azimuth = 0
        arm = self.geometry.unit_vector_along_arm("x")
        self.assertTrue(np.allclose(arm, np.array([0, 1, 0])))

    def test_unit_vector_along_arm_y(self):
        self.geometry.longitude = 0
        self.geometry.latitude = 0
        self.geometry.yarm_tilt = 0
        self.geometry.yarm_azimuth = 90
        arm = self.geometry.unit_vector_along_arm("y")
        print(arm)
        self.assertTrue(np.allclose(arm, np.array([0, 0, 1])))

    def test_repr(self):
        expected = (
            "InterferometerGeometry(length={}, latitude={}, longitude={}, elevation={}, xarm_azimuth={}, "
            "yarm_azimuth={}, xarm_tilt={}, yarm_tilt={})".format(
                float(self.length),
                float(self.latitude),
                float(self.longitude),
                float(self.elevation),
                float(self.xarm_azimuth),
                float(self.yarm_azimuth),
                float(self.xarm_tilt),
                float(self.yarm_tilt),
            )
        )
        self.assertEqual(expected, repr(self.geometry))


if __name__ == "__main__":
    unittest.main()
