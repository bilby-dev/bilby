import numpy as np

from .. import utils as gwutils


class InterferometerGeometry(object):
    def __init__(self, length, latitude, longitude, elevation, xarm_azimuth, yarm_azimuth,
                 xarm_tilt=0., yarm_tilt=0.):
        """
        Instantiate an Interferometer object.

        Parameters
        ----------

        length: float
            Length of the interferometer in km.
        latitude: float
            Latitude North in degrees (South is negative).
        longitude: float
            Longitude East in degrees (West is negative).
        elevation: float
            Height above surface in metres.
        xarm_azimuth: float
            Orientation of the x arm in degrees North of East.
        yarm_azimuth: float
            Orientation of the y arm in degrees North of East.
        xarm_tilt: float, optional
            Tilt of the x arm in radians above the horizontal defined by
            ellipsoid earth model in LIGO-T980044-08.
        yarm_tilt: float, optional
            Tilt of the y arm in radians above the horizontal.
        """

        self._x_updated = False
        self._y_updated = False
        self._vertex_updated = False
        self._detector_tensor_updated = False

        self.length = length
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.xarm_azimuth = xarm_azimuth
        self.yarm_azimuth = yarm_azimuth
        self.xarm_tilt = xarm_tilt
        self.yarm_tilt = yarm_tilt
        self._vertex = None
        self._x = None
        self._y = None
        self._detector_tensor = None

    def __eq__(self, other):
        for attribute in ['length', 'latitude', 'longitude', 'elevation',
                          'xarm_azimuth', 'yarm_azimuth', 'xarm_tilt', 'yarm_tilt']:
            if not getattr(self, attribute) == getattr(other, attribute):
                return False
        return True

    def __repr__(self):
        return self.__class__.__name__ + '(length={}, latitude={}, longitude={}, elevation={}, ' \
                                         'xarm_azimuth={}, yarm_azimuth={}, xarm_tilt={}, yarm_tilt={})' \
            .format(float(self.length), float(self.latitude), float(self.longitude),
                    float(self.elevation), float(self.xarm_azimuth), float(self.yarm_azimuth), float(self.xarm_tilt),
                    float(self.yarm_tilt))

    @property
    def latitude(self):
        """ Saves latitude in rad internally. Updates related quantities if set to a different value.

        Returns
        -------
        float: The latitude position of the detector in degree
        """
        return self._latitude * 180 / np.pi

    @latitude.setter
    def latitude(self, latitude):
        self._latitude = latitude * np.pi / 180
        self._x_updated = False
        self._y_updated = False
        self._vertex_updated = False

    @property
    def latitude_radians(self):
        """
        Returns
        -------
        float: The latitude position of the detector in radians
        """

        return self._latitude

    @property
    def longitude(self):
        """ Saves longitude in rad internally. Updates related quantities if set to a different value.

        Returns
        -------
        float: The longitude position of the detector in degree
        """
        return self._longitude * 180 / np.pi

    @longitude.setter
    def longitude(self, longitude):
        self._longitude = longitude * np.pi / 180
        self._x_updated = False
        self._y_updated = False
        self._vertex_updated = False

    @property
    def longitude_radians(self):
        """
        Returns
        -------
        float: The latitude position of the detector in radians
        """

        return self._longitude

    @property
    def elevation(self):
        """ Updates related quantities if set to a different values.

        Returns
        -------
        float: The height about the surface in meters
        """
        return self._elevation

    @elevation.setter
    def elevation(self, elevation):
        self._elevation = elevation
        self._vertex_updated = False

    @property
    def xarm_azimuth(self):
        """ Saves the x-arm azimuth in rad internally. Updates related quantities if set to a different values.

        Returns
        -------
        float: The x-arm azimuth in degrees.

        """
        return self._xarm_azimuth * 180 / np.pi

    @xarm_azimuth.setter
    def xarm_azimuth(self, xarm_azimuth):
        self._xarm_azimuth = xarm_azimuth * np.pi / 180
        self._x_updated = False

    @property
    def yarm_azimuth(self):
        """ Saves the y-arm azimuth in rad internally. Updates related quantities if set to a different values.

        Returns
        -------
        float: The y-arm azimuth in degrees.

        """
        return self._yarm_azimuth * 180 / np.pi

    @yarm_azimuth.setter
    def yarm_azimuth(self, yarm_azimuth):
        self._yarm_azimuth = yarm_azimuth * np.pi / 180
        self._y_updated = False

    @property
    def xarm_tilt(self):
        """ Updates related quantities if set to a different values.

        Returns
        -------
        float: The x-arm tilt in radians.

        """
        return self._xarm_tilt

    @xarm_tilt.setter
    def xarm_tilt(self, xarm_tilt):
        self._xarm_tilt = xarm_tilt
        self._x_updated = False

    @property
    def yarm_tilt(self):
        """ Updates related quantities if set to a different values.

        Returns
        -------
        float: The y-arm tilt in radians.

        """
        return self._yarm_tilt

    @yarm_tilt.setter
    def yarm_tilt(self, yarm_tilt):
        self._yarm_tilt = yarm_tilt
        self._y_updated = False

    @property
    def vertex(self):
        """ Position of the IFO vertex in geocentric coordinates in meters.

        Is automatically updated if related quantities are modified.

        Returns
        -------
        array_like: A 3D array representation of the vertex
        """
        if not self._vertex_updated:
            self._vertex = gwutils.get_vertex_position_geocentric(self._latitude, self._longitude,
                                                                  self.elevation)
            self._vertex_updated = True
        return self._vertex

    @property
    def x(self):
        """ A unit vector along the x-arm

        Is automatically updated if related quantities are modified.

        Returns
        -------
        array_like: A 3D array representation of a unit vector along the x-arm

        """
        if not self._x_updated:
            self._x = self.unit_vector_along_arm('x')
            self._x_updated = True
            self._detector_tensor_updated = False
        return self._x

    @property
    def y(self):
        """ A unit vector along the y-arm

        Is automatically updated if related quantities are modified.

        Returns
        -------
        array_like: A 3D array representation of a unit vector along the y-arm

        """
        if not self._y_updated:
            self._y = self.unit_vector_along_arm('y')
            self._y_updated = True
            self._detector_tensor_updated = False
        return self._y

    @property
    def detector_tensor(self):
        """
        Calculate the detector tensor from the unit vectors along each arm of the detector.

        See Eq. B6 of arXiv:gr-qc/0008066

        Is automatically updated if related quantities are modified.

        Returns
        -------
        array_like: A 3x3 array representation of the detector tensor

        """
        if not self._x_updated or not self._y_updated:
            _, _ = self.x, self.y  # noqa
        if not self._detector_tensor_updated:
            self._detector_tensor = 0.5 * (np.einsum('i,j->ij', self.x, self.x) - np.einsum('i,j->ij', self.y, self.y))
            self._detector_tensor_updated = True
        return self._detector_tensor

    def unit_vector_along_arm(self, arm):
        """
        Calculate the unit vector pointing along the specified arm in cartesian Earth-based coordinates.

        See Eqs. B14-B17 in arXiv:gr-qc/0008066

        Parameters
        -------
        arm: str
            'x' or 'y' (arm of the detector)

        Returns
        -------
        array_like: 3D unit vector along arm in cartesian Earth-based coordinates

        Raises
        -------
        ValueError: If arm is neither 'x' nor 'y'

        """
        if arm == 'x':
            return self._calculate_arm(self._xarm_tilt, self._xarm_azimuth)
        elif arm == 'y':
            return self._calculate_arm(self._yarm_tilt, self._yarm_azimuth)
        else:
            raise ValueError("Arm must either be 'x' or 'y'.")

    def _calculate_arm(self, arm_tilt, arm_azimuth):
        e_long = np.array([-np.sin(self._longitude), np.cos(self._longitude), 0])
        e_lat = np.array([-np.sin(self._latitude) * np.cos(self._longitude),
                          -np.sin(self._latitude) * np.sin(self._longitude), np.cos(self._latitude)])
        e_h = np.array([np.cos(self._latitude) * np.cos(self._longitude),
                        np.cos(self._latitude) * np.sin(self._longitude), np.sin(self._latitude)])

        return (np.cos(arm_tilt) * np.cos(arm_azimuth) * e_long +
                np.cos(arm_tilt) * np.sin(arm_azimuth) * e_lat +
                np.sin(arm_tilt) * e_h)
