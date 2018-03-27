class Source:
    def __init__(self, name):
        self.name = name


class Glitch(Source):
    def __init__(self, name):
        Source.__init__(self, name)


class AstrophysicalSource(Source):

    def __init__(self, name, right_ascension, declination, luminosity_distance):
        Source.__init__(self, name)
        self.right_ascension = right_ascension
        self.declination = declination
        self.luminosity_distance = luminosity_distance


class CompactBinaryCoalescence(AstrophysicalSource):
    def __init__(self, name, right_ascension, declination, luminosity_distance, mass_1, mass_2, spin_1, spin_2,
                 coalescence_time, inclination_angle, waveform_phase, polarisation_angle):
        AstrophysicalSource.__init__(self, name, right_ascension, declination, luminosity_distance)
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.spin_1 = spin_1
        self.spin_2 = spin_2
        self.coalescence_time = coalescence_time  # tc
        self.inclination_angle = inclination_angle  # iota
        self.waveform_phase = waveform_phase  # phi
        self.polarisation_angle = polarisation_angle  # psi


class Supernova(AstrophysicalSource):
    def __init__(self, name, right_ascension, declination, luminosity_distance):
        AstrophysicalSource.__init__(self, name, right_ascension, declination, luminosity_distance)


class BinaryBlackHole(CompactBinaryCoalescence):
    def __init__(self, name, right_ascension, declination, luminosity_distance, mass_1, mass_2, spin_1, spin_2,
                 coalescence_time, inclination_angle, waveform_phase, polarisation_angle):

        CompactBinaryCoalescence.__init__(self, name, right_ascension, declination, luminosity_distance, mass_1,
                                          mass_2, spin_1, spin_2, coalescence_time, inclination_angle, waveform_phase,
                                          polarisation_angle)


class BinaryBlackNeutronStar(CompactBinaryCoalescence):
    def __init__(self, name, right_ascension, declination, luminosity_distance, mass_1, mass_2, spin_1, spin_2,
                 coalescence_time, inclination_angle, waveform_phase, polarisation_angle, tidal_deformability):

        CompactBinaryCoalescence.__init__(self, name, right_ascension, declination, luminosity_distance, mass_1,
                                          mass_2, spin_1, spin_2, coalescence_time, inclination_angle, waveform_phase,
                                          polarisation_angle)
        self.tidal_deformability = tidal_deformability  #lambda paramter
