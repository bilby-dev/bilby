import numpy as np
import pytest


@pytest.mark.array_backend
def test_transform_precessing_spins(xp):
    """
    Verify that our port of this function matches the lalsimulation version.
    """
    import lal
    from bilby.core.prior import Uniform
    from bilby.gw.prior import BBHPriorDict
    from bilby.gw.geometry import transform_precessing_spins
    from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions

    priors = BBHPriorDict()
    priors["mass_1"] = Uniform(1, 1000)
    priors["mass_2"] = Uniform(1, 1000)
    priors["reference_frequency"] = Uniform(10, 100)

    # some default priors are problematic for some array backends
    for key in ["luminosity_distance", "chirp_mass", "mass_ratio"]:
        del priors[key]

    for _ in range(100):
        point = priors.sample(xp=xp)
        bilby_transformed = np.asarray(transform_precessing_spins(
            point["theta_jn"],
            point["phi_jl"],
            point["tilt_1"],
            point["tilt_2"],
            point["phi_12"],
            point["a_1"],
            point["a_2"],
            point["mass_1"],
            point["mass_2"],
            point["reference_frequency"],
            point["phase"],
        ))
        lalsim_transformed = np.asarray(SimInspiralTransformPrecessingNewInitialConditions(
            float(point["theta_jn"]),
            float(point["phi_jl"]),
            float(point["tilt_1"]),
            float(point["tilt_2"]),
            float(point["phi_12"]),
            float(point["a_1"]),
            float(point["a_2"]),
            float(point["mass_1"] * lal.MSUN_SI),
            float(point["mass_2"] * lal.MSUN_SI),
            float(point["reference_frequency"]),
            float(point["phase"]),
        ))
        np.testing.assert_allclose(bilby_transformed, lalsim_transformed, rtol=1e-10)


@pytest.mark.array_backend
def test_transform_precessing_spins_vectorized(xp):
    """
    Run the tests with vectorization, note that this returns a tuple of arrays.
    """
    import lal
    from bilby.core.prior import Uniform
    from bilby.gw.prior import BBHPriorDict
    from bilby.gw.geometry import transform_precessing_spins
    from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions

    priors = BBHPriorDict()
    priors["mass_1"] = Uniform(1, 1000)
    priors["mass_2"] = Uniform(1, 1000)
    priors["reference_frequency"] = Uniform(10, 100)

    # some default priors are problematic for some array backends
    for key in ["luminosity_distance", "chirp_mass", "mass_ratio"]:
        del priors[key]

    points = priors.sample(100, xp=xp)
    bilby_transformed = np.asarray(transform_precessing_spins(
        points["theta_jn"],
        points["phi_jl"],
        points["tilt_1"],
        points["tilt_2"],
        points["phi_12"],
        points["a_1"],
        points["a_2"],
        points["mass_1"],
        points["mass_2"],
        points["reference_frequency"],
        points["phase"],
    ))
    lalsim_transformed = list()
    for ii in range(len(points["theta_jn"])):
        point = {key: points[key][ii] for key in points.keys()}
        lalsim_transformed.append(np.asarray(SimInspiralTransformPrecessingNewInitialConditions(
            float(point["theta_jn"]),
            float(point["phi_jl"]),
            float(point["tilt_1"]),
            float(point["tilt_2"]),
            float(point["phi_12"]),
            float(point["a_1"]),
            float(point["a_2"]),
            float(point["mass_1"] * lal.MSUN_SI),
            float(point["mass_2"] * lal.MSUN_SI),
            float(point["reference_frequency"]),
            float(point["phase"]),
        )))
    lalsim_transformed = np.asarray(lalsim_transformed).T
    np.testing.assert_allclose(bilby_transformed, lalsim_transformed, rtol=1e-10)
