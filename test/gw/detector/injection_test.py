import bilby
import pytest
from gwpy.frequencyseries import FrequencySeries


@pytest.mark.flaky(reruns=3)
def test_injection_into_timeseries_matches_ifo_injections():
    """
    Test that injecting into a gwpy timeseries agree with the frequency-domain
    injection.

    Caveats:
    - there is a discrete alignment of the maximum strain in the time domain
      injection that means the overlap and SNRs don't exactly match. This is
      especially pronounced when the sampling frequency is low.
    """
    duration = 8
    sampling_frequency = 16384
    ifo = bilby.gw.detector.get_empty_interferometer("H1")
    ifo.set_strain_data_from_zero_noise(
        sampling_frequency=sampling_frequency, duration=duration
    )
    wfg = bilby.gw.waveform_generator.WaveformGenerator(
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        duration=duration,
        sampling_frequency=sampling_frequency,
    )
    priors = bilby.gw.prior.BBHPriorDict()
    priors["geocent_time"] = 6.0
    tseries = ifo.strain_data.to_gwpy_timeseries()
    parameters = priors.sample()
    ifo.inject_signal(parameters, waveform_generator=wfg, raise_error=False)
    signal_and_data, meta_data = bilby.gw.detector.inject_signal_into_gwpy_timeseries(
        tseries,
        waveform_generator=wfg,
        parameters=parameters,
        det="H1",
    )

    assert meta_data.keys() == ifo.meta_data.keys()
    assert meta_data["parameters"] == ifo.meta_data["parameters"]
    for key in ["optimal_SNR", "matched_filter_SNR"]:
        assert abs(meta_data[key] - ifo.meta_data[key]) < 1e-5

    asd = FrequencySeries(
        ifo.amplitude_spectral_density_array,
        frequencies=ifo.frequency_array,
        unit="1/Hz",
    )
    whitened_data_1 = signal_and_data.whiten(asd=asd).value
    whitened_data_2 = ifo.whitened_time_domain_strain

    mismatch = 1 - (
        sum(whitened_data_1 * whitened_data_2)
        / sum(whitened_data_1**2)**0.5
        / sum(whitened_data_2**2)**0.5
    )
    assert mismatch < 3e-3
