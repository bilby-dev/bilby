import array_api_compat as aac
import bilby
import numpy as np


class BackendWaveformGenerator(bilby.gw.waveform_generator.WaveformGenerator):
    """
    A thin wrapper to emulate different backends in the waveform generator.

    This ensures that all frequency arrays that might be used inside the
    source are cast to numpy for compatibility. The outputs are converted
    to the appropriate array type.
    """
    def __init__(self, wfg, xp):
        self.wfg = wfg
        self.xp = xp

    def __getattr__(self, name):
        if name == "xp":
            return self.xp
        return getattr(self.wfg, name)

    def convert_nested_dict(self, data):
        if aac.is_array_api_obj(data):
            return self.xp.asarray(data)
        elif isinstance(data, dict):
            return {key: self.convert_nested_dict(value) for key, value in data.items()}
        else:
            raise ValueError("Input must be an array API object or a dict of such objects.")

    def _strain_from_model(self, model_data_points, model, parameters, *, xp=None):
        model_data_points = np.asarray(model_data_points)
        return super()._strain_from_model(model_data_points, model, parameters)

    def frequency_domain_strain(self, parameters):
        self.wfg.frequency_array = np.asarray(self.wfg.frequency_array)
        if "frequency_nodes" in self.wfg.waveform_arguments:
            self.wfg.waveform_arguments["frequency_nodes"] = np.asarray(
                self.wfg.waveform_arguments["frequency_nodes"]
            )
        wf = self.wfg.__class__.frequency_domain_strain(self, parameters)
        return self.convert_nested_dict(wf)
