import random

import numpy as np

from ...core.sampler.proposal import JumpProposal


class SkyLocationWanderJump(JumpProposal):
    """
    Jump proposal for wandering over the sky location. Does a Gaussian step in
    RA and DEC depending on the temperature.
    """

    def __call__(self, sample, **kwargs):
        temperature = 1 / kwargs.get('inverse_temperature', 1.0)
        sigma = np.sqrt(temperature) / 2 / np.pi
        sample['ra'] += random.gauss(0, sigma)
        sample['dec'] += random.gauss(0, sigma)
        return super(SkyLocationWanderJump, self).__call__(sample)


class CorrelatedPolarisationPhaseJump(JumpProposal):
    """
    Correlated polarisation/phase jump proposal. Jumps between degenerate phi/psi regions.
    """

    def __call__(self, sample, **kwargs):
        alpha = sample['psi'] + sample['phase']
        beta = sample['psi'] - sample['phase']

        draw = random.random()
        if draw < 0.5:
            alpha = 3.0 * np.pi * random.random()
        else:
            beta = 3.0 * np.pi * random.random() - 2 * np.pi
        sample['psi'] = (alpha + beta) * 0.5
        sample['phase'] = (alpha - beta) * 0.5
        return super(CorrelatedPolarisationPhaseJump, self).__call__(sample)


class PolarisationPhaseJump(JumpProposal):
    """
    Correlated polarisation/phase jump proposal. Jumps between degenerate phi/psi regions.
    """

    def __call__(self, sample, **kwargs):
        sample['phase'] += np.pi
        sample['psi'] += np.pi / 2
        return super(PolarisationPhaseJump, self).__call__(sample)
