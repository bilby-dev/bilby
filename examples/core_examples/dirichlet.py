import numpy as np
import pandas as pd
from bilby.core.likelihood import Multinomial
from bilby.core.prior import DirichletPriorDict
from bilby.core.sampler import run_sampler

n_dim = 3
label = "dirichlet_"
priors = DirichletPriorDict(n_dim=n_dim, label=label)

injection_parameters = dict(
    dirichlet_0=1 / 3,
    dirichlet_1=1 / 3,
    dirichlet_2=1 / 3,
)
data = [injection_parameters[label + str(ii)] * 1000 for ii in range(n_dim)]

likelihood = Multinomial(data=data, n_dimensions=n_dim, base=label)

result = run_sampler(
    likelihood=likelihood,
    priors=priors,
    nlive=100,
    label="multinomial",
    injection_parameters=injection_parameters,
)

result.posterior[label + str(n_dim - 1)] = 1 - np.sum(
    [result.posterior[key] for key in priors], axis=0
)
result.plot_corner(parameters=injection_parameters)

samples = priors.sample(10000)
samples[label + str(n_dim - 1)] = 1 - np.sum([samples[key] for key in samples], axis=0)
result.posterior = pd.DataFrame(samples)
result.plot_corner(
    parameters=[key for key in samples], filename="outdir/dirichlet_prior_corner.png"
)
