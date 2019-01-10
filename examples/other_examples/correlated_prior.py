import matplotlib.pyplot as plt
import numpy as np
import corner

import bilby.gw.prior

# mass_1 = bilby.core.prior.Uniform(5, 100)
# mass_2 = bilby.gw.prior.CorrelatedSecondaryMassPrior(minimum=5, maximum=100)
#
# correlated_priors = bilby.core.prior.CorrelatedPriorDict(dictionary=dict(mass_1=mass_1, mass_2=mass_2))
#
# samples = correlated_priors.sample(10)
#
# primary_masses = samples['mass_1']
# secondary_masses = samples['mass_2']
# for i in range(len(primary_masses)):
#     if primary_masses[i] > secondary_masses[i]:
#         print('True')
#     else:
#         print('False')
#
# sample = dict(mass_1=25, mass_2=20)
# print(correlated_priors.prob(sample))


def correlation_func_a(mu, a=0):
    return mu + a**2 + 2*a + 3


def correlation_func_b(mu, a=0, b=0):
    return mu + 0.01 * a**2 + 0.01 * b**2 + 0.01 * a * b + 0.1 * b + 3


a = bilby.core.prior.Gaussian(mu=0., sigma=1)
b = bilby.core.prior.CorrelatedGaussian(mu=0., sigma=1, correlation_func=correlation_func_a)
c = bilby.core.prior.CorrelatedGaussian(mu=0, sigma=1, correlation_func=correlation_func_b)

correlated_uniform = bilby.core.prior.CorrelatedPriorDict(dictionary=dict(a=a, b=b, c=c))

samples = correlated_uniform.sample(1000000)

samples = np.array([samples['a'], samples['b'], samples['c']]).T
corner.corner(np.array(samples))
plt.show()


a = bilby.core.prior.Uniform(minimum=0, maximum=1)
b = bilby.core.prior.CorrelatedUniform(minimum=0, maximum=1, correlation_func=correlation_func_a)
c = bilby.core.prior.CorrelatedUniform(minimum=0, maximum=1, correlation_func=correlation_func_b)

correlated_uniform = bilby.core.prior.CorrelatedPriorDict(dictionary=dict(a=a, b=b, c=c))

samples = correlated_uniform.sample(1000000)

samples = np.array([samples['a'], samples['b'], samples['c']]).T
corner.corner(np.array(samples))
plt.show()
