import matplotlib.pyplot as plt
import numpy as np
import corner

import bilby.gw.prior


mass_1_min = 2
mass_1_max = 50

def condition_function(reference_params, mass_1):
    return dict(mu=reference_params['mu'])
    # return dict(minimum=np.maximum(reference_params['minimum'], mass_1_min / mass_1), maximum=reference_params['maximum'])

# condition_function = lambda reference_params, mass_1: dict(minimum=np.maximum(reference_params['minimum'], mass_1_min / mass_1), maximum=reference_params['maximum'])

mass_1 = bilby.core.prior.PowerLaw(alpha=-2.5, minimum=mass_1_min, maximum=mass_1_max, name='mass_1')
mass_ratio = bilby.core.prior.ConditionalExponential(mu=2, name='mass_ratio',
                                                     condition_func=condition_function)

correlated_dict = bilby.core.prior.ConditionalPriorDict(dictionary=dict(mass_1=mass_1, mass_ratio=mass_ratio))

res = correlated_dict.sample(100000)

plt.hist(res['mass_1'], bins='fd', alpha=0.6, density=True, label='Sampled')
plt.plot(np.linspace(2, 50, 200), correlated_dict['mass_1'].prob(np.linspace(2, 50, 200)), label='Powerlaw prior')
plt.xlabel('$m_1$')
plt.ylabel('$p(m_1)$')
plt.loglog()
plt.legend()
plt.tight_layout()
plt.show()
plt.clf()


plt.hist(res['mass_ratio'], bins='fd', alpha=0.6, density=True, label='Sampled')
plt.xlabel('$q$')
plt.ylabel('$p(q | m_1)$')
plt.loglog()
plt.legend()
plt.tight_layout()
plt.show()
plt.clf()


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


# def correlation_func_a(mu, a=0):
#     return mu + a**2 + 2 * a + 3
#
#
# def correlation_func_b(mu, a=0, b=0):
#     return mu + 0.01 * a**2 + 0.01 * b**2 + 0.01 * a * b + 0.1 * b + 3
#
#
# a = bilby.core.prior.Gaussian(mu=0., sigma=1)
# b = bilby.core.prior.CorrelatedGaussian(mu=0., sigma=1, correlation_func=correlation_func_a)
# c = bilby.core.prior.CorrelatedGaussian(mu=0, sigma=1, correlation_func=correlation_func_b)
#
# correlated_uniform = bilby.core.prior.CorrelatedPriorDict(dictionary=dict(a=a, b=b, c=c))
#
# samples = correlated_uniform.sample(1000000)
#
# samples = np.array([samples['a'], samples['b'], samples['c']]).T
# corner.corner(np.array(samples))
# plt.show()
#
#
# def correlation_func_min_max(extrema_dict, a, b):
#     maximum = extrema_dict['maximum'] + a**b
#     minimum = np.log(b)
#     return minimum, maximum
#
#
# a = bilby.core.prior.Uniform(minimum=0, maximum=1)
# b = bilby.core.prior.Uniform(minimum=1e-6, maximum=1e-1)
# c = bilby.core.prior.CorrelatedUniform(minimum=0, maximum=1, correlation_func=correlation_func_min_max)
#
# correlated_uniform = bilby.core.prior.CorrelatedPriorDict(dictionary=dict(a=a, b=b, c=c))
#
# samples = correlated_uniform.sample(1000000)
# samples = np.array([samples['a'], samples['b'], samples['c']]).T
# corner.corner(np.array(samples))
# plt.show()
#