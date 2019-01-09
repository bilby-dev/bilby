import bilby
import bilby.gw.prior

mass_1 = bilby.core.prior.Uniform(5, 100)
mass_2 = bilby.gw.prior.CorrelatedSecondaryMassPrior(minimum=5, maximum=100)

correlated_priors = bilby.core.prior.CorrelatedPriorDict(dictionary=dict(mass_1=mass_1, mass_2=mass_2))

samples = correlated_priors.sample(100)

primary_masses = samples['mass_1']
secondary_masses = samples['mass_2']
for i in range(len(primary_masses)):
    if primary_masses[i] < secondary_masses[i]:
        print('False')
        break
    else:
        print('True')
