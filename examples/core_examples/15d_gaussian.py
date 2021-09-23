import bilby
import numpy as np
from bilby.core.likelihood import (
    AnalyticalMultidimensionalBimodalCovariantGaussian,
    AnalyticalMultidimensionalCovariantGaussian,
)

logger = bilby.core.utils.logger

cov = [
    [
        0.045991865933182365,
        -0.005489748382557155,
        -0.01025067223674548,
        0.0020087713726603213,
        -0.0032648855847982987,
        -0.0034218261781145264,
        -0.0037173401838545774,
        -0.007694897715679858,
        0.005260905282822458,
        0.0013607957548231718,
        0.001970785895702776,
        0.006708452591621081,
        -0.005107684668720825,
        0.004402554308030673,
        -0.00334987648531921,
    ],
    [
        -0.005489748382557152,
        0.05478640427684032,
        -0.004786202916836846,
        -0.007930397407501268,
        -0.0005945107515129139,
        0.004858466255616657,
        -0.011667819871670204,
        0.003169780190169035,
        0.006761345004654851,
        -0.0037599761532668475,
        0.005571796842520162,
        -0.0071098291510566895,
        -0.004477773540640284,
        -0.011250694688474672,
        0.007465228985669282,
    ],
    [
        -0.01025067223674548,
        -0.004786202916836844,
        0.044324704403674524,
        -0.0010572820723801645,
        -0.009885693540838514,
        -0.0048321205972943464,
        -0.004576186966267275,
        0.0025107211483955676,
        -0.010126911913571181,
        0.01153595152487264,
        0.005773054728678472,
        0.005286558230422045,
        -0.0055438798694137734,
        0.0044772210361854765,
        -0.00620416958073918,
    ],
    [
        0.0020087713726603213,
        -0.007930397407501268,
        -0.0010572820723801636,
        0.029861342087731065,
        -0.007803477134405363,
        -0.0011466944120756021,
        0.009925736654597632,
        -0.0007664415942207051,
        -0.0057593957402320385,
        -0.00027248233573270216,
        0.003885350572544307,
        0.00022362281488693097,
        0.006609741178005571,
        -0.003292722856107429,
        -0.005873218251875897,
    ],
    [
        -0.0032648855847982987,
        -0.0005945107515129156,
        -0.009885693540838514,
        -0.007803477134405362,
        0.0538403407841302,
        -0.007446654755103316,
        -0.0025216534232170153,
        0.004499568241334517,
        0.009591034277125177,
        0.00008612746932654223,
        0.003386296829505543,
        -0.002600737873367083,
        0.000621148057330571,
        -0.006603857049454523,
        -0.009241221870695395,
    ],
    [
        -0.0034218261781145264,
        0.004858466255616657,
        -0.004832120597294347,
        -0.0011466944120756015,
        -0.007446654755103318,
        0.043746559133865104,
        0.008962713024625965,
        -0.011099652042761613,
        -0.0006620240117921668,
        -0.0012591530037708058,
        -0.006899982952117269,
        0.0019732354732442878,
        -0.002445676747004324,
        -0.006454778807421816,
        0.0033303577606412765,
    ],
    [
        -0.00371734018385458,
        -0.011667819871670206,
        -0.004576186966267273,
        0.009925736654597632,
        -0.0025216534232170153,
        0.008962713024625965,
        0.03664582756831382,
        -0.009470328827284009,
        -0.006213741694945105,
        0.007118775954484294,
        -0.0006741237990418526,
        -0.006003374957986355,
        0.005718636997353189,
        -0.0005191095254772077,
        -0.008466566781233205,
    ],
    [
        -0.007694897715679857,
        0.0031697801901690347,
        0.002510721148395566,
        -0.0007664415942207059,
        0.004499568241334515,
        -0.011099652042761617,
        -0.009470328827284016,
        0.057734267068088,
        0.005521731225009532,
        -0.017008048805405164,
        0.006749693090695894,
        -0.006348460110898,
        -0.007879244727681924,
        -0.005321753837620446,
        0.011126783289057604,
    ],
    [
        0.005260905282822458,
        0.0067613450046548505,
        -0.010126911913571181,
        -0.00575939574023204,
        0.009591034277125177,
        -0.0006620240117921668,
        -0.006213741694945106,
        0.005521731225009532,
        0.04610670018969681,
        -0.010427010812879566,
        -0.0009861561285861987,
        -0.008896020395949732,
        -0.0037627528719902485,
        0.00033704453138913093,
        -0.003173552163182467,
    ],
    [
        0.0013607957548231744,
        -0.0037599761532668475,
        0.01153595152487264,
        -0.0002724823357326985,
        0.0000861274693265406,
        -0.0012591530037708062,
        0.007118775954484294,
        -0.01700804880540517,
        -0.010427010812879568,
        0.05909125052583998,
        0.002192545816395299,
        -0.002057672237277737,
        -0.004801518314458135,
        -0.014065326026662672,
        -0.005619012077114913,
    ],
    [
        0.0019707858957027763,
        0.005571796842520162,
        0.005773054728678472,
        0.003885350572544309,
        0.003386296829505542,
        -0.006899982952117272,
        -0.0006741237990418522,
        0.006749693090695893,
        -0.0009861561285862005,
        0.0021925458163952988,
        0.024417715762416557,
        -0.003037163447600162,
        -0.011173674374382736,
        -0.0008193127407211239,
        -0.007137012700864866,
    ],
    [
        0.006708452591621083,
        -0.0071098291510566895,
        0.005286558230422046,
        0.00022362281488693216,
        -0.0026007378733670806,
        0.0019732354732442886,
        -0.006003374957986352,
        -0.006348460110897999,
        -0.008896020395949732,
        -0.002057672237277737,
        -0.003037163447600163,
        0.04762367868805726,
        0.0008818947598625008,
        -0.0007262691465810616,
        -0.006482422704208912,
    ],
    [
        -0.005107684668720825,
        -0.0044777735406402895,
        -0.005543879869413772,
        0.006609741178005571,
        0.0006211480573305693,
        -0.002445676747004324,
        0.0057186369973531905,
        -0.00787924472768192,
        -0.003762752871990247,
        -0.004801518314458137,
        -0.011173674374382736,
        0.0008818947598624995,
        0.042639958466440225,
        0.0010194948614718209,
        0.0033872675386130637,
    ],
    [
        0.004402554308030674,
        -0.011250694688474675,
        0.004477221036185477,
        -0.003292722856107429,
        -0.006603857049454523,
        -0.006454778807421815,
        -0.0005191095254772072,
        -0.005321753837620446,
        0.0003370445313891318,
        -0.014065326026662679,
        -0.0008193127407211239,
        -0.0007262691465810616,
        0.0010194948614718226,
        0.05244900188599414,
        -0.000256550861960499,
    ],
    [
        -0.00334987648531921,
        0.007465228985669282,
        -0.006204169580739178,
        -0.005873218251875899,
        -0.009241221870695395,
        0.003330357760641278,
        -0.008466566781233205,
        0.011126783289057604,
        -0.0031735521631824654,
        -0.005619012077114915,
        -0.007137012700864866,
        -0.006482422704208912,
        0.0033872675386130632,
        -0.000256550861960499,
        0.05380987317762257,
    ],
]

dim = 15
mean = np.zeros(dim)

label = "multidim_gaussian_unimodal"
outdir = "outdir"

likelihood = AnalyticalMultidimensionalCovariantGaussian(mean, cov)
priors = bilby.core.prior.PriorDict()
priors.update(
    {
        "x{0}".format(i): bilby.core.prior.Uniform(-5, 5, "x{0}".format(i))
        for i in range(dim)
    }
)

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    outdir=outdir,
    label=label,
    check_point_plot=True,
    resume=True,
)

result.plot_corner(parameters={"x{0}".format(i): mean[i] for i in range(dim)})

# The prior is constant and flat, and the likelihood is normalised such that the area under it is one.
# The analytical evidence is then given as 1/(prior volume)

log_prior_vol = np.sum(
    np.log([prior.maximum - prior.minimum for key, prior in priors.items()])
)
log_evidence = -log_prior_vol

sampled_std = [
    np.std(result.posterior[param]) for param in result.search_parameter_keys
]

logger.info("Analytic log evidence: " + str(log_evidence))
logger.info(
    "Sampled log evidence:  "
    + str(result.log_evidence)
    + " +/- "
    + str(result.log_evidence_err)
)

for i, search_parameter_key in enumerate(result.search_parameter_keys):
    logger.info(search_parameter_key)
    logger.info("Expected posterior standard deviation: " + str(likelihood.sigma[i]))
    logger.info("Sampled posterior standard deviation:  " + str(sampled_std[i]))

# BIMODAL distribution

label = "multidim_gaussian_bimodal"
dim = len(cov[0])

mean_1 = 4 * np.sqrt(np.diag(cov))
mean_2 = -4 * np.sqrt(np.diag(cov))

likelihood = AnalyticalMultidimensionalBimodalCovariantGaussian(mean_1, mean_2, cov)
priors = bilby.core.prior.PriorDict()
priors.update(
    {
        "x{0}".format(i): bilby.core.prior.Uniform(-5, 5, "x{0}".format(i))
        for i in range(dim)
    }
)

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    outdir=outdir,
    label=label,
    check_point_plot=True,
    resume=True,
)
result.plot_corner(
    parameters={"x{0}".format(i): mean_1[i] for i in range(dim)},
    filename=outdir + "/multidim_gaussian_bimodal_mode_1",
)
result.plot_corner(
    parameters={"x{0}".format(i): mean_2[i] for i in range(dim)},
    filename=outdir + "/multidim_gaussian_bimodal_mode_2",
)

log_prior_vol = np.sum(
    np.log([prior.maximum - prior.minimum for key, prior in priors.items()])
)
log_evidence = -log_prior_vol
sampled_std_1 = []
sampled_std_2 = []
for param in result.search_parameter_keys:
    samples = np.array(result.posterior[param])
    samples_1 = samples[np.where(samples < 0)]
    samples_2 = samples[np.where(samples > 0)]
    sampled_std_1.append(np.std(samples_1))
    sampled_std_2.append(np.std(samples_2))

logger.info("Analytic log evidence: " + str(log_evidence))
logger.info(
    "Sampled log evidence:  "
    + str(result.log_evidence)
    + " +/- "
    + str(result.log_evidence_err)
)

for i, search_parameter_key in enumerate(result.search_parameter_keys):
    logger.info(search_parameter_key)
    logger.info(
        "Expected posterior standard deviation both modes: " + str(likelihood.sigma[i])
    )
    logger.info(
        "Sampled posterior standard deviation first mode:  " + str(sampled_std_1[i])
    )
    logger.info(
        "Sampled posterior standard deviation second mode:  " + str(sampled_std_2[i])
    )
