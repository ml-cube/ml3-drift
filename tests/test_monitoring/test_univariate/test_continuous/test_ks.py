import numpy as np

from ml3_drift.monitoring.univariate.continuous.ks import KSAlgorithm


def test_ks():
    """This test defines two univariate Gaussian distributions"""
    rng = np.random.default_rng()

    mu_0, sigma_0 = 1.4, 0.4
    mu_1, sigma_1 = 2.4, 0.5

    alg = KSAlgorithm(comparison_size=100)

    alg.fit(rng.normal(mu_0, sigma_0, size=(300, 1)))

    # Expecting no drift
    output = alg.detect(rng.normal(mu_0, sigma_0, size=(300, 1)))
    assert all([not elem.drift_detected for elem in output])

    # Adding drifted data
    # Drift will be detected after the comparison data will be filled
    # with data from new distribution
    output = alg.detect(rng.normal(mu_1, sigma_1, size=(300, 1)))
    assert any([elem.drift_detected for elem in output])
