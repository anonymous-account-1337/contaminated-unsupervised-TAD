import numpy as np
from numpy.linalg import slogdet, solve
from scipy.linalg import sqrtm


def wasserstein_2_gaussian(mu0, cov0, mu1, cov1, eps=1e-6):
    cov0 += eps * np.eye(cov0.shape[0])
    cov1 += eps * np.eye(cov1.shape[0])

    # mean difference term
    delta_mu = mu0 - mu1
    mean_term = np.dot(delta_mu, delta_mu)

    # matrix sqrt term
    cov_sqrt = sqrtm(cov1 @ cov0 @ cov1)

    # numerical stability: ensure result is real
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    trace_term = np.trace(cov0 + cov1 - 2 * cov_sqrt)

    return np.sqrt(mean_term + trace_term)


def kl_gaussian(mu0, cov0, mu1, cov1, eps=1e-6):
    d = mu0.shape[0]

    # Regularize covariances (important in high D)
    cov0 = cov0 + eps * np.eye(d)
    cov1 = cov1 + eps * np.eye(d)

    # log(det(.)) via slogdet for stability
    sign0, logdet0 = slogdet(cov0)
    sign1, logdet1 = slogdet(cov1)

    if sign0 <= 0 or sign1 <= 0:
        raise ValueError("Covariance matrix is not positive definite")

    # Trace term: tr(Sigma1^{-1} Sigma0)
    trace_term = np.trace(solve(cov1, cov0))

    # Quadratic term
    diff = mu1 - mu0
    quad_term = diff.T @ solve(cov1, diff)

    kl = 0.5 * (trace_term + quad_term - d + (logdet1 - logdet0))
    return kl


def kl_symmetric_gaussian(mu0, cov0, mu1, cov1):
    return kl_gaussian(mu0, cov0, mu1, cov1) + kl_gaussian(mu1, cov1, mu0, cov0)
