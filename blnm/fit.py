"""Fitting routines.
"""
import numpy as np
import numpy.typing as npt
import typing

from scipy.stats import binom
from scipy import special as scisp

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from . import utils
from . import dist

INTEGRAL_N_SAMPLES = 1000


def _data_log_likelihood(zeroth_order: npt.NDArray) -> float:
    """Compute the data log likelihood.

    Args:
        zeroth_order : ((k mixture, N sample) np.ndarray) the values are
            those updated by the _e_step function.
    Returns:
        float
    """
    return np.sum(np.log(np.sum(zeroth_order, axis=0)))

def _e_step(x_counts: npt.NDArray,
            n_counts: npt.NDArray, 
            coefs: npt.NDArray,
            means: npt.NDArray,
            variance: float,
            k_mixtures: int,
            zeroth_order: npt.NDArray,
            first_order: npt.NDArray,
            second_order: npt.NDArray,
            integral_n_samples: int,
            seed: typing.Any) -> None:
    """E step of EM algorithm.

    Updates the conditional expectation for the zeroth, 
    first and second order equation in place.

    Args:
        x_counts : ((N, ) np.ndarray)
        n_counts : ((N, ) np.ndarray)
        coefs : ((k_mixtures, ) np.ndarray)
        means : ((k_mixtures, ) np.ndarray)
        variance : (float)
        k_mixtures : (int)
        zeroth_order : ((k_mixtures, N) np.ndarray)
        first_order : ((k_mixtures, N) np.ndarray)
        second_order : ((k_mixtures, N) np.ndarray)
        integral_n_samples : (int) number of samples to draw for
            computing expectations
        seed : any seed compatible with numpy random number
            generator object (e.g. numpy.random.default_rng(seed=seed))

    Returns
        None
    """

    rng = np.random.default_rng(seed=seed)

    for k, coef_k in enumerate(coefs):

        for i, (x_i, n_i) in enumerate(zip(x_counts, n_counts)):

            # generate s for computing binomial probabilities
            s = rng.normal(means[k], 
                           np.sqrt(variance),
                           size = integral_n_samples)

            # compute the binomial probabilites
            p = utils.logistic(s)
            # f_X = scisp.binom(n_i, x_i) * p**x_i * (1-p)**(n_i-x_i)
            f_X = binom.pmf(x_i, n_i, p)

            zeroth_order[k, i] = np.mean(f_X) * coef_k
            first_order[k, i] = np.mean(s * f_X) * coef_k
            second_order[k, i] = np.mean(s**2 * f_X) * coef_k


def _m_step(zeroth_order,
           first_order,
           second_order):
    """M step of EM algorithm

    Args:
        zeroth_order : ((k_mixtures, N) np.ndarray)
        first_order : ((k_mixtures, N) np.ndarray)
        second_order : ((k_mixtures, N) np.ndarray)

    Returns:
        coefs : ((k mixtures, ) np.ndarray) 
        means : ((k mixture, ) np.ndarray)
        variance : (float)
    """

    # compute helper values
    beta = np.sum(zeroth_order, axis=0)

    k_mixtures, N = zeroth_order.shape

    coefs = np.zeros(k_mixtures)
    means = np.zeros(k_mixtures)
    variance = 0

    for k in range(k_mixtures):
        # helper value
        N_k = np.sum(zeroth_order[k, :] / beta)

        coefs[k] = N_k / N
        means[k] = np.sum(first_order[k, :] / beta) / N_k
        variance += np.mean(second_order[k, :] / beta) - coefs[k]*means[k]**2

    return coefs, means, variance


def _find_k_from_bins(alt_allele_counts, n_counts):
    ratios = [a / n for a, n in zip(alt_allele_counts, n_counts)]
    ratios.sort()
    bin_counts = [0] * 10
    for r in ratios:
        idx = 9 if r == 1.0 else int(r * 10)
        bin_counts[idx] += 1
    peak_indices = []
    for i in range(1, 9):
        if bin_counts[i] > bin_counts[i - 1] and bin_counts[i] > bin_counts[i + 1]:
            peak_indices.append(i)
    if not peak_indices:
        if bin_counts[0] > bin_counts[1]:
            peak_indices.append(0)
        if bin_counts[9] > bin_counts[8]:
            peak_indices.append(9)
    if not peak_indices:
        peak_indices = [bin_counts.index(max(bin_counts))]
    return len(peak_indices)

def _find_k_gmm(d_reshaped, criterion="AIC"):
    best_k = None
    best_score = np.inf
    for candidate_k in range(1, 5):
        gmm = GaussianMixture(n_components=candidate_k, covariance_type="full", random_state=42)
        gmm.fit(d_reshaped)
        score = gmm.aic(d_reshaped) if criterion.upper() == "AIC" else gmm.bic(d_reshaped)
        if score < best_score:
            best_score = score
            best_k = candidate_k
    return best_k

def _plot_clusters(log_ref_clean, log_alt_clean, labels, sorted_centers):
    plt.figure(figsize=(8, 6))
    plt.scatter(log_ref_clean, log_alt_clean, c=labels, cmap='viridis', alpha=0.5)
    plt.xlabel("log(ref)")
    plt.ylabel("log(alt)")
    plt.title("Scatter Plot of log(ref) vs. log(alt) with Parallel Lines")
    x_vals = np.linspace(log_ref_clean.min(), log_ref_clean.max(), 200)
    for c in sorted_centers:
        y_vals = x_vals + c
        plt.plot(x_vals, y_vals, '--', label=f'y = x + {c:.2f}')
    plt.legend()
    plt.show()

def init_pars(
    alt_allele_counts=None,
    n_counts=None,
    k=None,
    method=None,
    do_plot=False
):
    if method == "partition":
        if k is None:
            raise ValueError("When using method='partition', the number of components k must be specified.")
        coefs = [1.0 / k] * k
        means = [math.log(p / (1 - p)) for p in [(i + 1) / (k + 1) for i in range(k)]]
        return {
            "k": k,
            "means": means,
            "coefs": coefs,
            "variance": 1.0
        }

    if alt_allele_counts is None or n_counts is None:
        raise ValueError("alt_allele_counts and n_counts must be provided.")

    alt_allele_counts = np.asarray(alt_allele_counts, dtype=float)
    n_counts = np.asarray(n_counts, dtype=float)

    if len(alt_allele_counts) != len(n_counts):
        raise ValueError("alt_allele_counts and n_counts must have the same length.")
    if len(alt_allele_counts) == 0:
        raise ValueError("Input arrays must not be empty.")

    for a, n in zip(alt_allele_counts, n_counts):
        if a < 0 or n <= 0:
            raise ValueError("All alt_allele_counts must be >= 0 and all n_counts > 0.")
        if a > n:
            raise ValueError("Each alt_allele_count must be <= the corresponding n_count.")

    ref = (n_counts - alt_allele_counts).astype(float)
    alt = alt_allele_counts.astype(float)
    log_ref = np.log(ref)
    log_alt = np.log(alt)

    mask = np.isfinite(log_ref) & np.isfinite(log_alt)
    log_ref_clean = log_ref[mask]
    log_alt_clean = log_alt[mask]

    if len(log_ref_clean) == 0:
        raise ValueError("No valid (finite) data points remain after log transform.")

    d = log_alt_clean - log_ref_clean
    d_reshaped = d.reshape(-1, 1)

    if k is None:
        if method is None:
            raise ValueError("If k is not specified, you must provide a method ('bins', 'AIC', or 'BIC').")
        if method == "bins":
            k = _find_k_from_bins(alt_allele_counts, n_counts)
        elif method.upper() in ["AIC", "BIC"]:
            k = _find_k_gmm(d_reshaped, criterion=method.upper())
        else:
            raise ValueError("method must be either 'bins', 'AIC', 'BIC', or 'partition'.")

    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
    gmm.fit(d_reshaped)
    labels = gmm.predict(d_reshaped)
    centers = gmm.means_.flatten()
    weights = gmm.weights_.flatten()

    order = np.argsort(centers)
    sorted_centers = centers[order]
    sorted_weights = weights[order]
    coefs = sorted_weights.tolist()

    cluster_variances = np.array([gmm.covariances_[i][0][0] for i in range(k)])
    sorted_var = cluster_variances[order]
    overall_variance = float(np.sum((sorted_weights ** 2) * sorted_var))

    if do_plot:
        _plot_clusters(log_ref_clean, log_alt_clean, labels, sorted_centers)

    return {
        "k": k,
        "means": sorted_centers.tolist(),
        "coefs": coefs,
        "variance": overall_variance
    }


def blnm(x_counts: npt.NDArray, 
         n_counts: npt.NDArray,
         k_mixtures: int, 
         coefs: npt.NDArray | None = None,
         means: npt.NDArray | None = None,
         variance: float | None = None,
         seed: None | int | np.random._generator.Generator = None,
         tolerance:float = 1e-6, 
         max_iter:int = 1000,
         disp: bool = True,
         integral_n_samples: int = INTEGRAL_N_SAMPLES) -> dict:
    """Fit mixture of BLN models.

    Args:
        x_counts : ((N,) np.ndaray) alternative allele specific expression 
            counts
        n_counts : ((N,) np.ndarray) alternative + reference allele 
            specific expression counts
        k_mixtures : (int) > 0 number of mixtures to fit
        coefs: ((k_mixtures,) np.ndarray) 
            The weights for each BLN probability mass funciton in the 
            mixture.  Each coefficient must be greater than 0 and the
            sum of coefficients be 1.  If None, pick coefficients randomly
            subject to our contraints.
        means: ((k_mixtures,) np.ndarray)
            The mean parameter for each BLN probability mass function.
            This can be any real number.  If None, pick coefficients randomly.
        variance: (float)
            A real number greater than zero representing the variance parameter
            of each BLN probability mass function.
        seed : (any input to numpy random Generator object)
        tolerance : (float) criterion for convergence
        max_iter : (int) maximum number of interations
        disp : (bool) print iteration information to standard out
        integral_n_samples : (int) number of samples for computing
            the BLN integral

    Returns:
        dict : 
            coefs: ((k_mixtures,) np.ndarray) mixture fractions
            means: ((k_mixtures,) np.ndarray) BLN mixture means
            variance: (float) variance of all BLN mixture
            log_likelihood: (float)
            converged: (int) 0 if converged 1 otherwise
            converge_message: (string)
            iterations: (int) number of iterations for convergence
    """
    # Validate inputs

    if k_mixtures < 1 or not isinstance(k_mixtures, int):
        raise ValueError("k_mixtures must be an integer greater than 1")
    elif x_counts.shape != n_counts.shape:
        raise ValueError("alt_allele_counts and ref_allele_counts must have"
                        "shape")
    elif x_counts.ndim != 1:
        raise ValueError

    # validate data
    for x_i, n_i in zip(x_counts, n_counts):
        if x_i < 0 or n_i < 0:
            raise ValueError("Counts must be 0 or positive integers.")
        elif x_i > n_i:
            raise ValueError("Alternative allele counts must be less than total counts.")
        #TODO come back to this point
#         elif not isinstance(x_i, int):
#             raise ValueError("Counts must be integers.")
#         elif not isinstance(n_i, int):
#             raise ValueError("Counts must be integers.")


    rng = np.random.default_rng(seed=seed)

    # initialize parameters
    if coefs is None:
        coefs = rng.uniform(low=0.1, high=0.9, size=k_mixtures)
        coefs = coefs / np.sum(coefs)

    # verify coefs
    if (coefs < 0).any():
        raise ValueError("All coefficients must be positive.")
    elif (np.isnan(coefs)).any():
        raise ValueError("All coefficients must be positive.")
    elif (s := np.sum(coefs)) < 0.9999 or s > 1.0001:
        raise ValueError("The sum of coefficients must be 1.")
    elif coefs.size != k_mixtures:
        raise ValueError("The number of coefficients must be"
                        " equal to the number of k_mixtures.")

    if means is None:
        p = rng.uniform(low=0.01, high=0.99, size=k_mixtures)
        means = np.log(p / (1-p))

    if means.size != k_mixtures:
        raise ValueError("The number of means must be"
                        " equal to the number of k_mixtures.")
    elif (np.isnan(means)).any():
        raise ValueError("The number of means must be"
                        " equal to the number of k_mixtures.")


    if variance is None:
        variance = rng.uniform(low=0.1, high=3)

    if variance <= 0 or np.isnan(variance):
        raise ValueError("Variance parameter must be a float greater than zero.")


    # preallocate memory for arrays constructed in the E step
    # each array represents
    # E_s[ s^order f_X(x;s,n) | j^th iteration parameters ]
    zeroth_order = np.zeros(shape=(k_mixtures, len(x_counts)))
    first_order = np.zeros(shape=(k_mixtures, len(x_counts)))
    second_order = np.zeros(shape=(k_mixtures, len(x_counts)))


    _e_step(x_counts, n_counts,
            coefs, means, variance, k_mixtures,
            zeroth_order,
            first_order,
            second_order,
            integral_n_samples,
            rng)

    # store log likelihoods
    log_likelihood = [None, _data_log_likelihood(zeroth_order)]

    # Perform E and M steps until the difference in log-likelihood
    # from iter_n to iter_n +1 iteration is less than tolerance,
    # or maximum number of iterations are reached

    delta = 100
    iter_n = 0

    if disp:
        print("Iter\tlog likelihood")
        print(f"{iter_n:04d}", 
              f"{log_likelihood[1]:0.4f}",
              sep="\t", end="\n")

    while delta >= tolerance and max_iter > iter_n:

        log_likelihood[0] = log_likelihood[1]


        # inplace assignment of order arrays
        _e_step(x_counts, n_counts,
                coefs, means, variance, k_mixtures,
                zeroth_order,
                first_order,
                second_order,
                integral_n_samples,
                rng)

        # sanity check
        # beta = np.sum(zeroth_order, axis=0)
        # tmp = 0
        # for k in range(k_mixtures):
        #     tmp += zeroth_order[k, :] / beta

        # assert np.sum(tmp) == x_counts.size

        coefs, means, variance = _m_step(zeroth_order,
                                        first_order,
                                        second_order)


        log_likelihood[1] = _data_log_likelihood(zeroth_order)

        delta = log_likelihood[1] - log_likelihood[0]

        iter_n += 1

        if disp:
            print(f"{iter_n:04d}", 
                  f"{log_likelihood[1]:0.4f}",
                  sep="\t", end="\n")

    out = {
            "coefs": coefs,
            "means": means,
            "variance": variance,
            "log_likelihood": log_likelihood[1],
            "converged": 0,
            "converge_message": "success",
            "iterations": iter_n
            }

    if iter_n == max_iter:
        out["converged"] = 1
        out["converge_message"] = "max iteration reached without convergence"

    return out


