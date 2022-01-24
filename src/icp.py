# Copyright 2020 Juan L Gamella

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""This module contains the finite sample implementation of Invariant
Causal Prediction, with a two-sample t-test and f-test to check the
invariance of the conditional distribution.

TODO  BEFORE PUBLISHING:
  - color output by termcolor is not portable to all OSs, so deactivate it

"""

import copy
import torch
import itertools
import scipy.stats
import scipy.linalg
import numpy as np

from scipy.stats import f
from scipy.stats import t
from scipy.stats import ttest_ind as ttest
from functools import reduce
from termcolor import colored
from lifelines import CoxPHFitter

from HOUDINI.Library import Loss
# ---------------------------------------------------------------------
# "Public" API: icp function


def icp_lganm(res, data, method):
    mean_pvalues = [1.]
    var_pvalues = [1.]
    p_val_Wilcoxon = [1.]
    res_envs = list()
    len_cum = 0
    for i in range(data.n_env):
        len_cur = data.targets[i].shape[0]
        mask = np.ones(res.shape[0], dtype=np.bool)
        mask[len_cum:len_cur + len_cum] = False
        res_env = res[~mask]
        res_others = res[mask]
        len_cum += len_cur

        if method == 'icp':
            mean_pvalues.append(t_test(res_env, res_others))
            var_pvalues.append(f_test(res_env, res_others))
        elif method == 'nicp':
            # ranksum relates to unpaired wilcox test
            p_val = scipy.stats.ranksums(res_env, res_others).pvalue
            p_val_Wilcoxon.append(p_val)
            res_envs.append(res_env)
        else:
            raise NotImplementedError()

    if method == 'icp':
        pvalue_mean = min(mean_pvalues) * data.n_env
        pvalue_var = min(var_pvalues) * data.n_env
        pvalue = min(pvalue_mean, pvalue_var) * 2
    elif method == 'nicp':
        coef_Wil = 1 if data.n_env == 2 else data.n_env
        p_val_Wilcoxon = min(p_val_Wilcoxon) * coef_Wil
        p_val_Levene = scipy.stats.levene(*res_envs, center='mean').pvalue
        pvalue = min(p_val_Levene, p_val_Wilcoxon) * 2
    return pvalue, 0


def icp_portec(envs, method):
    envs_fit = envs.drop(['PortecStudy'], axis=1)
    cph = CoxPHFitter()
    cph.fit(envs_fit, duration_col='RFSYears', event_col='RFSstatus')

    env1 = envs.loc[envs['PortecStudy'] == 1]
    env1 = env1.drop(['PortecStudy'], axis=1)
    env2 = envs.loc[envs['PortecStudy'] == 2]
    env2 = env2.drop(['PortecStudy'], axis=1)
    haz_env1 = cph.predict_log_partial_hazard(env1)
    haz_env2 = cph.predict_log_partial_hazard(env2)

    res_env1 = Loss.cox_ph_loss(torch.from_numpy(haz_env1.to_numpy()),
                                torch.from_numpy(env1[['RFSstatus', 'RFSYears']].to_numpy()))
    res_env1 = res_env1.cpu().numpy()

    res_env2 = Loss.cox_ph_loss(torch.from_numpy(haz_env2.to_numpy()),
                                torch.from_numpy(env2[['RFSstatus', 'RFSYears']].to_numpy()))
    res_env2 = res_env2.cpu().numpy()

    mean_pvalues = [1.]
    var_pvalues = [1.]
    p_val_Wilcoxon = [1.]
    res_envs = list()
    if method == 'icp':
        mean_pvalues.append(t_test(res_env1, res_env2))
        var_pvalues.append(f_test(res_env1, res_env2))
    elif method == 'nicp':
        # ranksum relates to unpaired wilcox test
        p_val = scipy.stats.ranksums(res_env1, res_env2).pvalue
        p_val_Wilcoxon.append(p_val)
        res_envs.append(res_env1)
        res_envs.append(res_env2)
    else:
        raise NotImplementedError()

    if method == 'icp':
        pvalue_mean = min(mean_pvalues)
        pvalue_var = min(var_pvalues)
        pvalue = min(pvalue_mean, pvalue_var) * 2
    elif method == 'nicp':
        p_val_Wilcoxon = min(p_val_Wilcoxon)
        p_val_Levene = scipy.stats.levene(*res_envs, center='mean').pvalue
        pvalue = min(p_val_Levene, p_val_Wilcoxon) * 2
    return pvalue, 0


def icp_baseline(environments,
                 target,
                 alpha,
                 max_predictors=None,
                 method=None,
                 dataset=None,
                 stop_early=False):
    """
    ICP on the given target using data from the given environments
    """
    if dataset == 'lganm':
        assert len(environments) > 1
        data = Data(environments, target)
        # Build set of candidates@ coefs
        max_predictors = data.p-1 if max_predictors is None else max_predictors
        base = set(range(data.p))
        base.remove(target)
        candidates = []
        for set_size in range(max_predictors+1):
            candidates += list(itertools.combinations(base, set_size))
    elif dataset == 'portec':
        # remove the column RFSstatus RFSYears PortecStudy
        cau_num = len(environments.columns) - 3
        base = set(range(cau_num))
        candidates = []
        for set_size in range(1, min(max_predictors, cau_num) + 1):
            candidates += list(itertools.combinations(base, set_size))
    else:
        raise NotImplementedError()

    # Evaluate candidates
    accepted = []  # To store the accepted sets
    rejected = []  # To store the sets that were rejected
    mses = []  # To store the MSE of the accepted sets
    S = base
    if dataset == 'lganm':
        X = data.pooled_data()
        Y = data.pooled_targets()
        X2 = X.T.dot(X)
        XY = X.T.dot(Y)
        coefs = np.zeros((X.shape[1], len(candidates)))
        for s_id, s in enumerate(candidates):
            supp = list(s) + [data.p]
            X2_sub = X2[supp, :][:, supp]
            XY_sub = XY[supp]
            coefs[supp, s_id] = np.linalg.solve(X2_sub, XY_sub)
        # print(Y.shape, X.shape, coefs.shape)
        res = np.expand_dims(Y, axis=-1) - X @ coefs
        # print(res.shape)

    for s_id, s in enumerate(candidates):
        # print(s)
        if dataset == 'lganm':
            s = set(s)
            p_value, error = icp_lganm(res[:, s_id], data, method)
        elif dataset == 'portec':
            s_new = s + (-3, -2, -1)
            sub_col = environments.columns[list(s_new)]
            env_sub = environments[sub_col]
            # print(env_sub.columns)
            p_value, error = icp_portec(env_sub, method)
        reject = p_value < alpha
        if reject:
            rejected.append(s)
        else:
            accepted.append(s)
            S = S.intersection(s)
            mses.append(error)
        if len(S) == 0 and stop_early:
            break
    return Result(S, accepted, rejected, mses, None)


def icp(environments, target, alpha, selection='all', max_predictors=None, debug=False, stop_early=False):
    """
    ICP on the given target using data from the given environments
    """
    assert len(environments) > 1
    data = Data(environments, target)
    # Build set of candidates
    if isinstance(selection, list):
        base = reduce(lambda union, s: set.union(union, s), selection, set())
        candidates = selection
    else:
        max_predictors = data.p-1 if max_predictors is None else max_predictors
        base = set(range(data.p))
        base.remove(target)
        candidates = []
        for set_size in range(max_predictors+1):
            candidates += list(itertools.combinations(base, set_size))
    # Evaluate candidates
    accepted = []  # To store the accepted sets
    rejected = []  # To store the sets that were rejected
    mses = []  # To store the MSE of the accepted sets
    S = base
    for s in candidates:
        s = set(s)
        # Find linear coefficients on pooled data
        (beta, error) = regress(s, data)
        assert((beta[list(base.difference(s))] == 0).all())
        p_value = test_hypothesis(beta, data, debug=debug)
        reject = p_value < alpha
        if reject:
            rejected.append(s)
        else:
            accepted.append(s)
            S = S.intersection(s)
            mses.append(error)
        if debug:
            color = "red" if reject else "green"
            beta_str = np.array_str(beta, precision=2)
            set_str = "rejected" if reject else "accepted"
            msg = colored("%s %s" % (s, set_str), color) + \
                " - (p=%0.2f) - S = %s %s MSE: %0.4f" % (p_value, S, beta_str, error)
            print(msg)
        if len(S) == 0 and stop_early:
            break
    return Result(S, accepted, rejected, mses, None)

# Support functions to icp


def test_hypothesis(coefs, data, debug=False):
    """Test hypothesis for a vector of coefficients coefs, using the t-test for the mean
    and f-test for the variances, and returning the p-value

    """
    mean_pvalues = np.zeros(data.n_env)
    var_pvalues = np.zeros(data.n_env)
    #residuals = data.pooled_targets() - data.pooled_data() @ coefs
    for i in range(data.n_env):
        (env_targets, env_data, others_targets, others_data) = data.split(i)
        residuals_env = env_targets - env_data @ coefs
        residuals_others = others_targets - others_data @ coefs
        # residuals_env = residuals[data.idx == i]
        # residuals_others = residuals[data.idx != i]
        mean_pvalues[i] = t_test(residuals_env, residuals_others)
        var_pvalues[i] = f_test(residuals_env, residuals_others)
        assert(mean_pvalues[i] <= 1)
        assert(var_pvalues[i] <= 1)
    # Combine via bonferroni correction
    pvalue_mean = min(mean_pvalues) * data.n_env
    pvalue_var = min(var_pvalues) * data.n_env
    # Return two times the smallest p-value
    return min(pvalue_mean, pvalue_var) * 2


def regress(s, data, pooling=True, debug=False):
    """
    Perform the linear regression of data.target over the variables indexed by s
    """
    supp = list(s) + [data.p]  # support is pred. set + intercept
    if pooling:
        X = data.pooled_data()[:, supp]
        Y = data.pooled_targets()
    coefs = np.zeros(data.p+1)
    coefs[supp] = np.linalg.lstsq(X, Y, None)[0]
    error = 0  # mse(Y, data.pooled_data() @ coefs)
    return coefs, error


def mse(true, pred):
    return np.sum((true - pred)**2) / len(true)


def t_test(X, Y):
    """Return the p-value of the two sample f-test for
    the given sample"""
    result = ttest(X, Y, equal_var=False)
    return result.pvalue


def f_test(X, Y):
    """Return the p-value of the two sample t-test for
    the given sample"""
    X = X[np.isfinite(X)]
    Y = Y[np.isfinite(Y)]
    F = np.var(X, ddof=1) / np.var(Y, ddof=1)
    p = f.cdf(F, len(X)-1, len(Y)-1)
    return 2*min(p, 1-p)


def confidence_intervals(s, coefs, data, alpha):
    """Compute the confidence intervals of the regression coefficients
    (coefs) of a predictor set s, given the level alpha.

    Under Gaussian errors, the confidence intervals are given by
    coefs +/- delta, where

    delta = quantile * variance of residuals @ diag(inv. corr. matrix)

    and variance and corr. matrix of residuals are estimates
    """
    s = list(s)
    supp = s + [data.p]  # Support is pred. set + intercept
    coefs = coefs[supp]
    # Quantile term
    dof = data.n - len(s) - 1
    quantile = t.ppf(1-alpha/2/len(s), dof)
    # Residual variance term
    Xs = data.pooled_data()[:, supp]
    residuals = data.pooled_targets() - Xs @ coefs
    variance = np.var(residuals)
    # Corr. matrix term
    sigma = np.diag(np.linalg.inv(Xs.T @ Xs))
    # Compute interval
    delta = quantile * np.sqrt(variance) * sigma
    return (coefs - delta, coefs + delta)

# ---------------------------------------------------------------------
# Data class and its support functions


class Data():
    """Class to handle access to the dataset. Takes a list of
    environments (each environment is an np array containing the
    observations) and the index of the target.

    Parameters:
      - p: the number of variables
      - n: the total number of samples
      - N: list with number of samples in each environment
      - n_env: the number of environments
      - targets: list with the observations of the target in each environment
      - data: list with the observations of the other vars. in each environment
      - target: the index of the target variable

    """

    def __init__(self, environments, target):
        """Initializes the object by separating the observations of the target
        from the rest of the data, and obtaining the number of
        variables, number of samples per environment and total number
        of samples.

        Arguments:
          - environments: list of np.arrays of dim. (n_e, p), each one
            containing the data of an environment. n_e is the number of
            samples for that environment and p is the number of variables.
          - target: the index of the target variable
        """
        environments = copy.deepcopy(
            environments)  # ensure the stored data is immutable
        self.N = np.array(list(map(len, environments)))
        self.p = environments[0].shape[1]
        self.n = np.sum(self.N)
        self.n_env = len(environments)
        # Extract targets and add a col. of 1s for the intercept
        self.targets = list(map(lambda e: e[:, target], environments))
        self.data = list(map(lambda e: np.hstack(
            [e, np.ones((len(e), 1))]), environments))
        self.target = target
        # Construct an index array
        self.idx = np.zeros(self.n)
        ends = np.cumsum(self.N)
        starts = np.zeros_like(ends)
        starts[1::] = ends[:-1]
        for i, start in enumerate(starts):
            end = ends[i]
            self.idx[start:end] = i

    def pooled_data(self):
        """Returns the observations of all variables except the target,
        pooled."""
        return pool(self.data, 0)

    def pooled_targets(self):
        """Returns all the observations of the target variable, pooled."""
        return pool(self.targets, 1)

    def split(self, i):
        """Splits the dataset into targets/data of environment i and
        targets/data of other environments pooled together."""
        rest_data = [d for k, d in enumerate(self.data) if k != i]
        rest_targets = [t for k, t in enumerate(self.targets) if k != i]
        self.data[i]
        return (self.targets[i], self.data[i], pool(rest_targets, 1), pool(rest_data, 0))


def pool(arrays, axis):
    """Takes a list() of numpy arrays and returns them in an new
    array, stacked along the given axis.
    """
    if len(arrays) == 1:
        return arrays[0]
    else:
        stack_fun = np.vstack if axis == 0 else np.hstack
        return reduce(lambda acc, array: stack_fun([acc, array]), arrays)

# ---------------------------------------------------------------------
# Results class


class Result():
    """Class to hold the estimate produced by ICP and any additional information"""

    def __init__(self, estimate, accepted, rejected, mses, conf_intervals=None):
        # The estimate produced by ICP ie. intersection of accepted sets
        self.estimate = estimate
        self.accepted = accepted  # Accepted sets
        self.rejected = rejected  # Rejected sets
        self.mses = np.array(mses)  # MSE of the accepted sets
        self.conf_intervals = conf_intervals  # Confidence intervals
